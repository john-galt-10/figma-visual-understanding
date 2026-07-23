"""Orchestrate OCR, icon matching, and optional VLM query generation."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from PIL import Image

from candidate_queries.base import CandidateQueryError, CandidateQueryGenerator, build_user_prompt
from candidate_queries.context import (
    CONTEXT_SYSTEM_INSTRUCTION,
    PreparedContextInput,
    prepare_context_input,
)
from candidate_queries.factory import create_candidate_query_generator
from candidate_queries.models import FocusBox
from icon_matching.candidate_generation.base import CandidateGenerationError
from icon_matching.candidate_generation.pipeline import IconCandidateGenerator
from icon_matching.matching.base import IconMatchingError
from icon_matching.matching.detection import DetectionScoreName, select_detection_score
from icon_matching.matching.pipeline import IconLibraryMatcher
from ocr.base import OcrError
from ocr.factory import create_ocr_engine
from ocr.filtering import filter_detections_by_confidence

from .config import ApplicationSettings
from .input_modes import VlmImageInput, create_input_mode_strategy
from .models import PipelineResult


class VisualPipelineError(RuntimeError):
    """Raised when a configured visual-signal stage cannot complete."""


class VisualSignalPipeline:
    """Build coordinate-free VLM evidence from selected screenshot-analysis stages."""

    def __init__(self, settings: ApplicationSettings) -> None:
        """Store validated settings for one pipeline configuration."""
        self.settings = settings

    def run(
        self,
        image_path: str | Path,
        textual_query: str | None = None,
        save_icon_crops: bool = False,
        output_directory: str | Path | None = None,
        context_image_path: str | Path | None = None,
        focus_bbox: FocusBox | None = None,
    ) -> PipelineResult:
        """Run enabled signals, optionally generate queries, and return the inspection artifact."""
        image_file = Path(image_path).expanduser().resolve()
        normalized_query = textual_query.strip() if textual_query and textual_query.strip() else None
        try:
            prepared_context = prepare_context_input(
                context_image_path,
                focus_bbox,
                output_directory,
            )
        except CandidateQueryError as error:
            raise VisualPipelineError(f"Unable to prepare context input: {error}") from error
        input_payload = {"image_path": str(image_file), "textual_query": normalized_query}
        if prepared_context is not None:
            input_payload["context_image_path"] = prepared_context.context_input.source_image.path
            input_payload["focus_bbox"] = (
                prepared_context.context_input.focus_bbox.model_dump()
                if prepared_context.context_input.focus_bbox is not None
                else None
            )
        ocr_signal = self._run_ocr(image_file)
        icon_signal, candidate_mapping, retained_crops, matches = self._run_icon_matching(
            image_file, save_icon_crops, output_directory
        )
        input_mode = self.settings.candidate_queries.input_mode
        if input_mode != "vanilla" and not self.settings.pipeline.icon_matching.enabled:
            raise VisualPipelineError(
                f"input_mode '{input_mode}' requires pipeline.icon_matching.enabled: true."
            )
        try:
            mode_selection = create_input_mode_strategy(input_mode).prepare(
                image_file,
                ocr_signal,
                [match for match in matches if match["detected"]],
                Path(output_directory).expanduser().resolve() if output_directory is not None else None,
                self.settings.pipeline.icon_matching.visualization.model_dump(mode="json"),
            )
        except ValueError as error:
            raise VisualPipelineError(f"Unable to prepare input_mode '{input_mode}': {error}") from error
        submitted_images, input_description = self._contextualize_images(
            image_file,
            mode_selection.images,
            mode_selection.input_description,
            prepared_context,
        )
        signals = {"ocr": ocr_signal, "icons": icon_signal}
        vlm_input = {
            "textual_query": normalized_query,
            "input_mode": mode_selection.input_mode,
            "images": [
                {"role": image.role, "path": str(image.path)} for image in submitted_images
            ],
            "annotated_image_path": (
                str(mode_selection.annotated_image_path)
                if mode_selection.annotated_image_path is not None else None
            ),
            "numbered_icon_mapping": mode_selection.numbered_icon_mapping,
            "input_description": input_description,
            "auxiliary_visual_evidence": mode_selection.evidence_block,
            "system_prompt": self.settings.candidate_queries.system_instruction,
            "user_prompt": build_user_prompt(
                normalized_query,
                mode_selection.evidence_block,
                input_description,
            ),
        }
        if prepared_context is not None:
            vlm_input["context_input"] = self._context_artifact(
                image_file,
                prepared_context,
                submitted_images,
                input_description,
                mode_selection.evidence_block,
                f"{self.settings.candidate_queries.system_instruction}\n\n{CONTEXT_SYSTEM_INSTRUCTION}",
            )
        output = self._run_vlm(
            submitted_images,
            normalized_query,
            mode_selection.evidence_block,
            input_description,
            context_image_path,
            focus_bbox,
            output_directory,
            prepared_context,
        )
        return PipelineResult(
            input=input_payload,
            signals=signals,
            vlm_input=vlm_input,
            output=output,
            icon_candidate_to_detected_name=candidate_mapping,
            retained_icon_crops=retained_crops,
        )

    def _run_ocr(self, image_path: Path) -> dict[str, Any]:
        """Scan OCR text and normalize it into screenshot reading order."""
        settings = self.settings.pipeline.ocr
        if not settings.enabled:
            return {"enabled": False, "visible_text": []}
        try:
            scan = create_ocr_engine(settings.engine, language=settings.language).scan(image_path)
        except (OcrError, ValueError) as error:
            raise VisualPipelineError(f"Enabled OCR stage failed: {error}") from error
        accepted_detections = filter_detections_by_confidence(
            scan.detections, settings.detection_threshold
        )
        detections = sorted(accepted_detections, key=_text_reading_order)
        return {
            "enabled": True,
            "engine": scan.engine,
            "visible_text": [detection.text for detection in detections],
            "detection_threshold": settings.detection_threshold,
            "detection_count": scan.detection_count,
            "accepted_detection_count": len(accepted_detections),
            "rejected_detection_count": scan.detection_count - len(accepted_detections),
            "processing_time_seconds": scan.processing_time_seconds,
        }

    def _run_icon_matching(
        self, image_path: Path, save_icon_crops: bool, output_directory: str | Path | None
    ) -> tuple[dict[str, Any], dict[str, str], list[dict[str, Any]], list[dict[str, Any]]]:
        """Match ordered candidates and return both inspectable and visual-match records."""
        settings = self.settings.pipeline.icon_matching
        if not settings.enabled:
            return {"enabled": False, "detected_icon_names": []}, {}, [], []
        try:
            candidates = IconCandidateGenerator(settings).generate(image_path, export_crops=False)
            matcher = IconLibraryMatcher(settings)
        except (CandidateGenerationError, IconMatchingError, ValueError) as error:
            raise VisualPipelineError(f"Enabled icon-matching stage failed: {error}") from error
        persistent_directory = None
        if save_icon_crops:
            if output_directory is None:
                raise VisualPipelineError("An output directory is required when saving icon crops.")
            persistent_directory = Path(output_directory).expanduser().resolve() / "icon_crops"
            persistent_directory.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix="figma-icon-crops-") as temporary_directory:
            crop_directory = persistent_directory or Path(temporary_directory)
            matches, retained = self._match_candidates(
                image_path,
                candidates.candidates,
                matcher,
                crop_directory,
                save_icon_crops,
                settings.matching.detection_score,
                settings.matching.detection_threshold,
            )
        accepted_matches = [match for match in matches if match["detected"]]
        names = [match["detected_icon_name"] for match in accepted_matches]
        mapping = (
            {match["candidate_id"]: match["detected_icon_name"] for match in accepted_matches}
            if save_icon_crops
            else {}
        )
        return (
            {
                "enabled": True,
                "detected_icon_names": names,
                "candidate_count": len(matches),
                "accepted_candidate_count": len(accepted_matches),
                "rejected_candidate_count": len(matches) - len(accepted_matches),
                "detection_score": settings.matching.detection_score,
                "detection_threshold": settings.matching.detection_threshold,
                "candidate_generation_processing_time_seconds": candidates.processing_time_seconds,
            },
            mapping,
            retained,
            matches,
        )

    @staticmethod
    def _match_candidates(
        image_path: Path,
        candidates: list[Any],
        matcher: IconLibraryMatcher,
        crop_directory: Path,
        retain: bool,
        detection_score_name: DetectionScoreName,
        detection_threshold: float,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Materialize crops, threshold top matches, and retain requested inspection evidence."""
        matched: list[dict[str, Any]] = []
        retained: list[dict[str, Any]] = []
        with Image.open(image_path) as image:
            source = image.convert("RGB")
            for candidate in candidates:
                box = candidate.crop_bbox
                crop_path = crop_directory / f"{candidate.id}.png"
                source.crop((box.x, box.y, box.right, box.bottom)).save(crop_path, format="PNG")
                try:
                    result = matcher.match(crop_path, top_k=1)
                except (IconMatchingError, ValueError) as error:
                    raise VisualPipelineError(
                        f"Enabled icon-matching stage failed for {candidate.id}: {error}"
                    ) from error
                if not result.results:
                    continue
                top_match = result.results[0]
                try:
                    threshold_score = select_detection_score(top_match, detection_score_name)
                except IconMatchingError as error:
                    raise VisualPipelineError(
                        f"Enabled icon-matching stage failed for {candidate.id}: {error}"
                    ) from error
                detected = threshold_score >= detection_threshold
                record = {
                    "candidate_id": candidate.id,
                    "detected": detected,
                    "detected_icon_name": top_match.label if detected else None,
                    "top_match_label": top_match.label,
                    "detection_score": detection_score_name,
                    "threshold_score": threshold_score,
                    "detection_threshold": detection_threshold,
                    "final_score": top_match.final_score,
                    "primary_score": top_match.primary_score,
                    "secondary_score": top_match.tie_breaker_score,
                    "content_bbox": candidate.content_bbox,
                }
                matched.append(record)
                if retain:
                    retained.append({**record, "crop_path": str(crop_path)})
        return matched, retained

    def _run_vlm(
        self,
        images: list[VlmImageInput],
        textual_query: str | None,
        evidence_block: str,
        input_description: str,
        context_image_path: str | Path | None,
        focus_bbox: FocusBox | None,
        output_directory: str | Path | None,
        prepared_context: PreparedContextInput | None,
    ) -> dict[str, Any]:
        """Generate retrieval queries only when the VLM component is enabled."""
        settings = self.settings.candidate_queries
        if not settings.enabled:
            return {"vlm_enabled": False, "retrieval_queries": []}
        try:
            result = create_candidate_query_generator(settings).generate(
                images[0].path,
                textual_query,
                visual_context=evidence_block,
                input_description=input_description,
                additional_image_paths=[
                    image.path for image in (images[2:] if prepared_context is not None else images[1:])
                ],
                context_image_path=context_image_path,
                focus_bbox=focus_bbox,
                context_artifact_directory=output_directory,
                prepared_context_input=prepared_context,
            )
        except (CandidateQueryError, ValueError) as error:
            raise VisualPipelineError(f"Enabled VLM stage failed: {error}") from error
        output = {
            "vlm_enabled": True,
            "generator": result.generator.model_dump(),
            "retrieval_queries": result.retrieval_queries,
            "reasoning_summary": result.reasoning_summary,
            "processing_time_seconds": result.processing_time_seconds,
            "metadata": result.metadata,
        }
        if result.screen_context is not None:
            output["screen_context"] = result.screen_context.model_dump()
        return output

    @staticmethod
    def _contextualize_images(
        focused_image: Path,
        mode_images: list[VlmImageInput],
        mode_input_description: str,
        prepared_context: PreparedContextInput | None,
    ) -> tuple[list[VlmImageInput], str]:
        """Insert optional context after the focused target while preserving crop-only mode behavior."""
        if prepared_context is None:
            return mode_images, mode_input_description
        mode_specific_images = [image for image in mode_images if image.path != focused_image]
        context_role = (
            "annotated_context_screenshot"
            if prepared_context.context_input.annotated_context_image_path is not None
            else "context_screenshot"
        )
        images = [
            VlmImageInput(path=focused_image, role="focused_screenshot"),
            VlmImageInput(path=prepared_context.submission_path, role=context_role),
            *mode_specific_images,
        ]
        focus_description = (
            "The second image is a full-screen context screenshot with a subtle rectangle "
            "showing where the focused target appears."
            if prepared_context.context_input.focus_bbox is not None
            else "The second image is a full-screen context screenshot."
        )
        overlay_description = (
            " The third image, when present, is a detection overlay for the focused target. "
            "Its numbered boxes map to detected icon names in Auxiliary visual evidence."
            if mode_specific_images
            else ""
        )
        return (
            images,
            "Images are ordered: the first image is the focused screenshot and primary target "
            f"for retrieval queries. {focus_description} Use it only to clarify surrounding "
            f"state or panels, not as a separate target.{overlay_description}",
        )

    @staticmethod
    def _context_artifact(
        focused_image: Path,
        prepared_context: PreparedContextInput,
        images: list[VlmImageInput],
        input_description: str,
        evidence_block: str,
        effective_system_instruction: str,
    ) -> dict[str, Any]:
        """Build the stable context metadata object retained in each context-enabled artifact."""
        return {
            "focused_source_path": str(focused_image),
            "context_source_path": prepared_context.context_input.source_image.path,
            "context_dimensions": {
                "width": prepared_context.context_input.source_image.width,
                "height": prepared_context.context_input.source_image.height,
            },
            "focus_bbox": (
                prepared_context.context_input.focus_bbox.model_dump()
                if prepared_context.context_input.focus_bbox is not None
                else None
            ),
            "annotated_context_image_path": prepared_context.context_input.annotated_context_image_path,
            "submitted_images": [
                {
                    "role": image.role,
                    "path": str(image.path),
                    "dimensions": {
                        "width": CandidateQueryGenerator.read_image_metadata(image.path).width,
                        "height": CandidateQueryGenerator.read_image_metadata(image.path).height,
                    },
                }
                for image in images
            ],
            "effective_system_instruction": effective_system_instruction,
            "input_description": input_description,
            "auxiliary_visual_evidence": evidence_block,
        }


def _text_reading_order(detection: Any) -> tuple[float, float]:
    """Sort a detected OCR polygon from top-left to bottom-right."""
    if not detection.polygon:
        return (float("inf"), float("inf"))
    return (min(point[1] for point in detection.polygon), min(point[0] for point in detection.polygon))
