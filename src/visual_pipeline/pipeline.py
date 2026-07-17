"""Orchestrate OCR, icon matching, and optional VLM query generation."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from PIL import Image

from candidate_queries.base import CandidateQueryError
from candidate_queries.factory import create_candidate_query_generator
from icon_matching.candidate_generation.base import CandidateGenerationError
from icon_matching.candidate_generation.pipeline import IconCandidateGenerator
from icon_matching.matching.base import IconMatchingError
from icon_matching.matching.detection import DetectionScoreName, select_detection_score
from icon_matching.matching.pipeline import IconLibraryMatcher
from ocr.base import OcrError
from ocr.factory import create_ocr_engine
from ocr.filtering import filter_detections_by_confidence

from .config import ApplicationSettings
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
    ) -> PipelineResult:
        """Run enabled signals, optionally generate queries, and return the inspection artifact."""
        image_file = Path(image_path).expanduser().resolve()
        normalized_query = textual_query.strip() if textual_query and textual_query.strip() else None
        input_payload = {"image_path": str(image_file), "textual_query": normalized_query}
        ocr_signal = self._run_ocr(image_file)
        icon_signal, candidate_mapping, retained_crops = self._run_icon_matching(
            image_file, save_icon_crops, output_directory
        )
        signals = {"ocr": ocr_signal, "icons": icon_signal}
        evidence_block = _render_evidence_block(ocr_signal, icon_signal)
        vlm_input = {
            "textual_query": normalized_query,
            "auxiliary_visual_evidence": evidence_block,
        }
        output = self._run_vlm(image_file, normalized_query, evidence_block)
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
    ) -> tuple[dict[str, Any], dict[str, str], list[dict[str, Any]]]:
        """Match ordered candidate regions while retaining crops only when requested."""
        settings = self.settings.pipeline.icon_matching
        if not settings.enabled:
            return {"enabled": False, "detected_icon_names": []}, {}, []
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
                }
                matched.append(record)
                if retain:
                    retained.append({**record, "crop_path": str(crop_path)})
        return matched, retained

    def _run_vlm(self, image_path: Path, textual_query: str | None, evidence_block: str) -> dict[str, Any]:
        """Generate retrieval queries only when the VLM component is enabled."""
        settings = self.settings.candidate_queries
        if not settings.enabled:
            return {"vlm_enabled": False, "retrieval_queries": []}
        try:
            result = create_candidate_query_generator(settings).generate(
                image_path, textual_query, visual_context=evidence_block
            )
        except (CandidateQueryError, ValueError) as error:
            raise VisualPipelineError(f"Enabled VLM stage failed: {error}") from error
        return {
            "vlm_enabled": True,
            "generator": result.generator.model_dump(),
            "retrieval_queries": result.retrieval_queries,
            "reasoning_summary": result.reasoning_summary,
            "processing_time_seconds": result.processing_time_seconds,
            "metadata": result.metadata,
        }


def _text_reading_order(detection: Any) -> tuple[float, float]:
    """Sort a detected OCR polygon from top-left to bottom-right."""
    if not detection.polygon:
        return (float("inf"), float("inf"))
    return (min(point[1] for point in detection.polygon), min(point[0] for point in detection.polygon))


def _render_evidence_block(ocr_signal: dict[str, Any], icon_signal: dict[str, Any]) -> str:
    """Render the concise, coordinate-free evidence block appended to the VLM prompt."""
    lines = ["OCR-visible text:"]
    lines.extend(f"- {text}" for text in ocr_signal["visible_text"])
    if not ocr_signal["visible_text"]:
        lines.append("- None detected or OCR disabled.")
    lines.extend(["", "Detected icons (top-left to bottom-right):"])
    lines.extend(f"- {name}" for name in icon_signal["detected_icon_names"])
    if not icon_signal["detected_icon_names"]:
        lines.append("- None detected or icon matching disabled.")
    return "\n".join(lines)
