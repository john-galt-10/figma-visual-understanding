"""Orchestration, merging, filtering, and OCR suppression for icon proposals."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

from .base import CandidateGenerationError
from .config import IconMatchingSettings
from .detectors import CannyContourDetector, MorphologyConnectedComponentsDetector, _cv2
from .models import BoundingBox, CandidateGenerationResult, IconCandidate, RegionProposal


class IconCandidateGenerator:
    """Generate matcher-ready icon region candidates from one raster screenshot."""

    def __init__(self, settings: IconMatchingSettings) -> None:
        """Configure all independently switchable stages for future runs."""
        self.settings = settings

    def generate(self, image_path: str | Path, export_crops: bool) -> CandidateGenerationResult:
        """Run enabled detectors and return final candidates without writing files."""
        path = Path(image_path).expanduser().resolve()
        image_metadata, color_image, grayscale_image = self._load_image(path)
        started_at = perf_counter()
        proposals = self._detect(grayscale_image)
        (
            filtered_proposals,
            geometric_rejected_before_merging,
            square_shape_rejected_before_merging,
        ) = self._filter_geometry(proposals)
        merged_proposals = self._merge(filtered_proposals)
        (
            final_filtered_proposals,
            geometric_rejected_after_merging,
            square_shape_rejected_after_merging,
        ) = self._filter_geometry(merged_proposals)
        retained_proposals, ocr_rejected, ocr_metadata = self._suppress_ocr(final_filtered_proposals, path)
        candidates = self._to_candidates(retained_proposals, color_image.shape[1], color_image.shape[0])
        return CandidateGenerationResult(
            schema_version="1.0",
            image=image_metadata,
            candidates=candidates,
            processing_time_seconds=round(perf_counter() - started_at, 6),
            components={
                "preprocessing": self.settings.preprocessing.enabled,
                "morphology": self.settings.detectors.morphology.enabled,
                "contours": self.settings.detectors.contours.enabled,
                "merging": self.settings.merging.enabled,
                "geometric_filters": self.settings.filters.enabled,
                "ocr_suppression": self.settings.ocr_suppression.enabled,
            },
            counts={
                "raw_proposals": len(proposals),
                "after_initial_geometric_filtering": len(filtered_proposals),
                "geometric_rejected_before_merging": geometric_rejected_before_merging,
                "square_shape_rejected_before_merging": square_shape_rejected_before_merging,
                "after_merging": len(merged_proposals),
                "geometric_rejected_after_merging": geometric_rejected_after_merging,
                "square_shape_rejected_after_merging": square_shape_rejected_after_merging,
                "ocr_rejected": ocr_rejected,
                "final_candidates": len(candidates),
            },
            crop_files_exported=export_crops,
            configuration=self.settings.model_dump(mode="json"),
            ocr=ocr_metadata,
        )

    def _load_image(self, path: Path) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
        """Validate, load, and optionally smooth a supported source image."""
        if not path.is_file():
            raise CandidateGenerationError(f"Image file does not exist: {path}")
        try:
            with Image.open(path) as image:
                image_format = (image.format or "UNKNOWN").upper()
                if image_format not in {"PNG", "JPEG", "WEBP", "BMP", "TIFF"}:
                    raise CandidateGenerationError(f"Unsupported image format: {image_format}")
                rgb_image = np.array(image.convert("RGB"))
        except UnidentifiedImageError as error:
            raise CandidateGenerationError(f"Input is not a readable image: {path}") from error
        cv2 = _cv2()
        color_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        grayscale_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        if self.settings.preprocessing.enabled and self.settings.preprocessing.blur_kernel_size > 1:
            size = self.settings.preprocessing.blur_kernel_size
            grayscale_image = cv2.GaussianBlur(grayscale_image, (size, size), 0)
        return (
            {
                "path": str(path),
                "filename": path.name,
                "format": image_format,
                "width": int(color_image.shape[1]),
                "height": int(color_image.shape[0]),
            },
            color_image,
            grayscale_image,
        )

    def _detect(self, grayscale_image: np.ndarray) -> list[RegionProposal]:
        """Run every enabled proposal detector against the same normalized image."""
        detectors = []
        if self.settings.detectors.morphology.enabled:
            detectors.append(MorphologyConnectedComponentsDetector(self.settings.detectors.morphology))
        if self.settings.detectors.contours.enabled:
            detectors.append(CannyContourDetector(self.settings.detectors.contours))
        return [proposal for detector in detectors for proposal in detector.detect(grayscale_image)]

    def _merge(self, proposals: list[RegionProposal]) -> list[RegionProposal]:
        """Merge highly overlapping proposals from different detectors only."""
        if not self.settings.merging.enabled:
            return proposals
        remaining = list(proposals)
        merged: list[RegionProposal] = []
        while remaining:
            current = remaining.pop(0)
            group = [current]
            changed = True
            while changed:
                changed = False
                for proposal in list(remaining):
                    if any(
                        _can_merge(proposal, member)
                        and _intersection_over_union(proposal.bbox, member.bbox)
                        >= self.settings.merging.overlap_threshold
                        for member in group
                    ):
                        group.append(proposal)
                        remaining.remove(proposal)
                        changed = True
            merged.append(_merge_group(group))
        return merged

    def _filter_geometry(self, proposals: list[RegionProposal]) -> tuple[list[RegionProposal], int, int]:
        """Remove regions outside configured icon-like dimensions and compactness."""
        if not self.settings.filters.enabled:
            return proposals, 0, 0
        settings = self.settings.filters
        retained = []
        square_shape_rejected = 0
        for proposal in proposals:
            bbox = proposal.bbox
            aspect_ratio = bbox.width / bbox.height
            compactness = min(bbox.width, bbox.height) / max(bbox.width, bbox.height)
            passes_base_geometry = (
                settings.minimum_width <= bbox.width <= settings.maximum_width
                and settings.minimum_height <= bbox.height <= settings.maximum_height
                and settings.minimum_aspect_ratio <= aspect_ratio <= settings.maximum_aspect_ratio
                and compactness >= settings.minimum_compactness
            )
            passes_square_shape = (
                not settings.square_shape.enabled
                or compactness >= settings.square_shape.minimum_compactness
            )
            if passes_base_geometry and passes_square_shape:
                proposal.evidence["geometry"] = {
                    "aspect_ratio": round(aspect_ratio, 4),
                    "compactness": round(compactness, 4),
                }
                retained.append(proposal)
            elif passes_base_geometry and not passes_square_shape:
                square_shape_rejected += 1
        return retained, len(proposals) - len(retained), square_shape_rejected

    def _suppress_ocr(
        self, proposals: list[RegionProposal], image_path: Path
    ) -> tuple[list[RegionProposal], int, dict[str, Any] | None]:
        """Optionally run OCR and remove candidates substantially occupied by text."""
        if not self.settings.ocr_suppression.enabled:
            return proposals, 0, None
        try:
            from ocr.factory import create_ocr_engine
            from ocr.base import OcrError
        except ImportError as error:
            raise CandidateGenerationError("The local OCR package is unavailable") from error
        settings = self.settings.ocr_suppression
        try:
            scan = create_ocr_engine(settings.engine, language=settings.language).scan(image_path)
        except (OcrError, ValueError) as error:
            raise CandidateGenerationError(f"OCR suppression failed: {error}") from error
        text_boxes = [
            _polygon_bbox(detection.polygon)
            for detection in scan.detections
            if (detection.confidence is None or detection.confidence >= settings.minimum_text_confidence)
        ]
        retained = [
            proposal
            for proposal in proposals
            if not any(_candidate_coverage(proposal.bbox, text_box) >= settings.candidate_text_overlap_threshold for text_box in text_boxes)
        ]
        return (
            retained,
            len(proposals) - len(retained),
            {
                "engine": scan.engine,
                "detection_count": scan.detection_count,
                "text_boxes_used": len(text_boxes),
                "processing_time_seconds": scan.processing_time_seconds,
            },
        )

    def _to_candidates(self, proposals: list[RegionProposal], image_width: int, image_height: int) -> list[IconCandidate]:
        """Assign stable IDs, padded crop boxes, and scores to final proposals."""
        ordered = sorted(proposals, key=lambda proposal: (proposal.bbox.y, proposal.bbox.x, proposal.bbox.width, proposal.bbox.height))
        return [
            IconCandidate(
                id=f"candidate-{index:03d}",
                content_bbox=proposal.bbox,
                crop_bbox=_padded_box(proposal.bbox, self.settings.crops.padding_pixels, image_width, image_height),
                detector_sources=sorted(_detector_evidence(proposal)),
                detector_evidence=_detector_evidence(proposal),
                proposal_score=round(_proposal_score(proposal), 4),
            )
            for index, proposal in enumerate(ordered, start=1)
        ]


def _merge_group(group: list[RegionProposal]) -> RegionProposal:
    """Combine a connected merge group into its union box and evidence map."""
    x = min(proposal.bbox.x for proposal in group)
    y = min(proposal.bbox.y for proposal in group)
    right = max(proposal.bbox.right for proposal in group)
    bottom = max(proposal.bbox.bottom for proposal in group)
    evidence: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for proposal in group:
        nested = proposal.evidence.get("detector_evidence")
        if isinstance(nested, dict):
            for detector, entries in nested.items():
                evidence[detector].extend(entries)
        else:
            evidence[proposal.detector].append(proposal.evidence)
    return RegionProposal(
        bbox=BoundingBox(x=x, y=y, width=right - x, height=bottom - y),
        detector="merged",
        evidence={"detector_evidence": dict(evidence)},
    )


def _intersection_over_union(first: BoundingBox, second: BoundingBox) -> float:
    """Return standard IoU, rejecting small boxes merely contained in a large one."""
    intersection = _intersection_area(first, second)
    union = first.area + second.area - intersection
    return intersection / union if intersection else 0.0


def _can_merge(first: RegionProposal, second: RegionProposal) -> bool:
    """Allow deduplication only when proposals do not share a detector source."""
    return not set(_detector_evidence(first)).intersection(_detector_evidence(second))


def _candidate_coverage(candidate: BoundingBox, text: BoundingBox) -> float:
    """Return the fraction of an icon candidate covered by a text detection."""
    return _intersection_area(candidate, text) / candidate.area if candidate.area else 0.0


def _intersection_area(first: BoundingBox, second: BoundingBox) -> int:
    """Return the shared area of two pixel rectangles."""
    width = max(0, min(first.right, second.right) - max(first.x, second.x))
    height = max(0, min(first.bottom, second.bottom) - max(first.y, second.y))
    return width * height


def _polygon_bbox(polygon: list[list[float]]) -> BoundingBox:
    """Convert an OCR quadrilateral into the smallest enclosing pixel rectangle."""
    if not polygon:
        return BoundingBox(0, 0, 0, 0)
    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]
    x, y = int(min(xs)), int(min(ys))
    return BoundingBox(x=x, y=y, width=max(1, int(max(xs)) - x), height=max(1, int(max(ys)) - y))


def _padded_box(bbox: BoundingBox, padding: int, image_width: int, image_height: int) -> BoundingBox:
    """Expand a candidate box while keeping it completely inside the source image."""
    x = max(0, bbox.x - padding)
    y = max(0, bbox.y - padding)
    right = min(image_width, bbox.right + padding)
    bottom = min(image_height, bbox.bottom + padding)
    return BoundingBox(x=x, y=y, width=right - x, height=bottom - y)


def _proposal_score(proposal: RegionProposal) -> float:
    """Give multi-detector proposals a transparent, non-identification priority score."""
    source_count = len(_detector_evidence(proposal))
    return min(1.0, 0.5 + 0.25 * source_count)


def _detector_evidence(proposal: RegionProposal) -> dict[str, list[dict[str, Any]]]:
    """Return a uniform detector-evidence map whether merging is enabled or not."""
    nested = proposal.evidence.get("detector_evidence")
    if isinstance(nested, dict):
        return nested
    return {proposal.detector: [proposal.evidence]}
