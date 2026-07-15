"""Serializable models used by icon candidate generation and future matching."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class BoundingBox:
    """Describe an axis-aligned rectangle in source-image pixel coordinates."""

    x: int
    y: int
    width: int
    height: int

    @property
    def right(self) -> int:
        """Return the exclusive horizontal end coordinate."""
        return self.x + self.width

    @property
    def bottom(self) -> int:
        """Return the exclusive vertical end coordinate."""
        return self.y + self.height

    @property
    def area(self) -> int:
        """Return the rectangle area in pixels."""
        return self.width * self.height


@dataclass
class RegionProposal:
    """Represent one unfiltered region returned by a named detector."""

    bbox: BoundingBox
    detector: str
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass
class IconCandidate:
    """Represent one final region proposal available to an icon matcher."""

    id: str
    content_bbox: BoundingBox
    crop_bbox: BoundingBox
    detector_sources: list[str]
    detector_evidence: dict[str, list[dict[str, Any]]]
    proposal_score: float
    crop_path: str | None = None
    crop_dimensions: dict[str, int] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready candidate without unset crop fields."""
        payload = asdict(self)
        return {key: value for key, value in payload.items() if value is not None}


@dataclass
class CandidateGenerationResult:
    """Contain final candidates plus inspectable run-level metadata."""

    schema_version: str
    image: dict[str, Any]
    candidates: list[IconCandidate]
    processing_time_seconds: float
    components: dict[str, bool]
    counts: dict[str, int]
    crop_files_exported: bool
    configuration: dict[str, Any]
    ocr: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the stable JSON artifact consumed by later pipeline stages."""
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "image": self.image,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "processing_time_seconds": self.processing_time_seconds,
            "components": self.components,
            "counts": self.counts,
            "crop_files_exported": self.crop_files_exported,
            "configuration": self.configuration,
        }
        if self.ocr is not None:
            payload["ocr"] = self.ocr
        return payload
