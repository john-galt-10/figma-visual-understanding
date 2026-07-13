"""Serializable data structures shared by every OCR implementation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


Polygon = list[list[float]]


@dataclass
class ImageMetadata:
    """Describes the raster image passed to an OCR engine."""

    path: str
    filename: str
    format: str
    width: int
    height: int


@dataclass
class TextDetection:
    """Represents one text region detected by an OCR engine."""

    text: str
    confidence: float | None
    polygon: Polygon
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class OcrScanResult:
    """Contains normalized OCR detections and scan-level evidence."""

    image: ImageMetadata
    engine: dict[str, Any]
    detections: list[TextDetection]
    visible_text: str
    detection_count: int
    processing_time_seconds: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return the result as a JSON-serializable dictionary."""
        return asdict(self)
