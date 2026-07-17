"""Confidence-based filtering helpers for normalized OCR detections."""

from __future__ import annotations

from .models import TextDetection


def filter_detections_by_confidence(
    detections: list[TextDetection], detection_threshold: float
) -> list[TextDetection]:
    """Return non-empty OCR detections whose confidence meets a threshold.

    Detections without a confidence score are excluded because they cannot be
    verified against the requested threshold.
    """
    if not 0.0 <= detection_threshold <= 1.0:
        raise ValueError("OCR detection threshold must be between 0.0 and 1.0.")
    return [
        detection
        for detection in detections
        if detection.text.strip()
        and detection.confidence is not None
        and detection.confidence >= detection_threshold
    ]
