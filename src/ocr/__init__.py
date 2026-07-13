"""Library-neutral OCR utilities for Figma screenshot experiments."""

from .base import OcrEngine
from .factory import create_ocr_engine
from .models import ImageMetadata, OcrScanResult, TextDetection

__all__ = [
    "ImageMetadata",
    "OcrEngine",
    "OcrScanResult",
    "TextDetection",
    "create_ocr_engine",
]
