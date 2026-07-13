"""Factory for selecting an OCR provider through a stable method name."""

from __future__ import annotations

from .base import OcrEngine


def create_ocr_engine(method: str, language: str = "en") -> OcrEngine:
    """Create an OCR engine for the given method name and recognition language."""
    normalized_method = method.strip().lower()
    if normalized_method == "easy":
        from .engine import EasyOcrEngine

        return EasyOcrEngine(language=language)
    if normalized_method == "paddle":
        from .engine import PaddleOcrEngine

        return PaddleOcrEngine(language=language)
    raise ValueError("Unknown OCR engine '{}'. Choose one of: paddle, easy.".format(method))
