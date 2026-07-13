"""Common interface and shared validation for OCR engines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from importlib import metadata
from pathlib import Path
from time import perf_counter
from typing import Any

from PIL import Image, UnidentifiedImageError

from .models import ImageMetadata, OcrScanResult


SUPPORTED_IMAGE_FORMATS = frozenset({"PNG", "JPEG", "WEBP", "BMP", "TIFF"})


class OcrError(RuntimeError):
    """Base error raised when an OCR scan cannot be completed."""


class OcrDependencyError(OcrError):
    """Raised when the selected OCR provider is not installed."""


class InvalidImageError(OcrError):
    """Raised when an OCR input is missing or is not a supported image."""


class OcrEngine(ABC):
    """Defines the stable contract implemented by every OCR provider adapter."""

    engine_name: str

    def __init__(self, language: str = "en") -> None:
        """Configure the OCR engine with a recognition language."""
        self.language = language

    def scan(self, image_path: str | Path) -> OcrScanResult:
        """Validate an image, run provider OCR, and build a normalized result."""
        path = Path(image_path).expanduser().resolve()
        image = self._read_image_metadata(path)
        started_at = perf_counter()
        detections, scan_metadata = self._scan_image(path)
        elapsed = perf_counter() - started_at
        return OcrScanResult(
            image=image,
            engine={
                "name": self.engine_name,
                "language": self.language,
                "version": self._package_version(),
            },
            detections=detections,
            visible_text="\n".join(detection.text for detection in detections),
            detection_count=len(detections),
            processing_time_seconds=round(elapsed, 6),
            metadata=scan_metadata,
        )

    @abstractmethod
    def _scan_image(self, image_path: Path) -> tuple[list[Any], dict[str, Any]]:
        """Run the provider-specific OCR operation and return normalized detections."""

    @abstractmethod
    def _package_name(self) -> str:
        """Return the installed package name used for version metadata."""

    def _package_version(self) -> str | None:
        """Return the selected provider's installed version, when discoverable."""
        try:
            return metadata.version(self._package_name())
        except metadata.PackageNotFoundError:
            return None

    @staticmethod
    def _read_image_metadata(path: Path) -> ImageMetadata:
        """Open an image safely and return its format and pixel dimensions."""
        if not path.is_file():
            raise InvalidImageError(f"Image file does not exist: {path}")
        try:
            with Image.open(path) as image:
                image_format = image.format or "UNKNOWN"
                if image_format.upper() not in SUPPORTED_IMAGE_FORMATS:
                    supported = ", ".join(sorted(SUPPORTED_IMAGE_FORMATS))
                    raise InvalidImageError(
                        f"Unsupported image format '{image_format}'. Supported formats: {supported}."
                    )
                return ImageMetadata(
                    path=str(path),
                    filename=path.name,
                    format=image_format.upper(),
                    width=image.width,
                    height=image.height,
                )
        except UnidentifiedImageError as error:
            raise InvalidImageError(f"Input is not a readable image: {path}") from error
