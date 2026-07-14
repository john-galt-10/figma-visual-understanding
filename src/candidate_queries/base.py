"""Common contract and validation shared by candidate-query providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from .models import CandidateQueryResult, ImageMetadata


SUPPORTED_IMAGE_FORMATS = frozenset({"PNG", "JPEG", "WEBP", "BMP", "TIFF"})


class CandidateQueryError(RuntimeError):
    """Base error raised when candidate queries cannot be generated."""


class CandidateQueryConfigurationError(CandidateQueryError):
    """Raised when candidate-query configuration is missing or invalid."""


class CandidateQueryDependencyError(CandidateQueryError):
    """Raised when a selected provider's optional package is unavailable."""


class CandidateQueryProviderError(CandidateQueryError):
    """Raised when a provider rejects or fails to complete a request."""


class InvalidImageError(CandidateQueryError):
    """Raised when a screenshot does not exist or is an unsupported image."""


class CandidateQueryGenerator(ABC):
    """Defines the stable interface implemented by every query provider."""

    provider_name: str

    @abstractmethod
    def generate(
        self, image_path: str | Path, textual_query: str | None = None
    ) -> CandidateQueryResult:
        """Generate normalized documentation-retrieval queries for one screenshot."""

    @staticmethod
    def read_image_metadata(image_path: str | Path) -> ImageMetadata:
        """Validate a local raster image and return safe metadata for the result."""
        path = Path(image_path).expanduser().resolve()
        if not path.is_file():
            raise InvalidImageError(f"Image file does not exist: {path}")
        try:
            with Image.open(path) as image:
                image_format = (image.format or "UNKNOWN").upper()
                if image_format not in SUPPORTED_IMAGE_FORMATS:
                    supported = ", ".join(sorted(SUPPORTED_IMAGE_FORMATS))
                    raise InvalidImageError(
                        f"Unsupported image format '{image_format}'. Supported formats: {supported}."
                    )
                return ImageMetadata(
                    path=str(path),
                    filename=path.name,
                    format=image_format,
                    width=image.width,
                    height=image.height,
                )
        except UnidentifiedImageError as error:
            raise InvalidImageError(f"Input is not a readable image: {path}") from error
