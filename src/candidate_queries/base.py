"""Common contract and validation shared by candidate-query providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from .models import CandidateQueryResult, ImageMetadata


SUPPORTED_IMAGE_FORMATS = frozenset({"PNG", "JPEG", "WEBP", "BMP", "TIFF"})


def build_user_prompt(
    textual_query: str | None,
    visual_context: str | None = None,
    input_description: str | None = None,
) -> str:
    """Build text content with separate input-instruction and evidence sections."""
    normalized_query = (textual_query or "").strip()
    if normalized_query:
        prompt = f"User question: {normalized_query}\nGenerate retrieval-query formulations."
    else:
        prompt = (
            "No user question was supplied. Inspect the screenshot and generate "
            "feature-identification retrieval-query formulations."
        )
    normalized_description = (input_description or "").strip()
    if normalized_description:
        prompt = f"{prompt}\n\nInput description:\n{normalized_description}"
    normalized_context = (visual_context or "").strip()
    if normalized_context:
        return f"{prompt}\n\nAuxiliary visual evidence:\n{normalized_context}"
    return prompt


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
        self,
        image_path: str | Path,
        textual_query: str | None = None,
        output_trace: bool | None = None,
        visual_context: str | None = None,
        input_description: str | None = None,
        additional_image_paths: list[str | Path] | None = None,
    ) -> CandidateQueryResult:
        """Generate queries with optional evidence and ordered supplemental images."""

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
