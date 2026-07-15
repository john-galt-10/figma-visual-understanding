"""Loading and validation for icon candidate-generation YAML settings."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, ValidationError, model_validator

from .base import CandidateGenerationConfigurationError


class PreprocessingSettings(BaseModel):
    """Configure source-image normalization shared by all detectors."""

    enabled: bool = True
    grayscale: bool = True
    blur_kernel_size: int = Field(default=3, ge=1)

    @model_validator(mode="after")
    def validate_kernel_size(self) -> "PreprocessingSettings":
        """Require an odd blur kernel accepted by OpenCV."""
        if self.blur_kernel_size % 2 == 0:
            raise ValueError("blur_kernel_size must be odd")
        return self


class MorphologySettings(BaseModel):
    """Configure local-contrast connected-component proposals."""

    enabled: bool = True
    local_contrast_threshold: int = Field(default=22, ge=1, le=255)
    adaptive_block_size: int = Field(default=21, ge=3)
    adaptive_constant: int = Field(default=5, ge=0)
    dilation_kernel_size: int = Field(default=2, ge=1)
    dilation_iterations: int = Field(default=1, ge=0)
    minimum_component_area: int = Field(default=9, ge=1)

    @model_validator(mode="after")
    def validate_block_size(self) -> "MorphologySettings":
        """Require an odd adaptive-threshold window accepted by OpenCV."""
        if self.adaptive_block_size % 2 == 0:
            raise ValueError("adaptive_block_size must be odd")
        return self


class ContourSettings(BaseModel):
    """Configure Canny contour proposals and local contour grouping."""

    enabled: bool = True
    canny_low_threshold: int = Field(default=45, ge=0, le=255)
    canny_high_threshold: int = Field(default=135, ge=1, le=255)
    minimum_contour_area: int = Field(default=4, ge=1)
    grouping_distance: int = Field(default=4, ge=0)

    @model_validator(mode="after")
    def validate_thresholds(self) -> "ContourSettings":
        """Require a meaningful low-to-high Canny threshold range."""
        if self.canny_low_threshold >= self.canny_high_threshold:
            raise ValueError("canny_low_threshold must be lower than canny_high_threshold")
        return self


class DetectorsSettings(BaseModel):
    """Collect settings for each independently switchable proposal detector."""

    morphology: MorphologySettings = Field(default_factory=MorphologySettings)
    contours: ContourSettings = Field(default_factory=ContourSettings)


class MergingSettings(BaseModel):
    """Configure cross-detector deduplication using intersection over union."""

    enabled: bool = True
    overlap_threshold: float = Field(default=0.5, gt=0.0, le=1.0)


class SquareShapeSettings(BaseModel):
    """Configure optional preference for square-like icon candidate boxes."""

    enabled: bool = True
    minimum_compactness: float = Field(default=0.6, gt=0.0, le=1.0)


class FilterSettings(BaseModel):
    """Configure inexpensive geometric filtering of proposed regions."""

    enabled: bool = True
    minimum_width: int = Field(default=6, ge=1)
    minimum_height: int = Field(default=6, ge=1)
    maximum_width: int = Field(default=96, ge=1)
    maximum_height: int = Field(default=96, ge=1)
    minimum_aspect_ratio: float = Field(default=0.2, gt=0.0)
    maximum_aspect_ratio: float = Field(default=5.0, gt=0.0)
    minimum_compactness: float = Field(default=0.05, ge=0.0, le=1.0)
    square_shape: SquareShapeSettings = Field(default_factory=SquareShapeSettings)

    @model_validator(mode="after")
    def validate_ranges(self) -> "FilterSettings":
        """Reject inverted geometric filter ranges."""
        if self.minimum_width > self.maximum_width or self.minimum_height > self.maximum_height:
            raise ValueError("minimum dimensions must not exceed maximum dimensions")
        if self.minimum_aspect_ratio > self.maximum_aspect_ratio:
            raise ValueError("minimum_aspect_ratio must not exceed maximum_aspect_ratio")
        return self


class OcrSuppressionSettings(BaseModel):
    """Configure optional internal OCR and candidate/text overlap rejection."""

    enabled: bool = False
    engine: str = "paddle"
    language: str = Field(default="en", min_length=1)
    minimum_text_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    candidate_text_overlap_threshold: float = Field(default=0.6, gt=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_engine(self) -> "OcrSuppressionSettings":
        """Limit the selection to engines exposed by the existing OCR factory."""
        if self.engine.lower() not in {"paddle", "easy"}:
            raise ValueError("engine must be 'paddle' or 'easy'")
        return self


class CropSettings(BaseModel):
    """Configure padding around detected content when materializing crops."""

    padding_pixels: int = Field(default=2, ge=0)


class VisualizationSettings(BaseModel):
    """Configure optional annotated overlay output."""

    enabled: bool = True
    line_width: int = Field(default=2, ge=1)
    label_font_scale: float = Field(default=0.45, gt=0.0)
    morphology_color_bgr: list[int] = Field(default_factory=lambda: [0, 180, 255], min_length=3, max_length=3)
    contours_color_bgr: list[int] = Field(default_factory=lambda: [255, 170, 0], min_length=3, max_length=3)
    combined_color_bgr: list[int] = Field(default_factory=lambda: [80, 220, 80], min_length=3, max_length=3)

    @model_validator(mode="after")
    def validate_colors(self) -> "VisualizationSettings":
        """Ensure every BGR channel is accepted by OpenCV drawing functions."""
        for color in (self.morphology_color_bgr, self.contours_color_bgr, self.combined_color_bgr):
            if any(channel < 0 or channel > 255 for channel in color):
                raise ValueError("visualization colors must contain values from 0 through 255")
        return self


class TemplatePreprocessingSettings(BaseModel):
    """Configure normalization shared by template building and library matching."""

    canvas_size: int = Field(default=32, ge=3)
    canvas_margin: int = Field(default=2, ge=0)

    @model_validator(mode="after")
    def validate_margin(self) -> "TemplatePreprocessingSettings":
        """Require an empty margin that still leaves canvas room for the glyph."""
        if self.canvas_margin * 2 >= self.canvas_size:
            raise ValueError("canvas_margin must leave room for the glyph")
        return self


class TemplateLibrarySettings(BaseModel):
    """Identify the manifest-backed template collection used for matching."""

    manifest_path: str = "assets/icon_library/templates/templates.json"


class PrimaryMatcherSettings(BaseModel):
    """Configure the independently selectable primary icon matcher."""

    provider: Literal["chamfer"] = "chamfer"
    translation_radius_pixels: int = Field(default=2, ge=0)


class TieBreakerSettings(BaseModel):
    """Configure optional soft-template reranking of leading primary matches."""

    enabled: bool = True
    provider: Literal["soft_ncc"] = "soft_ncc"
    candidate_pool_size: int = Field(default=5, gt=0)
    primary_weight: float = Field(default=0.8, ge=0.0, le=1.0)
    tie_breaker_weight: float = Field(default=0.2, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_weights(self) -> "TieBreakerSettings":
        """Require enabled weighted reranking to use a complete score budget."""
        if abs(self.primary_weight + self.tie_breaker_weight - 1.0) > 1e-9:
            raise ValueError("primary_weight and tie_breaker_weight must sum to 1.0")
        return self


class MatchResultSettings(BaseModel):
    """Configure the default number of ranked matches returned to callers."""

    default_top_k: int = Field(default=5, gt=0)


class MatchingSettings(BaseModel):
    """Collect all configuration needed by the template-library matcher."""

    library: TemplateLibrarySettings = Field(default_factory=TemplateLibrarySettings)
    primary: PrimaryMatcherSettings = Field(default_factory=PrimaryMatcherSettings)
    tie_breaker: TieBreakerSettings = Field(default_factory=TieBreakerSettings)
    results: MatchResultSettings = Field(default_factory=MatchResultSettings)


class IconMatchingSettings(BaseModel):
    """Contain all candidate-generation settings loaded from one YAML file."""

    preprocessing: PreprocessingSettings = Field(default_factory=PreprocessingSettings)
    detectors: DetectorsSettings = Field(default_factory=DetectorsSettings)
    merging: MergingSettings = Field(default_factory=MergingSettings)
    filters: FilterSettings = Field(default_factory=FilterSettings)
    ocr_suppression: OcrSuppressionSettings = Field(default_factory=OcrSuppressionSettings)
    crops: CropSettings = Field(default_factory=CropSettings)
    visualization: VisualizationSettings = Field(default_factory=VisualizationSettings)
    template_preprocessing: TemplatePreprocessingSettings = Field(default_factory=TemplatePreprocessingSettings)
    matching: MatchingSettings = Field(default_factory=MatchingSettings)


class ApplicationSettings(BaseModel):
    """Represent the root document of an icon-matching configuration file."""

    icon_matching: IconMatchingSettings


def load_settings(config_path: str | Path) -> IconMatchingSettings:
    """Load and validate the icon-matching section of a YAML document."""
    path = Path(config_path).expanduser().resolve()
    if not path.is_file():
        raise CandidateGenerationConfigurationError(f"Configuration file does not exist: {path}")
    try:
        with path.open("r", encoding="utf-8") as config_file:
            raw_settings = yaml.safe_load(config_file) or {}
        return ApplicationSettings.model_validate(raw_settings).icon_matching
    except (OSError, yaml.YAMLError, ValidationError) as error:
        raise CandidateGenerationConfigurationError(
            f"Invalid icon-matching configuration in '{path}': {error}"
        ) from error
