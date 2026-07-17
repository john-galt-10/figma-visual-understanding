"""Configuration loading for the unified visual-signal pipeline."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, ValidationError

from candidate_queries.config import CandidateQuerySettings
from icon_matching.candidate_generation.config import IconMatchingSettings


class OcrPipelineSettings(BaseModel):
    """Select whether OCR contributes text evidence and which backend it uses."""

    enabled: bool = True
    engine: str = "easy"
    language: str = Field(default="en", min_length=1)
    detection_threshold: float = Field(default=0.90, ge=0.0, le=1.0)


class IconPipelineSettings(IconMatchingSettings):
    """Extend icon-matching settings with a pipeline-level enable switch."""

    enabled: bool = True


class PipelineSettings(BaseModel):
    """Contains the independently switchable evidence stages of a pipeline run."""

    ocr: OcrPipelineSettings = Field(default_factory=OcrPipelineSettings)
    icon_matching: IconPipelineSettings = Field(default_factory=IconPipelineSettings)


class ApplicationSettings(BaseModel):
    """Represents the complete pipeline YAML document."""

    pipeline: PipelineSettings = Field(default_factory=PipelineSettings)
    candidate_queries: CandidateQuerySettings


class PipelineConfigurationError(RuntimeError):
    """Raised when the pipeline configuration is absent or invalid."""


def load_settings(config_path: str | Path) -> ApplicationSettings:
    """Load and validate all component settings from one YAML document."""
    path = Path(config_path).expanduser().resolve()
    if not path.is_file():
        raise PipelineConfigurationError(f"Configuration file does not exist: {path}")
    try:
        with path.open("r", encoding="utf-8") as config_file:
            return ApplicationSettings.model_validate(yaml.safe_load(config_file) or {})
    except (OSError, yaml.YAMLError, ValidationError) as error:
        raise PipelineConfigurationError(
            f"Invalid pipeline configuration in '{path}': {error}"
        ) from error
