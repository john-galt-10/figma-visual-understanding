"""Loading and validation for candidate-query YAML configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, ValidationError

from .base import CandidateQueryConfigurationError


class GenerationSettings(BaseModel):
    """Controls Gemini response generation behavior."""

    temperature: float = Field(ge=0.0, le=2.0)
    max_output_tokens: int = Field(gt=0)
    thinking_level: Literal["minimal", "low", "medium", "high"] = "minimal"
    output_reasoning_summary: bool = False


class CandidateQuerySettings(BaseModel):
    """Contains the provider-independent settings for one query generator."""

    provider: str
    model: str
    api_key_environment_variable: str
    max_candidates: int = Field(gt=0)
    system_instruction: str = Field(min_length=1)
    generation: GenerationSettings


class ApplicationSettings(BaseModel):
    """Represents the root configuration document."""

    candidate_queries: CandidateQuerySettings


def load_settings(config_path: str | Path) -> CandidateQuerySettings:
    """Load the candidate-query section from a YAML configuration file."""
    path = Path(config_path).expanduser().resolve()
    if not path.is_file():
        raise CandidateQueryConfigurationError(f"Configuration file does not exist: {path}")
    try:
        with path.open("r", encoding="utf-8") as config_file:
            raw_settings = yaml.safe_load(config_file) or {}
        return ApplicationSettings.model_validate(raw_settings).candidate_queries
    except (OSError, yaml.YAMLError, ValidationError) as error:
        raise CandidateQueryConfigurationError(
            f"Invalid candidate-query configuration in '{path}': {error}"
        ) from error
