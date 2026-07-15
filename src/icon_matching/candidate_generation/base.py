"""Shared exceptions and contracts for candidate-generation components."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .models import RegionProposal


class CandidateGenerationError(RuntimeError):
    """Base error raised when candidate generation cannot complete."""


class CandidateGenerationDependencyError(CandidateGenerationError):
    """Raised when an optional computer-vision dependency is unavailable."""


class CandidateGenerationConfigurationError(CandidateGenerationError):
    """Raised when candidate-generation YAML settings are invalid."""


class RegionDetector(ABC):
    """Define the common contract implemented by icon-region proposal methods."""

    name: str

    @abstractmethod
    def detect(self, grayscale_image: np.ndarray) -> list[RegionProposal]:
        """Return raw proposals in source-image pixel coordinates."""
