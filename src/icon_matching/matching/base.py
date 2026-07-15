"""Contracts and errors shared by icon-matching providers."""

from __future__ import annotations

from abc import ABC, abstractmethod

from icon_matching.normalization import NormalizedIcon

from .models import PrimaryMatch, TemplateRecord


class IconMatchingError(RuntimeError):
    """Raised when library loading or matching cannot complete."""


class IconMatchingEvaluationError(RuntimeError):
    """Raised when a labeled icon-matching evaluation cannot be completed."""


class IconMatcher(ABC):
    """Define the swappable primary-matcher contract used by the matching pipeline."""

    name: str

    @abstractmethod
    def match(self, query: NormalizedIcon, templates: list[TemplateRecord]) -> list[PrimaryMatch]:
        """Return one normalized similarity score for every supplied template."""


class IconTieBreaker(ABC):
    """Define an optional secondary score for reranking primary matches."""

    name: str

    @abstractmethod
    def score(self, query: NormalizedIcon, template: TemplateRecord) -> float:
        """Return a normalized similarity score from zero through one."""
