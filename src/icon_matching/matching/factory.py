"""Factories that select configured matching providers through stable names."""

from __future__ import annotations

from icon_matching.candidate_generation.config import PrimaryMatcherSettings, TieBreakerSettings

from .base import IconMatcher, IconMatchingError, IconTieBreaker
from .chamfer import ChamferMatcher
from .tie_breakers import SoftNccTieBreaker


def create_icon_matcher(settings: PrimaryMatcherSettings) -> IconMatcher:
    """Create the configured primary matcher implementation."""
    if settings.provider == "chamfer":
        return ChamferMatcher(settings.translation_radius_pixels)
    raise IconMatchingError(f"Unknown primary matcher provider '{settings.provider}'.")


def create_tie_breaker(settings: TieBreakerSettings) -> IconTieBreaker:
    """Create the configured optional tie-breaker implementation."""
    if settings.provider == "soft_ncc":
        return SoftNccTieBreaker()
    raise IconMatchingError(f"Unknown tie-breaker provider '{settings.provider}'.")
