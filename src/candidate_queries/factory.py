"""Factory for selecting a candidate-query provider through a stable name."""

from __future__ import annotations

from .base import CandidateQueryConfigurationError, CandidateQueryGenerator
from .config import CandidateQuerySettings


def create_candidate_query_generator(
    settings: CandidateQuerySettings,
) -> CandidateQueryGenerator:
    """Create the configured candidate-query provider implementation."""
    provider = settings.provider.strip().lower()
    if provider == "gemini":
        from .gemini import GeminiCandidateQueryGenerator

        return GeminiCandidateQueryGenerator(settings)
    raise CandidateQueryConfigurationError(
        f"Unknown or unregistered candidate-query provider '{settings.provider}'."
    )
