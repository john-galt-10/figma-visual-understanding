"""Shared score selection for thresholded icon detection consumers."""

from __future__ import annotations

from typing import Literal

from .base import IconMatchingError
from .models import MatchResult


DetectionScoreName = Literal["final", "primary", "secondary"]


def select_detection_score(match: MatchResult, score_name: DetectionScoreName) -> float:
    """Return one top-match score, rejecting unavailable secondary scoring."""
    if score_name == "final":
        return match.final_score
    if score_name == "primary":
        return match.primary_score
    if match.tie_breaker_score is None:
        raise IconMatchingError(
            "Secondary detection scoring is unavailable for the top result. "
            "Increase matching.tie_breaker.candidate_pool_size or select final/primary."
        )
    return match.tie_breaker_score
