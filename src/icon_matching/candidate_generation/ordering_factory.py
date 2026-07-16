"""Factory and registration point for candidate-ordering strategies."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .ordering import CandidateOrderingStrategy, CenterClusterOrdering, StrictTopLeftOrdering

if TYPE_CHECKING:
    from .config import OrderingSettings


ORDERING_STRATEGIES = {
    "center_cluster": lambda settings: CenterClusterOrdering(settings.row_center_tolerance_height_multiplier),
    "strict_top_left": lambda settings: StrictTopLeftOrdering(),
}


def create_candidate_ordering_strategy(settings: OrderingSettings) -> CandidateOrderingStrategy:
    """Create the configured registered candidate-ordering strategy."""
    try:
        return ORDERING_STRATEGIES[settings.provider](settings)
    except KeyError as error:
        raise ValueError(f"Unknown or unregistered candidate-ordering provider '{settings.provider}'.") from error


def registered_candidate_ordering_providers() -> set[str]:
    """Return the provider names accepted by the candidate-ordering factory."""
    return set(ORDERING_STRATEGIES)
