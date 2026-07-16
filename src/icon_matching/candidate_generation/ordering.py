"""Configurable visual reading-order strategies for final icon proposals."""

from __future__ import annotations

from abc import ABC, abstractmethod
from statistics import median

from .models import RegionProposal


class CandidateOrderingStrategy(ABC):
    """Order final filtered proposals before stable candidate IDs are assigned."""

    @abstractmethod
    def order(self, proposals: list[RegionProposal]) -> list[RegionProposal]:
        """Return proposals in the strategy's deterministic visual reading order."""


class StrictTopLeftOrdering(CandidateOrderingStrategy):
    """Preserve the legacy top-left coordinate ordering exactly."""

    def order(self, proposals: list[RegionProposal]) -> list[RegionProposal]:
        """Order proposals by vertical position, then horizontal position and dimensions."""
        return sorted(proposals, key=lambda proposal: _box_tie_breaker(proposal))


class CenterClusterOrdering(CandidateOrderingStrategy):
    """Cluster visually aligned proposals into rows before ordering them left to right."""

    def __init__(self, row_center_tolerance_height_multiplier: float) -> None:
        """Store the positive multiplier applied to the median proposal height."""
        self.row_center_tolerance_height_multiplier = row_center_tolerance_height_multiplier

    def order(self, proposals: list[RegionProposal]) -> list[RegionProposal]:
        """Group centers into rows, then return rows and their members in reading order."""
        if not proposals:
            return []
        tolerance = self.row_center_tolerance_height_multiplier * median(
            proposal.bbox.height for proposal in proposals
        )
        vertically_sorted = sorted(proposals, key=lambda proposal: (_vertical_center(proposal), *_box_tie_breaker(proposal)))
        rows: list[list[RegionProposal]] = []
        row_first_centers: list[float] = []
        for proposal in vertically_sorted:
            center = _vertical_center(proposal)
            if not rows or center - row_first_centers[-1] > tolerance:
                rows.append([proposal])
                row_first_centers.append(center)
            else:
                rows[-1].append(proposal)
        return [
            proposal
            for row in rows
            for proposal in sorted(
                row,
                key=lambda item: (_horizontal_center(item), *_box_tie_breaker(item)),
            )
        ]


def _horizontal_center(proposal: RegionProposal) -> float:
    """Return a proposal's horizontal bounding-box center."""
    return proposal.bbox.x + proposal.bbox.width / 2


def _vertical_center(proposal: RegionProposal) -> float:
    """Return a proposal's vertical bounding-box center."""
    return proposal.bbox.y + proposal.bbox.height / 2


def _box_tie_breaker(proposal: RegionProposal) -> tuple[int, int, int, int]:
    """Return the legacy deterministic coordinate and dimension tie-breaking tuple."""
    bbox = proposal.bbox
    return bbox.y, bbox.x, bbox.width, bbox.height
