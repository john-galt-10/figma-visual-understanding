"""Serializable records passed between the icon-library matching components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from icon_matching.normalization import NormalizedIcon


@dataclass(frozen=True)
class TemplateRecord:
    """Contain one loaded template label, canvas data, and manifest artifact paths."""

    label: str
    soft_canvas: np.ndarray
    binary_canvas: np.ndarray
    artifact_paths: dict[str, str]


@dataclass(frozen=True)
class PrimaryMatch:
    """Associate a template with its primary matcher similarity."""

    template: TemplateRecord
    primary_score: float


@dataclass(frozen=True)
class MatchResult:
    """Represent one ranked library result exposed by the CLI and JSON artifact."""

    rank: int
    label: str
    final_score: float
    primary_score: float
    tie_breaker_score: float | None
    artifact_paths: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        """Return an inspectable JSON-ready match result."""
        return {
            "rank": self.rank,
            "label": self.label,
            "final_score": round(self.final_score, 6),
            "primary_score": round(self.primary_score, 6),
            "tie_breaker_score": None if self.tie_breaker_score is None else round(self.tie_breaker_score, 6),
            "artifact_paths": self.artifact_paths,
        }


@dataclass(frozen=True)
class MatchRunResult:
    """Contain query preprocessing metadata and the ranked library results."""

    query_path: str
    query: NormalizedIcon
    primary_matcher: str
    tie_breaker: str | None
    results: list[MatchResult]

    def to_dict(self) -> dict[str, Any]:
        """Return the complete JSON artifact emitted by the example script."""
        return {
            "schema_version": "1.0",
            "query": {
                "path": self.query_path,
                "glyph_bbox": self.query.glyph_bbox,
                "threshold": self.query.threshold,
                "canvas_dimensions": {
                    "width": int(self.query.soft_canvas.shape[1]),
                    "height": int(self.query.soft_canvas.shape[0]),
                },
            },
            "primary_matcher": self.primary_matcher,
            "tie_breaker": self.tie_breaker,
            "results": [result.to_dict() for result in self.results],
        }
