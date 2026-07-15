"""Loading, normalization, scoring, and reranking for one icon-library query."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from icon_matching.candidate_generation.config import IconMatchingSettings
from icon_matching.normalization import IconNormalizationError, IconNormalizer

from .base import IconMatchingError
from .factory import create_icon_matcher, create_tie_breaker
from .models import MatchResult, MatchRunResult, PrimaryMatch, TemplateRecord


class IconLibraryMatcher:
    """Match a single icon image against a manifest-backed normalized template library."""

    def __init__(self, settings: IconMatchingSettings) -> None:
        """Load the configured library and instantiate the selected matching providers."""
        self.settings = settings
        preprocessing = settings.template_preprocessing
        self.normalizer = IconNormalizer(preprocessing.canvas_size, preprocessing.canvas_margin)
        self.templates = self._load_templates(settings.matching.library.manifest_path, preprocessing.canvas_size)
        self.primary_matcher = create_icon_matcher(settings.matching.primary)
        self.tie_breaker = create_tie_breaker(settings.matching.tie_breaker) if settings.matching.tie_breaker.enabled else None

    def match(self, image_path: str | Path, top_k: int | None = None) -> MatchRunResult:
        """Normalize one query image and return its configured ranked library matches."""
        requested_top_k = top_k or self.settings.matching.results.default_top_k
        if requested_top_k < 1:
            raise ValueError("top_k must be at least 1")
        try:
            query = self.normalizer.normalize_path(image_path)
        except IconNormalizationError as error:
            raise IconMatchingError(str(error)) from error
        primary_matches = self.primary_matcher.match(query, self.templates)
        results = self._rank(query, primary_matches)
        return MatchRunResult(
            query_path=str(Path(image_path).expanduser().resolve()),
            query=query,
            primary_matcher=self.primary_matcher.name,
            tie_breaker=None if self.tie_breaker is None else self.tie_breaker.name,
            results=results[:requested_top_k],
        )

    def _rank(self, query, primary_matches: list[PrimaryMatch]) -> list[MatchResult]:
        """Apply optional weighted reranking inside the configured leading candidate pool."""
        tie_settings = self.settings.matching.tie_breaker
        pool_size = min(tie_settings.candidate_pool_size, len(primary_matches))
        ranked: list[tuple[PrimaryMatch, float, float | None]]
        if self.tie_breaker is None:
            ranked = [(match, match.primary_score, None) for match in primary_matches]
        else:
            pool = []
            for match in primary_matches[:pool_size]:
                tie_score = self.tie_breaker.score(query, match.template)
                pool.append(
                    (
                        match,
                        tie_settings.primary_weight * match.primary_score + tie_settings.tie_breaker_weight * tie_score,
                        tie_score,
                    )
                )
            pool.sort(key=lambda item: item[1], reverse=True)
            ranked = pool + [(match, match.primary_score, None) for match in primary_matches[pool_size:]]
        return [
            MatchResult(
                rank=index,
                label=match.template.label,
                final_score=final_score,
                primary_score=match.primary_score,
                tie_breaker_score=tie_score,
                artifact_paths=match.template.artifact_paths,
            )
            for index, (match, final_score, tie_score) in enumerate(ranked, start=1)
        ]

    @staticmethod
    def _load_templates(manifest_path: str, expected_canvas_size: int) -> list[TemplateRecord]:
        """Load manifest entries and reject missing or incompatible template artifacts."""
        manifest_file = Path(manifest_path).expanduser().resolve()
        if not manifest_file.is_file():
            raise IconMatchingError(f"Template manifest does not exist: {manifest_file}")
        try:
            manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
            entries = manifest["templates"]
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
            raise IconMatchingError(f"Invalid template manifest '{manifest_file}': {error}") from error
        if not entries:
            raise IconMatchingError(f"Template manifest contains no templates: {manifest_file}")
        records: list[TemplateRecord] = []
        for entry in entries:
            try:
                soft_path = Path(entry["soft_canvas_path"])
                binary_path = Path(entry["binary_canvas_path"])
                soft_canvas = _read_canvas(soft_path, expected_canvas_size)
                binary_canvas = _read_canvas(binary_path, expected_canvas_size)
                if not np.isin(binary_canvas, [0, 255]).all():
                    raise IconMatchingError(f"Binary template is not black/white: {binary_path}")
                records.append(
                    TemplateRecord(
                        label=str(entry["label"]),
                        soft_canvas=soft_canvas,
                        binary_canvas=binary_canvas,
                        artifact_paths={
                            "soft_crop_path": str(entry["soft_crop_path"]),
                            "binary_crop_path": str(entry["binary_crop_path"]),
                            "soft_canvas_path": str(soft_path),
                            "binary_canvas_path": str(binary_path),
                        },
                    )
                )
            except (KeyError, TypeError, OSError, ValueError) as error:
                raise IconMatchingError(f"Invalid template entry in '{manifest_file}': {error}") from error
        return records


def _read_canvas(path: Path, expected_size: int) -> np.ndarray:
    """Read and validate a square grayscale template canvas."""
    if not path.is_file():
        raise IconMatchingError(f"Template artifact does not exist: {path}")
    with Image.open(path) as image:
        pixels = np.array(image.convert("L"), dtype=np.uint8)
    if pixels.shape != (expected_size, expected_size):
        raise IconMatchingError(
            f"Template canvas '{path}' is {pixels.shape[1]}x{pixels.shape[0]}, expected {expected_size}x{expected_size}."
        )
    return pixels
