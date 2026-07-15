"""Symmetric distance-transform Chamfer matching for small binary icon masks."""

from __future__ import annotations

import cv2
import numpy as np

from icon_matching.normalization import NormalizedIcon

from .base import IconMatcher
from .models import PrimaryMatch, TemplateRecord


class ChamferMatcher(IconMatcher):
    """Rank templates by symmetric edge distance while tolerating small translations."""

    name = "chamfer"

    def __init__(self, translation_radius_pixels: int) -> None:
        """Configure the maximum horizontal and vertical query shift to evaluate."""
        self.translation_radius_pixels = translation_radius_pixels

    def match(self, query: NormalizedIcon, templates: list[TemplateRecord]) -> list[PrimaryMatch]:
        """Return all templates ranked by normalized symmetric Chamfer similarity."""
        query_edges = _edge_map(query.binary_canvas)
        matches = [
            PrimaryMatch(template=template, primary_score=self._similarity(query_edges, _edge_map(template.binary_canvas)))
            for template in templates
        ]
        return sorted(matches, key=lambda match: match.primary_score, reverse=True)

    def _similarity(self, query_edges: np.ndarray, template_edges: np.ndarray) -> float:
        """Evaluate the best symmetric edge distance over the configured translations."""
        height, width = query_edges.shape
        diagonal = float(np.hypot(height, width))
        template_distance = _distance_transform(template_edges)
        best_distance = float("inf")
        for offset_y in range(-self.translation_radius_pixels, self.translation_radius_pixels + 1):
            for offset_x in range(-self.translation_radius_pixels, self.translation_radius_pixels + 1):
                shifted_query = _translate(query_edges, offset_x, offset_y)
                query_distance = _distance_transform(shifted_query)
                forward = _directed_distance(shifted_query, template_distance)
                reverse = _directed_distance(template_edges, query_distance)
                best_distance = min(best_distance, (forward + reverse) / 2.0)
        return float(np.clip(1.0 - best_distance / diagonal, 0.0, 1.0))


def _edge_map(binary_canvas: np.ndarray) -> np.ndarray:
    """Return a one-pixel-ish binary outline, retaining sparse glyphs as foreground."""
    edges = cv2.morphologyEx(binary_canvas, cv2.MORPH_GRADIENT, np.ones((3, 3), dtype=np.uint8))
    return edges if np.any(edges) else binary_canvas


def _distance_transform(edges: np.ndarray) -> np.ndarray:
    """Return each pixel's Euclidean-like distance to the nearest edge pixel."""
    return cv2.distanceTransform(cv2.bitwise_not(edges), cv2.DIST_L2, 3)


def _directed_distance(source_edges: np.ndarray, target_distance: np.ndarray) -> float:
    """Return mean distance from source edge pixels to their nearest target edge pixel."""
    values = target_distance[source_edges > 0]
    return float(values.mean()) if values.size else float(max(target_distance.shape))


def _translate(image: np.ndarray, offset_x: int, offset_y: int) -> np.ndarray:
    """Shift an edge map without wrapping pixels around the canvas boundary."""
    matrix = np.float32([[1, 0, offset_x], [0, 1, offset_y]])
    return cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]), borderValue=0)
