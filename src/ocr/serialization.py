"""Helpers for preserving provider evidence in JSON output."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def to_json_safe(value: Any) -> Any:
    """Recursively convert common OCR-library values into JSON-compatible data."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): to_json_safe(item) for key, item in value.items()}
    if hasattr(value, "tolist"):
        return to_json_safe(value.tolist())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [to_json_safe(item) for item in value]
    return str(value)


def to_polygon(value: Any) -> list[list[float]]:
    """Normalize a provider quadrilateral into JSON-compatible pixel coordinates."""
    points = to_json_safe(value)
    if not isinstance(points, list):
        return []
    polygon: list[list[float]] = []
    for point in points:
        if isinstance(point, list) and len(point) >= 2:
            polygon.append([float(point[0]), float(point[1])])
    return polygon
