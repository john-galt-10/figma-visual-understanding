"""Serializable pipeline records that intentionally omit VLM-irrelevant coordinates."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class PipelineResult:
    """Store the VLM-ready signals and optional generated retrieval-query output."""

    input: dict[str, Any]
    signals: dict[str, Any]
    vlm_input: dict[str, Any]
    output: dict[str, Any]
    icon_candidate_to_detected_name: dict[str, str] = field(default_factory=dict)
    retained_icon_crops: list[dict[str, Any]] = field(default_factory=list)
    schema_version: str = "1.0"

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-ready artifact."""
        return {"schema_version": self.schema_version, **asdict(self)}
