"""Prepare and validate optional full-screen context for VLM requests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw

from .base import CandidateQueryConfigurationError, CandidateQueryGenerator
from .models import ContextInput, FocusBox, ImageMetadata


CONTEXT_SYSTEM_INSTRUCTION = """
When multiple images are provided, follow the image roles defined in the Input description.
Treat the focused target image as the sole subject of retrieval queries. Use any full-screen
context image only to clarify visible state, selected-layer type, or referenced panel. Do not
make unrelated controls elsewhere on the screen the target.

When a detection overlay is provided, use its numbered boxes together with the detected-icon
mapping in Auxiliary visual evidence to identify the highlighted icons.
""".strip()


@dataclass(frozen=True)
class PreparedContextInput:
    """Keep the validated source, submitted image, and stable artifact metadata together."""

    context_input: ContextInput
    submission_path: Path


def prepare_context_input(
    context_image_path: str | Path | None,
    focus_bbox: FocusBox | None,
    output_directory: str | Path | None = None,
) -> PreparedContextInput | None:
    """Validate optional context and create a retained focus overlay when requested."""
    if context_image_path is None:
        if focus_bbox is not None:
            raise CandidateQueryConfigurationError("focus_bbox requires context_image_path.")
        return None

    source_metadata = CandidateQueryGenerator.read_image_metadata(context_image_path)
    if focus_bbox is None:
        return PreparedContextInput(
            context_input=ContextInput(
                source_image=source_metadata,
                submitted_image=source_metadata,
            ),
            submission_path=Path(source_metadata.path),
        )

    _validate_focus_bounds(focus_bbox, source_metadata)
    if output_directory is None:
        raise CandidateQueryConfigurationError(
            "An output directory is required to retain an annotated context screenshot."
        )
    destination = Path(output_directory).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    annotated_path = destination / "annotated_context_screenshot.png"
    _write_focus_overlay(Path(source_metadata.path), annotated_path, focus_bbox)
    submitted_metadata = CandidateQueryGenerator.read_image_metadata(annotated_path)
    return PreparedContextInput(
        context_input=ContextInput(
            source_image=source_metadata,
            submitted_image=submitted_metadata,
            focus_bbox=focus_bbox,
            annotated_context_image_path=str(annotated_path),
        ),
        submission_path=annotated_path,
    )


def _validate_focus_bounds(focus_bbox: FocusBox, context_image: ImageMetadata) -> None:
    """Reject a focus rectangle that is not wholly contained by the context image."""
    if focus_bbox.x + focus_bbox.width > context_image.width or focus_bbox.y + focus_bbox.height > context_image.height:
        raise CandidateQueryConfigurationError(
            "focus_bbox must stay completely within the context image bounds "
            f"({context_image.width}x{context_image.height})."
        )


def _write_focus_overlay(source_path: Path, destination: Path, focus_bbox: FocusBox) -> None:
    """Write a subtle, non-obscuring outline around the focused region."""
    with Image.open(source_path) as image:
        overlay = image.convert("RGB")
    draw = ImageDraw.Draw(overlay)
    left, top = focus_bbox.x, focus_bbox.y
    right, bottom = left + focus_bbox.width - 1, top + focus_bbox.height - 1
    line_width = max(2, min(5, round(min(overlay.width, overlay.height) / 400)))
    draw.rectangle((left, top, right, bottom), outline=(255, 105, 0), width=line_width)
    overlay.save(destination, format="PNG")
