"""Mode strategies that pair VLM prompt evidence with screenshot inputs."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from icon_matching.candidate_generation.output import write_detection_overlay


@dataclass(frozen=True)
class VlmImageInput:
    """Describe one ordered local image submitted to the VLM."""

    path: Path
    role: str


@dataclass(frozen=True)
class InputModeSelection:
    """Contain mode-specific VLM images, evidence, and inspection metadata."""

    input_mode: str
    images: list[VlmImageInput]
    input_description: str
    evidence_block: str
    annotated_image_path: Path | None
    numbered_icon_mapping: dict[str, str]


class InputModeStrategy(ABC):
    """Define how one VLM input mode prepares images and icon prompt evidence."""

    mode_name: str
    input_description: str

    @abstractmethod
    def prepare(
        self,
        image_path: Path,
        ocr_signal: dict[str, Any],
        accepted_matches: list[dict[str, Any]],
        output_directory: Path | None,
        visualization_settings: dict[str, object],
    ) -> InputModeSelection:
        """Return the mode's VLM inputs after creating an overlay when required."""


class VanillaInputMode(InputModeStrategy):
    """Provide the source screenshot and legacy ordered icon-name evidence."""

    mode_name = "vanilla"
    input_description = "You are given one original Figma screenshot."

    def prepare(
        self,
        image_path: Path,
        ocr_signal: dict[str, Any],
        accepted_matches: list[dict[str, Any]],
        output_directory: Path | None,
        visualization_settings: dict[str, object],
    ) -> InputModeSelection:
        """Keep the original one-image VLM request behavior unchanged."""
        return InputModeSelection(
            input_mode=self.mode_name,
            images=[VlmImageInput(path=image_path, role="original_screenshot")],
            input_description=self.input_description,
            evidence_block=_render_evidence(
                ocr_signal,
                accepted_matches,
                numbered=False,
            ),
            annotated_image_path=None,
            numbered_icon_mapping={},
        )


class NumberedOverlayInputMode(InputModeStrategy):
    """Base strategy shared by modes that ground icon names in an overlay."""

    def prepare(
        self,
        image_path: Path,
        ocr_signal: dict[str, Any],
        accepted_matches: list[dict[str, Any]],
        output_directory: Path | None,
        visualization_settings: dict[str, object],
    ) -> InputModeSelection:
        """Create a retained accepted-icon overlay and numbered prompt mapping."""
        if output_directory is None:
            raise ValueError(
                f"An output directory is required when input_mode is '{self.mode_name}'."
            )
        annotated_image_path = output_directory / "annotated_screenshot.png"
        numbered_matches = [
            {**match, "overlay_number": index}
            for index, match in enumerate(accepted_matches, start=1)
        ]
        write_detection_overlay(
            image_path,
            annotated_image_path,
            [(match["content_bbox"], str(match["overlay_number"])) for match in numbered_matches],
            visualization_settings,
        )
        mapping = {
            str(match["overlay_number"]): str(match["detected_icon_name"])
            for match in numbered_matches
        }
        return InputModeSelection(
            input_mode=self.mode_name,
            images=self._images(image_path, annotated_image_path),
            input_description=self.input_description,
            evidence_block=_render_evidence(
                ocr_signal,
                numbered_matches,
                numbered=True,
            ),
            annotated_image_path=annotated_image_path,
            numbered_icon_mapping=mapping,
        )

    @abstractmethod
    def _images(self, image_path: Path, annotated_image_path: Path) -> list[VlmImageInput]:
        """Return the ordered image sequence required by this numbered mode."""


class SegmentedInputMode(NumberedOverlayInputMode):
    """Replace the screenshot with its accepted-icon numbered overlay."""

    mode_name = "segmented"
    input_description = (
        "You are given one annotated Figma screenshot. Numbered boxes highlight only "
        "icon detections accepted by the icon matcher. The detected-icon list below maps "
        "each number to its detector-suggested icon name; use it to locate the highlighted "
        "icon and correct the name only when the image clearly disagrees."
    )

    def _images(self, image_path: Path, annotated_image_path: Path) -> list[VlmImageInput]:
        """Send only the annotated screenshot to the VLM."""
        return [VlmImageInput(path=annotated_image_path, role="annotated_screenshot")]


class HybridInputMode(NumberedOverlayInputMode):
    """Send source context first and the accepted-icon overlay second."""

    mode_name = "hybrid"
    input_description = (
        "You are given two related Figma screenshots: first the original screenshot, then "
        "an annotated version of the same screenshot. Numbered boxes in the annotated image "
        "highlight only icon detections accepted by the icon matcher. Use the original image "
        "for unobscured UI details and broader context. The detected-icon list below maps each "
        "number to its detector-suggested icon name; use it to locate the highlighted icon and "
        "correct the name only when the image clearly disagrees."
    )

    def _images(self, image_path: Path, annotated_image_path: Path) -> list[VlmImageInput]:
        """Keep the original screenshot before its annotation for provider ordering."""
        return [
            VlmImageInput(path=image_path, role="original_screenshot"),
            VlmImageInput(path=annotated_image_path, role="annotated_screenshot"),
        ]


def create_input_mode_strategy(input_mode: str) -> InputModeStrategy:
    """Return the registered strategy selected by validated configuration."""
    strategies: dict[str, type[InputModeStrategy]] = {
        "vanilla": VanillaInputMode,
        "segmented": SegmentedInputMode,
        "hybrid": HybridInputMode,
    }
    try:
        return strategies[input_mode]()
    except KeyError as error:
        raise ValueError(f"Unsupported VLM input mode: {input_mode}") from error


def _render_evidence(
    ocr_signal: dict[str, Any],
    matches: list[dict[str, Any]],
    numbered: bool,
) -> str:
    """Render only OCR and icon observations for the user prompt evidence section."""
    lines = ["OCR-visible text:"]
    lines.extend(f"- {text}" for text in ocr_signal["visible_text"])
    if not ocr_signal["visible_text"]:
        lines.append("- None detected or OCR disabled.")
    if numbered:
        lines.extend(
            [
                "",
                "Detected icons in the annotated screenshot (number: detected icon name):",
            ]
        )
        lines.extend(
            f"- #{match['overlay_number']}: {match['detected_icon_name']}" for match in matches
        )
        if not matches:
            lines.append("- None detected or icon matching disabled.")
    else:
        lines.extend(["", "Detected icons (top-left to bottom-right):"])
        lines.extend(f"- {match['detected_icon_name']}" for match in matches)
        if not matches:
            lines.append("- None detected or icon matching disabled.")
    return "\n".join(lines)
