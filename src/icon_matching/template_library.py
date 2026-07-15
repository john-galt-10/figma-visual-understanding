"""Build inspectable soft and binary icon templates from manual screenshots."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .normalization import IconNormalizationError, IconNormalizer


class TemplateLibraryError(RuntimeError):
    """Raised when an icon-template library cannot be built safely."""


@dataclass(frozen=True)
class TemplateArtifact:
    """Describe all template files generated for one labeled source image."""

    label: str
    source_path: str
    glyph_bbox: dict[str, int]
    crop_dimensions: dict[str, int]
    canvas_dimensions: dict[str, int]
    threshold: int
    soft_crop_path: str
    binary_crop_path: str
    soft_canvas_path: str
    binary_canvas_path: str

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready representation of this generated template."""
        return asdict(self)


class IconTemplateLibraryBuilder:
    """Convert single-glyph manual screenshots into normalized matching templates."""

    def __init__(self, canvas_size: int = 32, canvas_margin: int = 2) -> None:
        """Configure the square output canvas and its empty border in pixels."""
        if canvas_size < 3:
            raise ValueError("canvas_size must be at least 3 pixels")
        if canvas_margin < 0 or canvas_margin * 2 >= canvas_size:
            raise ValueError("canvas_margin must leave room for the glyph")
        self.canvas_size = canvas_size
        self.canvas_margin = canvas_margin
        self.normalizer = IconNormalizer(canvas_size, canvas_margin)

    def build(self, input_dir: str | Path, output_dir: str | Path) -> list[TemplateArtifact]:
        """Build templates for every supported image and write them to ``output_dir``."""
        sources = self._find_sources(Path(input_dir))
        destination = Path(output_dir).expanduser().resolve()
        artifacts: list[TemplateArtifact] = []
        labels: set[str] = set()
        for source_path in sources:
            label = source_path.stem
            if label in labels:
                raise TemplateLibraryError(f"Duplicate template label '{label}'.")
            labels.add(label)
            artifacts.append(self._build_one(source_path, destination, label))
        self._write_manifest(destination, artifacts)
        return artifacts

    @staticmethod
    def _find_sources(input_dir: Path) -> list[Path]:
        """Return deterministic supported-image paths or raise a useful input error."""
        source = input_dir.expanduser().resolve()
        if not source.is_dir():
            raise TemplateLibraryError(f"Input directory does not exist: {source}")
        supported_extensions = {".png", ".jpg", ".jpeg", ".webp"}
        images = sorted(path for path in source.rglob("*") if path.is_file() and path.suffix.lower() in supported_extensions)
        if not images:
            raise TemplateLibraryError(f"No supported images found in: {source}")
        return images

    def _build_one(self, source_path: Path, destination: Path, label: str) -> TemplateArtifact:
        """Extract one glyph and write its crop and standard-canvas representations."""
        try:
            normalized = self.normalizer.normalize_path(source_path)
        except IconNormalizationError as error:
            raise TemplateLibraryError(str(error)) from error
        paths = self._artifact_paths(destination, label)
        for path in paths.values():
            path.parent.mkdir(parents=True, exist_ok=True)
        self._write_image(paths["soft_crop"], normalized.soft_crop)
        self._write_image(paths["binary_crop"], normalized.binary_crop)
        self._write_image(paths["soft_canvas"], normalized.soft_canvas)
        self._write_image(paths["binary_canvas"], normalized.binary_canvas)
        return TemplateArtifact(
            label=label,
            source_path=str(source_path.resolve()),
            glyph_bbox=normalized.glyph_bbox,
            crop_dimensions={"width": int(normalized.soft_crop.shape[1]), "height": int(normalized.soft_crop.shape[0])},
            canvas_dimensions={"width": self.canvas_size, "height": self.canvas_size},
            threshold=normalized.threshold,
            soft_crop_path=str(paths["soft_crop"]),
            binary_crop_path=str(paths["binary_crop"]),
            soft_canvas_path=str(paths["soft_canvas"]),
            binary_canvas_path=str(paths["binary_canvas"]),
        )

    @staticmethod
    def _artifact_paths(destination: Path, label: str) -> dict[str, Path]:
        """Return the fixed, inspectable output locations for one label."""
        return {
            "soft_crop": destination / "soft" / "crops" / f"{label}.png",
            "binary_crop": destination / "binary" / "crops" / f"{label}.png",
            "soft_canvas": destination / "soft" / "canvases" / f"{label}.png",
            "binary_canvas": destination / "binary" / "canvases" / f"{label}.png",
        }

    @staticmethod
    def _write_image(path: Path, pixels: np.ndarray) -> None:
        """Write a single-channel template image as a PNG."""
        Image.fromarray(pixels, mode="L").save(path, format="PNG")

    @staticmethod
    def _write_manifest(destination: Path, artifacts: list[TemplateArtifact]) -> None:
        """Record generated template metadata in a stable JSON manifest."""
        destination.mkdir(parents=True, exist_ok=True)
        manifest = {
            "schema_version": "1.0",
            "template_count": len(artifacts),
            "templates": [artifact.to_dict() for artifact in artifacts],
        }
        with (destination / "templates.json").open("w", encoding="utf-8") as output_file:
            json.dump(manifest, output_file, indent=2, ensure_ascii=False)
