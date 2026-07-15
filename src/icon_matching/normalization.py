"""Shared preprocessing for icon-library construction and query matching."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


class IconNormalizationError(RuntimeError):
    """Raised when an image cannot be converted into a usable icon mask."""


@dataclass(frozen=True)
class NormalizedIcon:
    """Contain soft and binary icon representations derived from one source image."""

    soft_crop: np.ndarray
    binary_crop: np.ndarray
    soft_canvas: np.ndarray
    binary_canvas: np.ndarray
    glyph_bbox: dict[str, int]
    threshold: int


class IconNormalizer:
    """Create consistently cropped and centered icon masks from light-on-dark images."""

    def __init__(self, canvas_size: int = 32, canvas_margin: int = 2) -> None:
        """Configure the common square output canvas and its empty border."""
        if canvas_size < 3:
            raise ValueError("canvas_size must be at least 3 pixels")
        if canvas_margin < 0 or canvas_margin * 2 >= canvas_size:
            raise ValueError("canvas_margin must leave room for the glyph")
        self.canvas_size = canvas_size
        self.canvas_margin = canvas_margin

    def normalize_path(self, image_path: str | Path) -> NormalizedIcon:
        """Load and normalize one supported raster image from disk."""
        path = Path(image_path).expanduser().resolve()
        if not path.is_file():
            raise IconNormalizationError(f"Image file does not exist: {path}")
        try:
            with Image.open(path) as image:
                grayscale = np.array(image.convert("L"), dtype=np.uint8)
        except (OSError, ValueError) as error:
            raise IconNormalizationError(f"Could not read image '{path}': {error}") from error
        return self.normalize_grayscale(grayscale, path)

    def normalize_grayscale(self, grayscale: np.ndarray, source: str | Path = "image") -> NormalizedIcon:
        """Normalize one grayscale image array into crop and canvas representations."""
        if grayscale.ndim != 2 or grayscale.size == 0:
            raise IconNormalizationError(f"'{source}' is not a non-empty grayscale image.")
        pixels = grayscale.astype(np.uint8, copy=False)
        threshold, binary = cv2.threshold(pixels, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        points = cv2.findNonZero(binary)
        if points is None:
            raise IconNormalizationError(f"No light glyph foreground found in '{source}'.")
        x, y, width, height = (int(value) for value in cv2.boundingRect(points))
        soft = self._soft_mask(pixels, source)
        soft_crop = soft[y : y + height, x : x + width]
        binary_crop = binary[y : y + height, x : x + width]
        return NormalizedIcon(
            soft_crop=soft_crop,
            binary_crop=binary_crop,
            soft_canvas=self._standard_canvas(soft_crop, is_binary=False),
            binary_canvas=self._standard_canvas(binary_crop, is_binary=True),
            glyph_bbox={"x": x, "y": y, "width": width, "height": height},
            threshold=int(threshold),
        )

    @staticmethod
    def _soft_mask(grayscale: np.ndarray, source: str | Path) -> np.ndarray:
        """Normalize brightness so the dark background becomes black and glyph stays soft."""
        background_level = float(np.percentile(grayscale, 5))
        foreground_level = float(np.percentile(grayscale, 99))
        if foreground_level <= background_level:
            raise IconNormalizationError(f"'{source}' has insufficient contrast for a soft mask.")
        normalized = (grayscale.astype(np.float32) - background_level) * 255.0
        normalized /= foreground_level - background_level
        return np.clip(normalized, 0, 255).astype(np.uint8)

    def _standard_canvas(self, crop: np.ndarray, is_binary: bool) -> np.ndarray:
        """Center a crop on the common canvas while preserving its glyph proportions."""
        available_size = self.canvas_size - 2 * self.canvas_margin
        height, width = crop.shape
        scale = min(available_size / width, available_size / height)
        resized_width = max(1, round(width * scale))
        resized_height = max(1, round(height * scale))
        interpolation = cv2.INTER_NEAREST if is_binary else cv2.INTER_LANCZOS4
        resized = cv2.resize(crop, (resized_width, resized_height), interpolation=interpolation)
        if is_binary:
            resized = np.where(resized >= 128, 255, 0).astype(np.uint8)
        canvas = np.zeros((self.canvas_size, self.canvas_size), dtype=np.uint8)
        x = (self.canvas_size - resized_width) // 2
        y = (self.canvas_size - resized_height) // 2
        canvas[y : y + resized_height, x : x + resized_width] = resized
        return canvas
