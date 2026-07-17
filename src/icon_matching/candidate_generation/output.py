"""Artifact writers for candidate JSON, crop images, and annotated overlays."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from .detectors import _cv2
from .models import CandidateGenerationResult


def write_artifacts(
    result: CandidateGenerationResult,
    image_path: str | Path,
    output_dir: str | Path,
    export_crops: bool,
    write_overlay: bool,
) -> Path:
    """Write the requested inspection artifacts and return the JSON artifact path."""
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    color_image = _read_bgr_image(Path(image_path))
    if export_crops:
        crops_dir = destination / "crops"
        crops_dir.mkdir(exist_ok=True)
        for candidate in result.candidates:
            crop = _crop(color_image, candidate.crop_bbox.x, candidate.crop_bbox.y, candidate.crop_bbox.right, candidate.crop_bbox.bottom)
            crop_path = crops_dir / f"{candidate.id}.png"
            _write_bgr_image(crop_path, crop)
            candidate.crop_path = str(crop_path)
            candidate.crop_dimensions = {"width": int(crop.shape[1]), "height": int(crop.shape[0])}
    if write_overlay:
        _write_overlay(destination / "overlay.png", color_image, result)
    json_path = destination / "candidates.json"
    with json_path.open("w", encoding="utf-8") as output_file:
        json.dump(result.to_dict(), output_file, indent=2, ensure_ascii=False)
    return json_path


def _read_bgr_image(path: Path) -> np.ndarray:
    """Read an image through Pillow and convert it to OpenCV's BGR arrangement."""
    cv2 = _cv2()
    with Image.open(path) as image:
        return cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)


def _write_bgr_image(path: Path, image: np.ndarray) -> None:
    """Save a BGR image through Pillow to keep output encoding consistent."""
    cv2 = _cv2()
    Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).save(path, format="PNG")


def _crop(image: np.ndarray, x: int, y: int, right: int, bottom: int) -> np.ndarray:
    """Return an already-clipped image slice for a candidate crop rectangle."""
    return image[y:bottom, x:right]


def _write_overlay(path: Path, image: np.ndarray, result: CandidateGenerationResult) -> None:
    """Draw final candidate boxes and detector provenance over the source screenshot."""
    settings = result.configuration["visualization"]
    annotations = [
        (candidate.content_bbox, candidate.id, _candidate_color(candidate.detector_sources, settings))
        for candidate in result.candidates
    ]
    _write_annotations(path, image, annotations, settings)


def write_detection_overlay(
    image_path: str | Path,
    output_path: str | Path,
    detections: list[tuple[object, str]],
    visualization_settings: dict[str, object],
) -> Path:
    """Write a numbered overlay for threshold-accepted icon detections only.

    Each detection is a ``BoundingBox``-compatible object plus its display label.
    The shared renderer deliberately keeps this VLM artifact separate from the
    raw-candidate diagnostic overlay used by candidate generation.
    """
    source = _read_bgr_image(Path(image_path))
    color = tuple(int(channel) for channel in visualization_settings["combined_color_bgr"])
    annotations = [(bbox, label, color) for bbox, label in detections]
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    _write_annotations(destination, source, annotations, visualization_settings)
    return destination


def _write_annotations(
    path: Path,
    image: np.ndarray,
    annotations: list[tuple[object, str, tuple[int, int, int]]],
    settings: dict[str, object],
) -> None:
    """Draw supplied bounding-box labels with the candidate overlay appearance."""
    cv2 = _cv2()
    overlay = image.copy()
    for bbox, label, color in annotations:
        x, y = int(getattr(bbox, "x")), int(getattr(bbox, "y"))
        right, bottom = int(getattr(bbox, "right")), int(getattr(bbox, "bottom"))
        cv2.rectangle(overlay, (x, y), (right, bottom), color, int(settings["line_width"]))
        cv2.putText(
            overlay,
            label,
            (x, max(12, y - 3)),
            cv2.FONT_HERSHEY_SIMPLEX,
            float(settings["label_font_scale"]),
            color,
            int(settings["line_width"]),
            cv2.LINE_AA,
        )
    _write_bgr_image(path, overlay)


def _candidate_color(sources: list[str], settings: dict[str, object]) -> tuple[int, int, int]:
    """Choose a provenance color for morphology, contours, or combined candidates."""
    if len(sources) > 1:
        color = settings["combined_color_bgr"]
    elif sources == ["morphology"]:
        color = settings["morphology_color_bgr"]
    else:
        color = settings["contours_color_bgr"]
    return tuple(int(channel) for channel in color)  # type: ignore[arg-type]
