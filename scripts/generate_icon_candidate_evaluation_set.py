"""Generate a flat, provenance-tracked set of icon-candidate crops from screenshots."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from PIL import Image


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from icon_matching.candidate_generation.base import CandidateGenerationError  # noqa: E402
from icon_matching.candidate_generation.config import load_settings  # noqa: E402
from icon_matching.candidate_generation.pipeline import IconCandidateGenerator  # noqa: E402


DEFAULT_OUTPUT_DIR = "outputs/icon-candidates/evaluation_candidates"
MANIFEST_FILENAME = "crop_provenance.jsonl"


def parse_arguments() -> argparse.Namespace:
    """Parse the screenshot list, shared generator configuration, and output settings."""
    parser = argparse.ArgumentParser(
        description="Generate flat icon-candidate crops and a JSONL crop-provenance manifest."
    )
    parser.add_argument(
        "--input-list-path",
        default="screenshots_example/screenshots_example.lst",
        help="Path to a .lst file containing one screenshot path per line.",
    )
    parser.add_argument(
        "--config-path",
        default="icon_matching.yaml",
        help="Shared icon-candidate YAML settings (default: icon_matching.yaml).",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Empty destination for flat PNG crops and {MANIFEST_FILENAME} (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--clear-output",
        action="store_true",
        help="Delete existing output-folder contents before generating artifacts.",
    )
    return parser.parse_args()


def load_screenshot_paths(list_path: Path) -> list[Path]:
    """Read non-empty, non-comment list entries and resolve relative paths from the list file."""
    paths: list[Path] = []
    for line_number, line in enumerate(list_path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        entry = line.strip()
        if not entry or entry.startswith("#"):
            continue
        screenshot_path = Path(entry).expanduser()
        if not screenshot_path.is_absolute():
            screenshot_path = list_path.parent / screenshot_path
        screenshot_path = screenshot_path.resolve()
        if not screenshot_path.is_file():
            raise ValueError(f"Screenshot on line {line_number} does not exist: {screenshot_path}")
        paths.append(screenshot_path)
    if not paths:
        raise ValueError(f"Screenshot list contains no usable paths: {list_path}")
    return paths


def prepare_output_directory(output_dir: Path, clear_output: bool) -> None:
    """Create an empty output folder, preserving existing data unless explicitly cleared."""
    if output_dir.exists() and any(output_dir.iterdir()):
        if not clear_output:
            raise ValueError(
                f"Output directory is not empty: {output_dir}. Use --clear-output to replace its contents."
            )
        for child in output_dir.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)


def write_candidate_crop(image: Image.Image, crop_bbox: dict[str, int], destination: Path) -> dict[str, int]:
    """Crop a candidate rectangle from an RGB screenshot, save it as PNG, and return its dimensions."""
    x = crop_bbox["x"]
    y = crop_bbox["y"]
    width = crop_bbox["width"]
    height = crop_bbox["height"]
    crop = image.crop((x, y, x + width, y + height))
    crop.save(destination, format="PNG")
    return {"width": crop.width, "height": crop.height}


def main() -> int:
    """Generate all crops and write one JSONL record per crop with source provenance."""
    arguments = parse_arguments()
    try:
        list_path = Path(arguments.input_list_path).expanduser().resolve()
        if list_path.suffix.lower() != ".lst":
            raise ValueError(f"Input list must use the .lst extension: {list_path}")
        if not list_path.is_file():
            raise ValueError(f"Input list does not exist: {list_path}")
        screenshot_paths = load_screenshot_paths(list_path)
        output_dir = Path(arguments.output_dir).expanduser().resolve()
        prepare_output_directory(output_dir, arguments.clear_output)
        settings = load_settings(arguments.config_path)
        generator = IconCandidateGenerator(settings)

        crop_count = 0
        manifest_path = output_dir / MANIFEST_FILENAME
        with manifest_path.open("w", encoding="utf-8") as manifest_file:
            for source_index, screenshot_path in enumerate(screenshot_paths, start=1):
                result = generator.generate(image_path=screenshot_path, export_crops=False)
                with Image.open(screenshot_path) as source_image:
                    rgb_image = source_image.convert("RGB")
                    for candidate in result.candidates:
                        candidate_data = candidate.to_dict()
                        crop_bbox = candidate_data["crop_bbox"]
                        crop_filename = f"{source_index:04d}_{screenshot_path.stem}_{candidate.id}.png"
                        crop_path = output_dir / crop_filename
                        crop_dimensions = write_candidate_crop(rgb_image, crop_bbox, crop_path)
                        record = {
                            "crop_path": crop_filename,
                            "source_image_path": str(screenshot_path),
                            "source_candidate_id": candidate.id,
                            "label": "",
                            "content_bbox": candidate_data["content_bbox"],
                            "crop_bbox": crop_bbox,
                            "crop_dimensions": crop_dimensions,
                            "detector_sources": candidate.detector_sources,
                            "proposal_score": candidate.proposal_score,
                        }
                        manifest_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                        crop_count += 1
    except (CandidateGenerationError, OSError, ValueError) as error:
        print(f"Evaluation-crop generation failed: {error}", file=sys.stderr)
        return 1

    print(f"Generated {crop_count} crops from {len(screenshot_paths)} screenshots. Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
