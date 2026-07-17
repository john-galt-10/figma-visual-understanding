"""Run a selected OCR backend against one screenshot and print JSON evidence."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from ocr.base import OcrError  # noqa: E402
from ocr.factory import create_ocr_engine  # noqa: E402
from ocr.filtering import filter_detections_by_confidence  # noqa: E402


def confidence_threshold(value: str) -> float:
    """Parse one OCR confidence threshold constrained to the inclusive unit interval."""
    try:
        threshold = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a number between 0.0 and 1.0") from error
    if not 0.0 <= threshold <= 1.0:
        raise argparse.ArgumentTypeError("must be between 0.0 and 1.0")
    return threshold


def parse_arguments() -> argparse.Namespace:
    """Parse the screenshot path and OCR backend selected by the caller."""
    parser = argparse.ArgumentParser(
        description="Scan a raster screenshot with PaddleOCR or EasyOCR and print JSON."
    )
    parser.add_argument("--image-path", required=True, help="Path to a PNG, JPEG, WebP, BMP, or TIFF screenshot.")
    parser.add_argument("--engine", required=True, choices=("paddle", "easy"), help="OCR backend to run.")
    parser.add_argument(
        "--detection-threshold",
        type=confidence_threshold,
        help="Optional minimum OCR confidence from 0.0 through 1.0; omitted preserves raw detections.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        help="Optional path that also receives the final JSON output.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the selected OCR engine and print its normalized scan result."""
    arguments = parse_arguments()
    try:
        result = create_ocr_engine(arguments.engine).scan(arguments.image_path)
    except (OcrError, ValueError) as error:
        print(f"OCR scan failed: {error}", file=sys.stderr)
        return 1
    if arguments.detection_threshold is not None:
        original_detection_count = result.detection_count
        detections = filter_detections_by_confidence(result.detections, arguments.detection_threshold)
        result = replace(
            result,
            detections=detections,
            visible_text="\n".join(detection.text for detection in detections),
            detection_count=len(detections),
            metadata={
                **result.metadata,
                "confidence_filter": {
                    "detection_threshold": arguments.detection_threshold,
                    "original_detection_count": original_detection_count,
                },
            },
        )
    output_json = json.dumps(result.to_dict(), indent=2, ensure_ascii=False)
    if arguments.output_path is not None:
        output_path = arguments.output_path.expanduser().resolve()
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(f"{output_json}\n", encoding="utf-8")
        except OSError as error:
            print(f"Could not write OCR JSON output: {error}", file=sys.stderr)
            return 1
    print(output_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
