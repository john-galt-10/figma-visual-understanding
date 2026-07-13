"""Run a selected OCR backend against one screenshot and print JSON evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from ocr.base import OcrError  # noqa: E402
from ocr.factory import create_ocr_engine  # noqa: E402


def parse_arguments() -> argparse.Namespace:
    """Parse the screenshot path and OCR backend selected by the caller."""
    parser = argparse.ArgumentParser(
        description="Scan a raster screenshot with PaddleOCR or EasyOCR and print JSON."
    )
    parser.add_argument("--image-path", required=True, help="Path to a PNG, JPEG, WebP, BMP, or TIFF screenshot.")
    parser.add_argument("--engine", required=True, choices=("paddle", "easy"), help="OCR backend to run.")
    return parser.parse_args()


def main() -> int:
    """Run the selected OCR engine and print its normalized scan result."""
    arguments = parse_arguments()
    try:
        result = create_ocr_engine(arguments.engine).scan(arguments.image_path)
    except (OcrError, ValueError) as error:
        print(f"OCR scan failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
