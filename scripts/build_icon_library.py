"""Build soft and binary icon templates from manually collected screenshots."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from icon_matching.template_library import IconTemplateLibraryBuilder, TemplateLibraryError  # noqa: E402
from icon_matching.candidate_generation.base import CandidateGenerationError  # noqa: E402
from icon_matching.candidate_generation.config import load_settings  # noqa: E402


def parse_arguments() -> argparse.Namespace:
    """Parse icon-library input, output, and common-canvas settings."""
    parser = argparse.ArgumentParser(
        description="Build normalized soft and binary icon templates from manual screenshots."
    )
    parser.add_argument(
        "--input-dir",
        default="assets/icon_library/raw_screenshots",
        help="Directory containing one manually collected icon screenshot per file.",
    )
    parser.add_argument(
        "--output-dir",
        default="assets/icon_library/templates",
        help="Directory under assets/icon_library that receives generated template artifacts.",
    )
    parser.add_argument(
        "--config-path",
        default="icon_matching.yaml",
        help="Shared icon-matching YAML settings, including template preprocessing.",
    )
    return parser.parse_args()


def main() -> int:
    """Build the template library and print the location of its manifest."""
    arguments = parse_arguments()
    try:
        settings = load_settings(arguments.config_path)
        preprocessing = settings.template_preprocessing
        artifacts = IconTemplateLibraryBuilder(
            canvas_size=preprocessing.canvas_size,
            canvas_margin=preprocessing.canvas_margin,
        ).build(arguments.input_dir, arguments.output_dir)
    except (CandidateGenerationError, TemplateLibraryError, ValueError) as error:
        print(f"Icon-library build failed: {error}", file=sys.stderr)
        return 1
    manifest_path = Path(arguments.output_dir).expanduser().resolve() / "templates.json"
    print(f"Built {len(artifacts)} icon templates. Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
