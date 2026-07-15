"""Generate classical-CV icon candidates and inspectable artifacts for one screenshot."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from icon_matching.candidate_generation.base import CandidateGenerationError  # noqa: E402
from icon_matching.candidate_generation.config import load_settings  # noqa: E402
from icon_matching.candidate_generation.output import write_artifacts  # noqa: E402
from icon_matching.candidate_generation.pipeline import IconCandidateGenerator  # noqa: E402


def parse_arguments() -> argparse.Namespace:
    """Parse candidate-generation inputs and optional crop-file suppression."""
    parser = argparse.ArgumentParser(
        description="Propose likely icon regions in a Figma screenshot with classical CV."
    )
    parser.add_argument("--image-path", required=True, help="Path to the cropped Figma screenshot.")
    parser.add_argument("--config-path", required=True, help="Path to icon_matching.yaml.")
    parser.add_argument("--output-dir", required=True, help="Directory for JSON, overlay, and optional crops.")
    parser.add_argument(
        "--no-crop-files",
        action="store_true",
        help="Write only candidates.json and the optional overlay, without per-candidate PNG crops.",
    )
    return parser.parse_args()


def main() -> int:
    """Run candidate generation and write the configured inspection artifacts."""
    arguments = parse_arguments()
    try:
        settings = load_settings(arguments.config_path)
        result = IconCandidateGenerator(settings).generate(
            image_path=arguments.image_path,
            export_crops=not arguments.no_crop_files,
        )
        json_path = write_artifacts(
            result=result,
            image_path=arguments.image_path,
            output_dir=arguments.output_dir,
            export_crops=not arguments.no_crop_files,
            write_overlay=settings.visualization.enabled,
        )
    except (CandidateGenerationError, ValueError) as error:
        print(f"Icon candidate generation failed: {error}", file=sys.stderr)
        return 1
    print(
        f"Generated {result.counts['final_candidates']} candidates "
        f"from {result.counts['raw_proposals']} raw proposals. JSON: {json_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
