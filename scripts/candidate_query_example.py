"""Generate Figma documentation retrieval-query candidates from a screenshot."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from candidate_queries.base import CandidateQueryError  # noqa: E402
from candidate_queries.config import load_settings  # noqa: E402
from candidate_queries.factory import create_candidate_query_generator  # noqa: E402
from candidate_queries.models import FocusBox  # noqa: E402


def parse_arguments() -> argparse.Namespace:
    """Parse the screenshot, configuration, and optional user question for the CLI."""
    parser = argparse.ArgumentParser(
        description="Generate structured Figma documentation retrieval queries from a screenshot."
    )
    parser.add_argument(
        "--image-path",
        required=True,
        help="Path to a PNG, JPEG, WebP, BMP, or TIFF Figma screenshot.",
    )
    parser.add_argument(
        "--context-image-path",
        type=Path,
        help=(
            "Optional full-screen Figma screenshot. The required --image-path remains the "
            "focused retrieval target."
        ),
    )
    parser.add_argument(
        "--focus-bbox",
        nargs=4,
        type=int,
        metavar=("X", "Y", "WIDTH", "HEIGHT"),
        help=(
            "Optional zero-based focus rectangle in the context image. Requires "
            "--context-image-path and writes a retained annotated context image."
        ),
    )
    parser.add_argument(
        "--config-path",
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "query_gen_config.yaml",
        help=(
            "Path to the candidate-query YAML configuration file "
            "(default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--text-query",
        help="Optional user question to refine using the screenshot context.",
    )
    parser.add_argument(
        "--output-trace",
        action="store_true",
        default=None,
        help=(
            "Include a short reasoning summary in the JSON output, overriding the "
            "configuration setting."
        ),
    )
    return parser.parse_args()


def main() -> int:
    """Load configuration, generate queries, and print JSON to standard output."""
    arguments = parse_arguments()
    try:
        settings = load_settings(arguments.config_path)
        generator = create_candidate_query_generator(settings)
        focus_bbox = FocusBox(
            x=arguments.focus_bbox[0],
            y=arguments.focus_bbox[1],
            width=arguments.focus_bbox[2],
            height=arguments.focus_bbox[3],
        ) if arguments.focus_bbox is not None else None
        result = generator.generate(
            arguments.image_path,
            arguments.text_query,
            output_trace=arguments.output_trace,
            context_image_path=arguments.context_image_path,
            focus_bbox=focus_bbox,
            context_artifact_directory=(
                _default_context_artifact_directory(Path(arguments.image_path))
                if focus_bbox is not None
                else None
            ),
        )
    except (CandidateQueryError, ValueError) as error:
        print(f"Candidate query generation failed: {error}", file=sys.stderr)
        return 1
    print(result.model_dump_json(indent=2, exclude_none=True))
    return 0


def _default_context_artifact_directory(image_path: Path) -> Path:
    """Create an isolated default location for a standalone focus-overlay artifact."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return REPOSITORY_ROOT / "outputs" / "candidate-queries" / f"{image_path.stem}_{timestamp}"


if __name__ == "__main__":
    raise SystemExit(main())
