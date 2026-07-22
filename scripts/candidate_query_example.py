"""Generate Figma documentation retrieval-query candidates from a screenshot."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from candidate_queries.base import CandidateQueryError  # noqa: E402
from candidate_queries.config import load_settings  # noqa: E402
from candidate_queries.factory import create_candidate_query_generator  # noqa: E402


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
        result = generator.generate(
            arguments.image_path,
            arguments.text_query,
            output_trace=arguments.output_trace,
        )
    except (CandidateQueryError, ValueError) as error:
        print(f"Candidate query generation failed: {error}", file=sys.stderr)
        return 1
    print(result.model_dump_json(indent=2, exclude_none=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
