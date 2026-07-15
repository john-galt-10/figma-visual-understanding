"""Match one single-icon PNG against the configured local icon-template library."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from icon_matching.candidate_generation.config import load_settings  # noqa: E402
from icon_matching.candidate_generation.base import CandidateGenerationError  # noqa: E402
from icon_matching.matching.base import IconMatchingError  # noqa: E402
from icon_matching.matching.pipeline import IconLibraryMatcher  # noqa: E402


def parse_arguments() -> argparse.Namespace:
    """Parse one query image plus optional ranking and JSON-output overrides."""
    parser = argparse.ArgumentParser(description="Match one icon PNG against the local template library.")
    parser.add_argument("--image-path", required=True, help="PNG or other supported image containing one icon candidate.")
    parser.add_argument(
        "--config-path",
        default="icon_matching.yaml",
        help="Shared icon-matching YAML settings (default: icon_matching.yaml).",
    )
    parser.add_argument("--top-k", type=int, help="Override the configured number of displayed results.")
    parser.add_argument("--output-path", help="Optional JSON destination for the complete match run.")
    return parser.parse_args()


def main() -> int:
    """Match one normalized query icon and present its leading library results."""
    arguments = parse_arguments()
    try:
        settings = load_settings(arguments.config_path)
        result = IconLibraryMatcher(settings).match(arguments.image_path, arguments.top_k)
    except (CandidateGenerationError, IconMatchingError, ValueError) as error:
        print(f"Icon matching failed: {error}", file=sys.stderr)
        return 1
    _print_results(result)
    if arguments.output_path:
        output_path = Path(arguments.output_path).expanduser().resolve()
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(result.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        except OSError as error:
            print(f"Could not write JSON artifact '{output_path}': {error}", file=sys.stderr)
            return 1
        print(f"JSON artifact: {output_path}")
    return 0


def _print_results(result) -> None:
    """Print the compact ranked table intended for interactive experiment runs."""
    print(f"Primary matcher: {result.primary_matcher}; tie breaker: {result.tie_breaker or 'disabled'}")
    print(f"{'Rank':<6}{'Label':<32}{'Final':>10}{'Chamfer':>10}{'Soft NCC':>11}")
    for match in result.results:
        tie_breaker = "-" if match.tie_breaker_score is None else f"{match.tie_breaker_score:.4f}"
        print(
            f"{match.rank:<6}{match.label:<32}{match.final_score:>10.4f}"
            f"{match.primary_score:>10.4f}{tie_breaker:>11}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
