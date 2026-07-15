"""Evaluate configured icon matching against a manually labeled crop JSONL manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from icon_matching.candidate_generation.base import CandidateGenerationError  # noqa: E402
from icon_matching.candidate_generation.config import load_settings  # noqa: E402
from icon_matching.matching.base import IconMatchingError, IconMatchingEvaluationError  # noqa: E402
from icon_matching.matching.evaluation import IconMatchingEvaluator, mine_detection_threshold  # noqa: E402
from icon_matching.matching.pipeline import IconLibraryMatcher  # noqa: E402


DEFAULT_MANIFEST_PATH = "outputs/icon-candidates/evaluation_candidates/crop_labeling.jsonl"
YELLOW = "\033[33m"
RESET = "\033[0m"


def parse_arguments() -> argparse.Namespace:
    """Parse manifest, matcher configuration, threshold, and optional report settings."""
    parser = argparse.ArgumentParser(description="Evaluate thresholded icon matching against labeled candidate crops.")
    parser.add_argument("--manifest-path", default=DEFAULT_MANIFEST_PATH, help=f"Labeled crop JSONL (default: {DEFAULT_MANIFEST_PATH}).")
    parser.add_argument("--config-path", default="icon_matching.yaml", help="Shared icon-matching YAML settings.")
    parser.add_argument("--detection-threshold", type=float, default=0.9, help="Minimum detection similarity from 0 through 1 (default: 0.9).")
    parser.add_argument(
        "--threshold-score",
        choices=("final", "primary", "secondary"),
        default="final",
        help="Top-match score used only to accept or reject a detection (default: final).",
    )
    parser.add_argument(
        "--mine-threshold",
        action="store_true",
        help="Re-evaluate using the smallest threshold that rejects every labeled non-icon crop.",
    )
    parser.add_argument("--json-output-path", help="Optional destination for detailed JSON evaluation results.")
    return parser.parse_args()


def main() -> int:
    """Evaluate all labeled crops, print aggregate metrics, and optionally save detailed results."""
    arguments = parse_arguments()
    if arguments.mine_threshold:
        print(
            f"{YELLOW}Warning: --mine-threshold provided; overriding --detection-threshold "
            "for the final evaluation result." + RESET
        )
    try:
        settings = load_settings(arguments.config_path)
        matcher = IconLibraryMatcher(settings)
        result = IconMatchingEvaluator(matcher).evaluate(
            manifest_path=arguments.manifest_path,
            detection_threshold=arguments.detection_threshold,
            detection_score_name=arguments.threshold_score,
        )
        if arguments.mine_threshold:
            mined_threshold = mine_detection_threshold(result)
            result = IconMatchingEvaluator(matcher).evaluate(
                manifest_path=arguments.manifest_path,
                detection_threshold=mined_threshold,
                detection_score_name=arguments.threshold_score,
            )
        if arguments.json_output_path:
            output_path = Path(arguments.json_output_path).expanduser().resolve()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(result.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    except (CandidateGenerationError, IconMatchingError, IconMatchingEvaluationError, OSError, ValueError) as error:
        print(f"Icon-matching evaluation failed: {error}", file=sys.stderr)
        return 1
    _print_summary(result)
    if arguments.mine_threshold:
        print(f"Threshold mined from non-icon scores: {result.detection_threshold:.17g}")
    if arguments.json_output_path:
        print(f"JSON report: {Path(arguments.json_output_path).expanduser().resolve()}")
    return 0


def _print_summary(result) -> None:
    """Print concise aggregate metrics for interactive experiment runs."""
    print(f"Detection score: {result.detection_score_name}; threshold: {result.detection_threshold:.4f}")
    print(f"Crops: {result.total_crops} total ({result.labeled_crops} labeled, {result.unlabeled_crops} non-icons)")
    print(f"TP: {result.true_positives}; FP: {result.false_positives}; FN: {result.false_negatives}; TN: {result.true_negatives}")
    print(f"Precision: {result.precision:.4f}; Recall: {result.recall:.4f}; MRR: {result.mean_reciprocal_rank:.4f}")
    print(f"End-to-end accuracy: {result.end_to_end_accuracy:.4f}; Detection accuracy: {result.detection_accuracy:.4f}")


if __name__ == "__main__":
    raise SystemExit(main())
