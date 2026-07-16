"""Run the unified Figma visual-signal pipeline and print its VLM-ready evidence."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from visual_pipeline.config import PipelineConfigurationError, load_settings  # noqa: E402
from visual_pipeline.pipeline import VisualPipelineError, VisualSignalPipeline  # noqa: E402


def parse_arguments() -> argparse.Namespace:
    """Parse screenshot input, pipeline configuration, and inspection-output controls."""
    parser = argparse.ArgumentParser(
        description="Build inspectable OCR/icon signals and optionally generate Figma retrieval queries."
    )
    parser.add_argument("--image-path", required=True, help="Path to a supported Figma screenshot.")
    parser.add_argument("--text-query", help="Optional user question to refine with visual context.")
    parser.add_argument(
        "--config-path",
        type=Path,
        default=REPOSITORY_ROOT / "pipeline_config.yaml",
        help="Unified pipeline YAML configuration (default: %(default)s).",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        help=(
            "Optional JSON artifact path. By default, each run writes "
            "outputs/pipeline/<image-stem>_<timestamp>/results_details.json."
        ),
    )
    parser.add_argument(
        "--save-icon-crops",
        "--keep-crops",
        action="store_true",
        help="Keep candidate crop PNGs beside the JSON output and include their candidate mapping.",
    )
    parser.add_argument(
        "--no-vlm",
        action="store_true",
        help="Skip VLM generation for this run, overriding candidate_queries.enabled in the YAML.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the configured stages, print VLM input/output, and persist the JSON artifact."""
    arguments = parse_arguments()
    image_path = Path(arguments.image_path).expanduser().resolve()
    output_path = arguments.output_path or _default_output_path(image_path)
    output_path = output_path.expanduser().resolve()
    try:
        settings = load_settings(arguments.config_path)
        if arguments.no_vlm:
            settings.candidate_queries.enabled = False
        result = VisualSignalPipeline(settings).run(
            image_path=image_path,
            textual_query=arguments.text_query,
            save_icon_crops=arguments.save_icon_crops,
            output_directory=output_path.parent,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
        )
    except (PipelineConfigurationError, VisualPipelineError, OSError, ValueError) as error:
        print(f"Pipeline run failed: {error}", file=sys.stderr)
        return 1
    print("VLM input signals")
    print("=" * 17)
    print(f"User question: {result.vlm_input['textual_query'] or 'None supplied'}\n")
    print(result.vlm_input["auxiliary_visual_evidence"])
    print("\nVLM output")
    print("=" * 10)
    if result.output["vlm_enabled"]:
        for query in result.output["retrieval_queries"]:
            print(f"- {query}")
    else:
        print("VLM generation disabled by configuration; no tokens were used.")
    print(f"\nJSON artifact: {output_path}")
    return 0


def _default_output_path(image_path: Path) -> Path:
    """Create a unique, timestamped run folder and standardized JSON artifact name."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (
        REPOSITORY_ROOT
        / "outputs"
        / "pipeline"
        / f"{image_path.stem}_{timestamp}"
        / "results_details.json"
    )


if __name__ == "__main__":
    raise SystemExit(main())
