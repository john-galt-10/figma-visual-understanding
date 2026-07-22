"""Run visual-query annotations through the unified pipeline for later evaluation."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from visual_pipeline.config import PipelineConfigurationError, load_settings  # noqa: E402
from visual_pipeline.pipeline import VisualPipelineError, VisualSignalPipeline  # noqa: E402


DEFAULT_INPUT_PATH = REPOSITORY_ROOT / "annotation" / "visual_query_annotations.jsonl"
DEFAULT_OUTPUT_PARENT = REPOSITORY_ROOT / "eval" / "results" / "query-generation"
REQUIRED_ANNOTATION_FIELDS = (
    "screenshot_id",
    "image_path",
    "text_query",
    "target_element_name",
    "intent",
)
CONFIG_SNAPSHOT_FILENAME = "config.yaml"
GENERATED_QUERIES_FILENAME = "generated_queries.jsonl"


def parse_arguments() -> argparse.Namespace:
    """Parse the annotation dataset, pipeline configuration, and artifact controls."""
    parser = argparse.ArgumentParser(
        description="Generate self-contained query-generation evaluation records from annotations."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Visual-query annotation JSONL (default: %(default)s).",
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=REPOSITORY_ROOT / "pipeline_config.yaml",
        help="Unified pipeline YAML configuration (default: %(default)s).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Destination results folder. By default, writes "
            "eval/results/query-generation/<input-stem>_<YYYYMMDD_HHMMSS>/"
        ),
    )
    parser.add_argument(
        "--save-icon-crops",
        "--keep-crops",
        action="store_true",
        help="Retain icon crop PNGs in an icon_crops directory inside the results folder.",
    )
    return parser.parse_args()


def load_annotations(input_path: Path) -> list[dict[str, Any]]:
    """Read and validate the annotation JSONL while retaining its record order."""
    if not input_path.is_file():
        raise ValueError(f"Input annotation file does not exist: {input_path}")

    annotations: list[dict[str, Any]] = []
    for line_number, line in enumerate(input_path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSON in {input_path} at line {line_number}.") from error
        if not isinstance(record, dict):
            raise ValueError(f"Expected an object in {input_path} at line {line_number}.")
        missing_or_invalid = [
            field
            for field in REQUIRED_ANNOTATION_FIELDS
            if not isinstance(record.get(field), str) or not record[field].strip()
        ]
        if missing_or_invalid:
            fields = ", ".join(missing_or_invalid)
            raise ValueError(f"Invalid annotation schema in {input_path} at line {line_number}: {fields}.")
        annotations.append(record)
    if not annotations:
        raise ValueError(f"Input annotation file contains no records: {input_path}")
    return annotations


def resolve_image_path(annotation_path: str) -> Path:
    """Resolve repository-relative annotation image paths for pipeline input."""
    path = Path(annotation_path).expanduser()
    if not path.is_absolute():
        path = REPOSITORY_ROOT / path
    return path.resolve()


def default_output_directory(input_path: Path, timestamp: str) -> Path:
    """Build the timestamped default folder name from the input JSONL stem."""
    return DEFAULT_OUTPUT_PARENT / f"{input_path.stem}_{timestamp}"


def artifact_output_directory(
    results_directory: Path, annotation: dict[str, Any], index: int, retain_artifacts: bool
) -> Path:
    """Return an isolated pipeline output directory only when a run writes visual artifacts."""
    if not retain_artifacts:
        return results_directory
    safe_screenshot_id = "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in annotation["screenshot_id"]
    )
    return results_directory / "artifacts" / f"{index:04d}_{safe_screenshot_id}"


def relocate_artifact_paths(value: Any, temporary_root: Path, final_root: Path) -> Any:
    """Replace temporary output paths in a serialized result with their final locations."""
    temporary_prefix = str(temporary_root)
    final_prefix = str(final_root)
    if isinstance(value, dict):
        return {
            key: relocate_artifact_paths(item, temporary_root, final_root)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [relocate_artifact_paths(item, temporary_root, final_root) for item in value]
    if isinstance(value, str) and value.startswith(temporary_prefix):
        return final_prefix + value[len(temporary_prefix) :]
    return value


def build_record(
    annotation: dict[str, Any],
    input_path: Path,
    config_path: Path,
    settings: Any,
    generated_at: str,
    result: dict[str, Any] | None = None,
    error: Exception | None = None,
) -> dict[str, Any]:
    """Create one evaluable success or failure record without annotation dependencies."""
    image_path = resolve_image_path(annotation["image_path"])
    record: dict[str, Any] = {
        "schema_version": "1.0",
        "status": "succeeded" if error is None else "failed",
        "generated_at": generated_at,
        "annotation": annotation,
        "image": {
            "annotation_image_path": annotation["image_path"],
            "resolved_image_path": str(image_path),
        },
        "provenance": {
            "input_annotation_path": str(input_path),
            "source_config_path": str(config_path),
            "config_snapshot_path": CONFIG_SNAPSHOT_FILENAME,
        },
        "effective_pipeline_config": settings.model_dump(mode="json"),
    }
    if error is None:
        record["pipeline_result"] = result
    else:
        record["error"] = {
            "stage": "visual_pipeline",
            "type": type(error).__name__,
            "message": str(error),
        }
    return record


def main() -> int:
    """Generate a complete query-evaluation artifact folder from annotated screenshots."""
    arguments = parse_arguments()
    temporary_output_directory: Path | None = None
    try:
        input_path = arguments.input_path.expanduser().resolve()
        config_path = arguments.config_path.expanduser().resolve()
        annotations = load_annotations(input_path)
        settings = load_settings(config_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
        output_directory = (
            arguments.output_dir.expanduser().resolve()
            if arguments.output_dir is not None
            else default_output_directory(input_path, timestamp)
        )
        if output_directory.exists():
            raise ValueError(f"Output directory already exists: {output_directory}")
        output_directory.parent.mkdir(parents=True, exist_ok=True)
        temporary_output_directory = Path(
            tempfile.mkdtemp(prefix=f".{output_directory.name}.", dir=output_directory.parent)
        )
        shutil.copyfile(config_path, temporary_output_directory / CONFIG_SNAPSHOT_FILENAME)

        pipeline = VisualSignalPipeline(settings)
        failures: list[str] = []
        results_path = temporary_output_directory / GENERATED_QUERIES_FILENAME
        with results_path.open("w", encoding="utf-8", newline="\n") as results_file:
            retain_artifacts = arguments.save_icon_crops or settings.candidate_queries.input_mode != "vanilla"
            for index, annotation in enumerate(annotations, start=1):
                try:
                    image_path = resolve_image_path(annotation["image_path"])
                    if not image_path.is_file():
                        raise ValueError(f"Annotation image does not exist: {image_path}")
                    run_output_directory = artifact_output_directory(
                        temporary_output_directory, annotation, index, retain_artifacts
                    )
                    if retain_artifacts:
                        run_output_directory.mkdir(parents=True, exist_ok=True)
                    result = pipeline.run(
                        image_path=image_path,
                        textual_query=annotation["text_query"],
                        save_icon_crops=arguments.save_icon_crops,
                        output_directory=run_output_directory,
                    )
                    serialized_result = relocate_artifact_paths(
                        result.to_dict(), temporary_output_directory, output_directory
                    )
                    record = build_record(
                        annotation, input_path, config_path, settings, generated_at, result=serialized_result
                    )
                except (VisualPipelineError, OSError, ValueError) as error:
                    failures.append(annotation["screenshot_id"])
                    record = build_record(
                        annotation, input_path, config_path, settings, generated_at, error=error
                    )
                results_file.write(json.dumps(record, ensure_ascii=False) + "\n")

        temporary_output_directory.replace(output_directory)
        temporary_output_directory = None
    except (PipelineConfigurationError, OSError, ValueError) as error:
        if temporary_output_directory is not None:
            shutil.rmtree(temporary_output_directory, ignore_errors=True)
        print(f"Query-generation evaluation failed: {error}", file=sys.stderr)
        return 1

    print(f"Processed {len(annotations)} annotations. Results folder: {output_directory}")
    if failures:
        print(f"Failed {len(failures)} annotations: {', '.join(failures)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
