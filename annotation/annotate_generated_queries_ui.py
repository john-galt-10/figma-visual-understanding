"""Launch a local Gradio interface for evaluating generated Figma query candidates."""

from __future__ import annotations

import argparse
import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import gradio as gr


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
GENERATED_QUERIES_FILENAME = "generated_queries.jsonl"
ANNOTATIONS_FILENAME = "query_annotations.jsonl"
GROUNDING_CHOICES = ["true", "false", "acceptable"]


def parse_arguments() -> argparse.Namespace:
    """Parse generated-query input, annotation output, and local-server options."""
    parser = argparse.ArgumentParser(
        description="Annotate human-quality ratings for generated Figma query candidates."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        required=True,
        help=f"Generated-query JSONL, normally {GENERATED_QUERIES_FILENAME}.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        help=(
            "Reviewer annotation JSONL. Defaults to "
            f"{ANNOTATIONS_FILENAME} beside --input-path."
        ),
    )
    parser.add_argument("--port", type=int, help="Optional local Gradio server port.")
    return parser.parse_args()


def load_generated_records(input_path: Path) -> list[dict[str, Any]]:
    """Load generated-query records in their source JSONL order."""
    if not input_path.is_file():
        raise ValueError(f"Generated-query file does not exist: {input_path}")
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(input_path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSON in {input_path} at line {line_number}.") from error
        if not isinstance(record, dict) or not isinstance(record.get("annotation"), dict):
            raise ValueError(f"Invalid generated-query record in {input_path} at line {line_number}.")
        records.append(record)
    if not records:
        raise ValueError(f"Generated-query file contains no records: {input_path}")
    return records


def generated_queries(record: dict[str, Any]) -> list[str]:
    """Return usable candidate queries from one successful pipeline record."""
    output = record.get("pipeline_result", {}).get("output", {})
    queries = output.get("retrieval_queries", []) if isinstance(output, dict) else []
    return [query.strip() for query in queries if isinstance(query, str) and query.strip()]


def is_reviewable(record: dict[str, Any]) -> bool:
    """Return whether a record has generated queries that can receive quality ratings."""
    return record.get("status") == "succeeded" and bool(generated_queries(record))


def load_annotations(output_path: Path, record_count: int) -> dict[int, dict[str, Any]]:
    """Load saved reviewer annotations keyed by their one-based source-record index."""
    if not output_path.exists():
        return {}
    annotations: dict[int, dict[str, Any]] = {}
    for line_number, line in enumerate(output_path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSON in {output_path} at line {line_number}.") from error
        index = record.get("source_record_index") if isinstance(record, dict) else None
        if not isinstance(index, int) or not 1 <= index <= record_count:
            raise ValueError(f"Invalid annotation schema in {output_path} at line {line_number}.")
        annotations[index] = record
    return annotations


def write_annotations(
    output_path: Path, records: list[dict[str, Any]], annotations: dict[int, dict[str, Any]]
) -> None:
    """Atomically rewrite saved annotations in generated-query source order."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        delete=False,
    ) as file_handle:
        for index, _ in enumerate(records, start=1):
            if index in annotations:
                file_handle.write(json.dumps(annotations[index], ensure_ascii=False) + "\n")
        temporary_path = Path(file_handle.name)
    temporary_path.replace(output_path)


def first_unannotated_index(records: list[dict[str, Any]], annotations: dict[int, dict[str, Any]]) -> int:
    """Return the first reviewable, unsaved zero-based record index."""
    return next(
        (
            index
            for index, record in enumerate(records)
            if is_reviewable(record) and index + 1 not in annotations
        ),
        0,
    )


def _screenshot_path(record: dict[str, Any]) -> str | None:
    """Find the source screenshot path retained in a generated-query record."""
    image = record.get("image", {})
    resolved_path = image.get("resolved_image_path") if isinstance(image, dict) else None
    if isinstance(resolved_path, str) and Path(resolved_path).is_file():
        return resolved_path
    annotation = record.get("annotation", {})
    annotation_path = annotation.get("image_path") if isinstance(annotation, dict) else None
    if isinstance(annotation_path, str):
        candidate = Path(annotation_path)
        if not candidate.is_absolute():
            candidate = REPOSITORY_ROOT / candidate
        if candidate.is_file():
            return str(candidate.resolve())
    return None


def _unavailable_message(record: dict[str, Any]) -> str:
    """Explain why a source record cannot receive a human-quality annotation."""
    if record.get("status") == "failed":
        error = record.get("error", {})
        message = error.get("message", "Unknown pipeline error") if isinstance(error, dict) else "Unknown pipeline error"
        return f"**Unavailable:** pipeline run failed: {message}"
    output = record.get("pipeline_result", {}).get("output", {})
    if isinstance(output, dict) and not output.get("vlm_enabled", True):
        return "**Unavailable:** VLM query generation was disabled for this record."
    return "**Unavailable:** this record has no generated candidate queries."


def _view_values(
    index: int, records: list[dict[str, Any]], annotations: dict[int, dict[str, Any]]
) -> tuple[object, ...]:
    """Create UI component values for one generated-query record."""
    record = records[index]
    source_index = index + 1
    source_annotation = record["annotation"]
    saved_annotation = annotations.get(source_index, {})
    reviewable = is_reviewable(record)
    reviewable_total = sum(is_reviewable(item) for item in records)
    progress = (
        f"**Progress:** {len(annotations)} / {reviewable_total} reviewable records saved "
        f"({len(records) - reviewable_total} unavailable)"
    )
    queries = generated_queries(record)
    query_markdown = "\n".join(f"{number}. {query}" for number, query in enumerate(queries, start=1))
    output = record.get("pipeline_result", {}).get("output", {})
    vlm_input = record.get("pipeline_result", {}).get("vlm_input", {})
    evidence = vlm_input.get("auxiliary_visual_evidence", "No evidence retained.") if isinstance(vlm_input, dict) else "No evidence retained."
    prompt = vlm_input.get("user_prompt", "No prompt retained.") if isinstance(vlm_input, dict) else "No prompt retained."
    generator = output.get("generator", {}) if isinstance(output, dict) else {}
    generator_text = json.dumps(generator, ensure_ascii=False, indent=2) if generator else "No generator metadata retained."
    reasoning_summary = output.get("reasoning_summary") if isinstance(output, dict) else None
    if not isinstance(reasoning_summary, str) or not reasoning_summary.strip():
        reasoning_summary = ""
    availability = "**Ready for review.**" if reviewable else _unavailable_message(record)
    return (
        _screenshot_path(record),
        f"### Record {source_index}: `{source_annotation.get('screenshot_id', 'unknown')}`",
        progress,
        availability,
        source_annotation.get("text_query", ""),
        source_annotation.get("target_element_name", ""),
        source_annotation.get("intent", ""),
        f"## Candidate queries\n\n{query_markdown or 'No generated candidate queries.'}",
        prompt,
        evidence,
        generator_text,
        gr.Textbox(value=reasoning_summary, visible=bool(reasoning_summary), interactive=False),
        gr.Dropdown(value=saved_annotation.get("is_target_correct"), interactive=reviewable),
        gr.Dropdown(value=saved_annotation.get("is_intent_preserved"), interactive=reviewable),
        gr.Dropdown(value=saved_annotation.get("is_grounded"), interactive=reviewable),
        gr.Textbox(
            value="\n".join(saved_annotation.get("unsupported_claims", [])), interactive=reviewable
        ),
        gr.Dropdown(value=saved_annotation.get("is_standalone"), interactive=reviewable),
        gr.Textbox(value=saved_annotation.get("general_comment", ""), interactive=reviewable),
        gr.Button(interactive=reviewable),
    )


def create_application(
    records: list[dict[str, Any]],
    input_path: Path,
    output_path: Path,
    annotations: dict[int, dict[str, Any]],
) -> gr.Blocks:
    """Build the reviewer UI and connect navigation and atomic-save callbacks."""
    start_index = first_unannotated_index(records, annotations)
    initial_values = _view_values(start_index, records, annotations)

    with gr.Blocks(title="Generated Figma Query Evaluation") as application:
        gr.Markdown("# Generated Figma query evaluation")
        current_index = gr.State(start_index)
        screenshot = gr.Image(value=initial_values[0], label="Source screenshot", interactive=False)
        title = gr.Markdown(initial_values[1])
        progress = gr.Markdown(initial_values[2])
        availability = gr.Markdown(initial_values[3])
        with gr.Row():
            text_query = gr.Textbox(value=initial_values[4], label="User query", interactive=False)
            target_element = gr.Textbox(value=initial_values[5], label="Target element", interactive=False)
            intent = gr.Textbox(value=initial_values[6], label="Annotated intent", interactive=False)
        candidate_queries = gr.Markdown(initial_values[7])
        with gr.Accordion("VLM context", open=False):
            user_prompt = gr.Textbox(value=initial_values[8], label="Exact user prompt", lines=7, interactive=False)
            evidence = gr.Textbox(value=initial_values[9], label="Auxiliary visual evidence", lines=7, interactive=False)
            generator = gr.Textbox(value=initial_values[10], label="Generator metadata", lines=4, interactive=False)
            reasoning_summary = gr.Textbox(
                value=initial_values[11],
                label="Reasoning summary",
                lines=4,
                interactive=False,
                visible=bool(initial_values[11]),
            )

        with gr.Row():
            is_target_correct = gr.Dropdown(
                choices=[True, False],
                value=initial_values[12],
                label="is_target_correct",
                info="The queries refer to the UI element the user's query refers to.",
                interactive=is_reviewable(records[start_index]),
            )
            is_intent_preserved = gr.Dropdown(
                choices=[True, False],
                value=initial_values[13],
                label="is_intent_preserved",
                info="The user intent, such as identifying, comparing, or explaining behavior, is reflected in the suggested queries.",
                interactive=is_reviewable(records[start_index]),
            )
        with gr.Row():
            is_grounded = gr.Dropdown(
                choices=GROUNDING_CHOICES,
                value=initial_values[14],
                label="is_grounded",
                info="No query mentions facts or elements that are not completely deducible by the provided context.",
                interactive=is_reviewable(records[start_index]),
            )
            is_standalone = gr.Dropdown(
                choices=[True, False],
                value=initial_values[16],
                label="is_standalone",
                info="The queries are understandable without the surrounding context knowledge.",
                interactive=is_reviewable(records[start_index]),
            )
        unsupported_claims = gr.Textbox(
            label="unsupported_claims",
            value=initial_values[15],
            lines=4,
            info="Write one unsupported claim per line. Required when is_grounded is false or acceptable.",
            interactive=is_reviewable(records[start_index]),
        )
        general_comment = gr.Textbox(
            label="general_comment",
            value=initial_values[17],
            lines=4,
            info="Optional overall comment about the generated query set.",
            interactive=is_reviewable(records[start_index]),
        )
        status = gr.Markdown("Save an annotation before navigating to retain changes.")
        with gr.Row():
            previous_button = gr.Button("Previous")
            save_button = gr.Button(
                "Save", variant="primary", interactive=is_reviewable(records[start_index])
            )
            next_button = gr.Button("Next")

        view_outputs = [
            screenshot, title, progress, availability, text_query, target_element, intent,
            candidate_queries, user_prompt, evidence, generator, reasoning_summary,
            is_target_correct, is_intent_preserved, is_grounded, unsupported_claims,
            is_standalone, general_comment, save_button,
        ]

        def navigate(index: int, direction: int) -> tuple[object, ...]:
            """Move through source records and display saved values when available."""
            next_index = min(max(index + direction, 0), len(records) - 1)
            values = _view_values(next_index, records, annotations)
            return (*values, "Unsaved edits are not retained when navigating.", next_index)

        def save_annotation(
            index: int,
            target_correct: bool | None,
            intent_preserved: bool | None,
            grounded: str | None,
            claims_text: str,
            standalone: bool | None,
            comment: str,
        ) -> tuple[str, str]:
            """Validate and atomically persist one quality annotation."""
            source_index = index + 1
            source_record = records[index]
            if not is_reviewable(source_record):
                return (_view_values(index, records, annotations)[2], "This record is unavailable for annotation.")
            claims = [line.strip() for line in (claims_text or "").splitlines() if line.strip()]
            if (
                target_correct is None
                or intent_preserved is None
                or standalone is None
                or grounded not in GROUNDING_CHOICES
            ):
                return (_view_values(index, records, annotations)[2], "Fill in every quality rating before saving.")
            if grounded in {"false", "acceptable"} and not claims:
                return (
                    _view_values(index, records, annotations)[2],
                    "Add at least one unsupported claim when grounding is false or acceptable.",
                )
            annotations[source_index] = {
                "schema_version": "1.0",
                "annotated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "source_generated_queries_path": str(input_path),
                "source_record_index": source_index,
                "source_record": source_record,
                "is_target_correct": target_correct,
                "is_intent_preserved": intent_preserved,
                "is_grounded": grounded,
                "unsupported_claims": claims,
                "is_standalone": standalone,
                "general_comment": (comment or "").strip(),
            }
            write_annotations(output_path, records, annotations)
            return (
                _view_values(index, records, annotations)[2],
                f"Saved record {source_index} to `{output_path.name}`.",
            )

        previous_button.click(
            lambda index: navigate(index, -1),
            inputs=current_index,
            outputs=[*view_outputs, status, current_index],
        )
        next_button.click(
            lambda index: navigate(index, 1),
            inputs=current_index,
            outputs=[*view_outputs, status, current_index],
        )
        save_button.click(
            save_annotation,
            inputs=[
                current_index, is_target_correct, is_intent_preserved, is_grounded,
                unsupported_claims, is_standalone, general_comment,
            ],
            outputs=[progress, status],
        )

    return application


def main() -> int:
    """Load generated queries and launch the local annotation interface."""
    arguments = parse_arguments()
    try:
        input_path = arguments.input_path.expanduser().resolve()
        output_path = (
            arguments.output_path.expanduser().resolve()
            if arguments.output_path is not None
            else input_path.parent / ANNOTATIONS_FILENAME
        )
        records = load_generated_records(input_path)
        annotations = load_annotations(output_path, len(records))
    except (OSError, ValueError) as error:
        print(f"Unable to start generated-query annotation interface: {error}")
        return 1
    application = create_application(records, input_path, output_path, annotations)
    application.launch(server_name="127.0.0.1", server_port=arguments.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
