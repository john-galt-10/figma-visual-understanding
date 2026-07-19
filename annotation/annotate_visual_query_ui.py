"""Launch a local Gradio interface for annotating Figma evaluation screenshots."""

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gradio as gr


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCREENSHOT_DIRECTORY = REPOSITORY_ROOT / "eval_screenshots"
DEFAULT_OUTPUT_PATH = REPOSITORY_ROOT / "annotation" / "visual_query_annotations.jsonl"
SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}
INTENT_CHOICES = [
    "identify_and_define",
    "how_to",
    "explain_behavior",
    "troubleshoot",
    "compare",
]


@dataclass(frozen=True)
class Screenshot:
    """Describe one screenshot that can receive a visual-query annotation."""

    screenshot_id: str
    path: Path
    relative_path: str


def parse_arguments() -> argparse.Namespace:
    """Parse local-server and annotation-path command-line options."""
    parser = argparse.ArgumentParser(
        description="Annotate Figma evaluation screenshots in a local Gradio interface."
    )
    parser.add_argument(
        "--screenshot-dir",
        type=Path,
        default=DEFAULT_SCREENSHOT_DIRECTORY,
        help="Directory containing evaluation screenshots (default: %(default)s).",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="JSONL file for saved annotations (default: %(default)s).",
    )
    parser.add_argument(
        "--port",
        type=int,
        help="Optional local Gradio server port.",
    )
    return parser.parse_args()


def discover_screenshots(screenshot_directory: Path) -> list[Screenshot]:
    """Return supported screenshots in filename order with repository-relative paths."""
    if not screenshot_directory.is_dir():
        raise ValueError(f"Screenshot directory does not exist: {screenshot_directory}")

    screenshots: list[Screenshot] = []
    for path in sorted(screenshot_directory.iterdir(), key=lambda item: item.name.casefold()):
        if not path.is_file() or path.suffix.casefold() not in SUPPORTED_IMAGE_EXTENSIONS:
            continue
        try:
            relative_path = path.resolve().relative_to(REPOSITORY_ROOT.resolve()).as_posix()
        except ValueError as error:
            raise ValueError(
                "Screenshots must be located inside the repository so their saved "
                "image paths are repository-relative."
            ) from error
        screenshots.append(
            Screenshot(
                screenshot_id=path.stem,
                path=path.resolve(),
                relative_path=relative_path,
            )
        )

    if not screenshots:
        raise ValueError(f"No supported images found in: {screenshot_directory}")
    return screenshots


def load_annotations(output_path: Path, known_ids: set[str]) -> dict[str, dict[str, str]]:
    """Load valid saved annotations keyed by screenshot ID, keeping the latest duplicate."""
    if not output_path.exists():
        return {}

    annotations: dict[str, dict[str, str]] = {}
    with output_path.open("r", encoding="utf-8") as file_handle:
        for line_number, line in enumerate(file_handle, start=1):
            if not line.strip():
                continue
            try:
                record: Any = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {output_path} at line {line_number}."
                ) from error
            if not isinstance(record, dict):
                raise ValueError(f"Expected an object in {output_path} at line {line_number}.")

            required_keys = {
                "screenshot_id",
                "image_path",
                "text_query",
                "target_element_name",
                "intent",
            }
            if not required_keys.issubset(record) or not all(
                isinstance(record[key], str) for key in required_keys
            ):
                raise ValueError(
                    f"Invalid annotation schema in {output_path} at line {line_number}."
                )
            if record["screenshot_id"] in known_ids:
                annotations[record["screenshot_id"]] = {
                    key: record[key] for key in required_keys
                }
    return annotations


def write_annotations(
    output_path: Path,
    screenshots: list[Screenshot],
    annotations: dict[str, dict[str, str]],
) -> None:
    """Atomically rewrite annotations in screenshot order without duplicate IDs."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ordered_records = [
        annotations[screenshot.screenshot_id]
        for screenshot in screenshots
        if screenshot.screenshot_id in annotations
    ]
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        delete=False,
    ) as file_handle:
        for record in ordered_records:
            file_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        temporary_path = Path(file_handle.name)
    temporary_path.replace(output_path)


def initial_index(screenshots: list[Screenshot], annotations: dict[str, dict[str, str]]) -> int:
    """Return the first unannotated screenshot index, or zero when all are complete."""
    return next(
        (
            index
            for index, screenshot in enumerate(screenshots)
            if screenshot.screenshot_id not in annotations
        ),
        0,
    )


def view_values(
    index: int, screenshots: list[Screenshot], annotations: dict[str, dict[str, str]]
) -> tuple[str, str, str, str, str, str]:
    """Create component values for one screenshot, including any saved annotation."""
    screenshot = screenshots[index]
    annotation = annotations.get(screenshot.screenshot_id, {})
    progress = f"**Progress:** {len(annotations)} / {len(screenshots)} saved"
    title = f"### Screenshot `{screenshot.screenshot_id}`"
    return (
        str(screenshot.path),
        title,
        progress,
        annotation.get("text_query", ""),
        annotation.get("target_element_name", ""),
        annotation.get("intent", None),
    )


def create_application(
    screenshots: list[Screenshot], output_path: Path, annotations: dict[str, dict[str, str]]
) -> gr.Blocks:
    """Build the single-user annotation interface and connect its callbacks."""
    start_index = initial_index(screenshots, annotations)
    initial_values = view_values(start_index, screenshots, annotations)

    with gr.Blocks(title="Figma Visual Query Annotation") as application:
        gr.Markdown("# Figma visual-query annotation")
        current_index = gr.State(start_index)
        screenshot_image = gr.Image(
            value=initial_values[0], label="Screenshot", interactive=False
        )
        screenshot_title = gr.Markdown(initial_values[1])
        progress = gr.Markdown(initial_values[2])
        text_query = gr.Textbox(value=initial_values[3], label="text_query", lines=3)
        target_element_name = gr.Textbox(
            value=initial_values[4], label="target_element_name"
        )
        intent = gr.Dropdown(
            choices=INTENT_CHOICES,
            value=initial_values[5],
            label="intent",
        )
        status = gr.Markdown("Save an annotation before navigating to retain changes.")

        with gr.Row():
            previous_button = gr.Button("Previous")
            save_button = gr.Button("Save", variant="primary")
            next_button = gr.Button("Next")

        navigation_outputs = [
            screenshot_image,
            screenshot_title,
            progress,
            text_query,
            target_element_name,
            intent,
            status,
            current_index,
        ]

        def navigate(index: int, direction: int) -> tuple[object, ...]:
            """Move to an adjacent screenshot and load its saved values, if any."""
            next_index = min(max(index + direction, 0), len(screenshots) - 1)
            values = view_values(next_index, screenshots, annotations)
            return (*values, "Unsaved edits are not retained when navigating.", next_index)

        def save_annotation(
            index: int, query: str, element_name: str, selected_intent: str | None
        ) -> tuple[str, str]:
            """Validate and persist the current screenshot annotation."""
            query = (query or "").strip()
            element_name = (element_name or "").strip()
            if not query or not element_name or selected_intent not in INTENT_CHOICES:
                return (
                    f"**Progress:** {len(annotations)} / {len(screenshots)} saved",
                    "⚠️ Fill in `text_query`, `target_element_name`, and `intent` before saving.",
                )

            screenshot = screenshots[index]
            annotations[screenshot.screenshot_id] = {
                "screenshot_id": screenshot.screenshot_id,
                "image_path": screenshot.relative_path,
                "text_query": query,
                "target_element_name": element_name,
                "intent": selected_intent,
            }
            write_annotations(output_path, screenshots, annotations)
            return (
                f"**Progress:** {len(annotations)} / {len(screenshots)} saved",
                f"✅ Saved `{screenshot.screenshot_id}` to `{output_path.name}`.",
            )

        previous_button.click(
            lambda index: navigate(index, -1),
            inputs=current_index,
            outputs=navigation_outputs,
        )
        next_button.click(
            lambda index: navigate(index, 1),
            inputs=current_index,
            outputs=navigation_outputs,
        )
        save_button.click(
            save_annotation,
            inputs=[current_index, text_query, target_element_name, intent],
            outputs=[progress, status],
        )

    return application


def main() -> int:
    """Load screenshots and saved records, then start the local Gradio server."""
    arguments = parse_arguments()
    try:
        screenshots = discover_screenshots(arguments.screenshot_dir.resolve())
        annotations = load_annotations(
            arguments.output_path.resolve(), {item.screenshot_id for item in screenshots}
        )
    except ValueError as error:
        print(f"Unable to start annotation interface: {error}")
        return 1

    application = create_application(screenshots, arguments.output_path.resolve(), annotations)
    application.launch(server_name="127.0.0.1", server_port=arguments.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
