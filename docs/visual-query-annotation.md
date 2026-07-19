# Visual-query annotation interface

`annotation/annotate_visual_query_ui.py` launches a local Gradio app for creating
one visual-query annotation per screenshot in `eval_screenshots/`. Screenshots are
shown in filename order, so the current `figma_eval_001.png` through
`figma_eval_023.png` naming provides stable annotation IDs.

## Run

Install the project requirements in the `figma-navigator` environment, then run:

```powershell
conda run -n figma-navigator python annotation/annotate_visual_query_ui.py
```

Open the local address printed by Gradio. Use `--port 7860` to request a specific
local port. `--screenshot-dir` and `--output-path` can override the default image
directory and annotation dataset path.

Each screenshot requires a `text_query`, `target_element_name`, and one intent:
`identify_and_define`, `how_to`, `explain_behavior`, `troubleshoot`, or `compare`.
The Previous and Next buttons discard unsaved edits; select Save before moving away
from a screenshot.

## Dataset and resume behavior

The interface writes `annotation/visual_query_annotations.jsonl`. Each line is one
JSON record:

```json
{
  "screenshot_id": "figma_eval_001",
  "image_path": "eval_screenshots/figma_eval_001.png",
  "text_query": "What does this toolbar control do?",
  "target_element_name": "Move tool",
  "intent": "identify_and_define"
}
```

When it starts, the app loads this file and opens the first screenshot without an
annotation. Saved fields are prefilled when revisiting an image. Saving an existing
ID replaces that record, and the JSONL file is rewritten in screenshot order so it
contains no duplicate screenshot IDs.
