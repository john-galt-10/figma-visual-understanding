# VLM image-input modes

The unified visual pipeline can vary how detected icon evidence is spatially grounded for the VLM. Set `candidate_queries.input_mode` in `pipeline_config.yaml` to compare the modes without changing OCR, candidate generation, or matching settings.

## Modes

- `vanilla` sends the original screenshot and lists accepted icon names in configured candidate order. This is the default and preserves the original pipeline behavior.
- `segmented` sends only `annotated_screenshot.png`. Every threshold-accepted icon is boxed and labeled with a contiguous number; the prompt maps each number to its detected icon name.
- `hybrid` sends the original screenshot followed by `annotated_screenshot.png`, with the same numbered prompt mapping as `segmented`.

Only matcher detections at or above `pipeline.icon_matching.matching.detection_threshold` appear in an annotation. Rejected candidates are neither boxed nor named. Numbering follows `pipeline.icon_matching.ordering` after this thresholding step.

## Artifacts and comparison

Segmented and hybrid runs always retain `annotated_screenshot.png` beside `results_details.json`, even if `--save-icon-crops` is omitted. The JSON records `vlm_input.input_mode`, ordered image roles and paths, the annotated-image path, the numbered icon mapping, and the exact prompt evidence.

Use `--no-vlm` to compare the prepared image inputs and prompt evidence without sending an API request. This makes it practical to run identical screenshots across all three modes before evaluating retrieval-query quality.


## Code flow diagram

```
Screenshot path + optional user question
                │
                ▼
VisualSignalPipeline.run(...)
src/visual_pipeline/pipeline.py
                │
                ├── OCR stage
                │   └── Produces visible text for prompt evidence
                │
                ├── Icon candidate + matching stages
                │   └── Produces accepted icon matches after score thresholding
                │
                ▼
create_input_mode_strategy(input_mode)
src/visual_pipeline/input_modes.py
                │
                ├── vanilla
                │   ├── Images: [original screenshot]
                │   └── Prompt icons: "- move", "- frame", ...
                │
                ├── segmented
                │   ├── Draw accepted matches as numbered overlay
                │   │   └── write_detection_overlay(...)
                │   │       src/icon_matching/candidate_generation/output.py
                │   ├── Images: [annotated screenshot]
                │   └── Prompt icons: "- #1: move", "- #2: frame", ...
                │
                └── hybrid
                    ├── Draw the same numbered overlay
                    ├── Images: [original screenshot, annotated screenshot]
                    └── Prompt icons: "- #1: move", "- #2: frame", ...
                │
                ▼
InputModeSelection
                │
                ├── ordered image paths
                ├── mode-specific auxiliary evidence text
                └── inspection metadata for results JSON
                │
                ▼
VisualSignalPipeline._run_vlm(...)
src/visual_pipeline/pipeline.py
                │
                ├── First image  → image_path
                └── Later images → additional_image_paths
                │
                ▼
GeminiCandidateQueryGenerator.generate(...)
src/candidate_queries/gemini.py
                │
                ├── _build_prompt(...)
                │   └── Combines user question + auxiliary evidence text
                │
                ├── Opens each local image with Pillow
                │
                ▼
client.models.generate_content(...)
                │
                └── contents = [prompt, *images]

                         ┌──────────────────────────┐
                         │       Gemini API          │
                         │                          │
                         │  1. prompt text          │
                         │  2. original image       │
                         │  3. annotated image      │
                         │     (hybrid only)         │
                         └──────────────────────────┘
```