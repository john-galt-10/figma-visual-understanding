# Unified Visual-Signal Pipeline

`scripts/run_pipeline_example.py` combines the standalone OCR, icon-candidate/matching, and candidate-query experiments. It produces coordinate-free evidence for the VLM while keeping the complete run inspectable in one JSON artifact.

## Run it

Use the project interpreter:

```powershell
C:\Users\samue\miniconda3\envs\figma-navigator\python.exe scripts/run_pipeline_example.py `
  --image-path screenshots_example/example.png `
  --text-query "What does this control do?"
```

`--image-path` is required. `--text-query` is optional. `--config-path` selects the unified YAML file, and `--output-path` optionally chooses the JSON artifact. Without `--output-path`, each run receives a timestamped folder at `outputs/pipeline/<image-stem>_<YYYYMMDD_HHMMSS>/`, containing `results_details.json`. Pass `--no-vlm` to skip generation for one run regardless of the YAML setting, which is useful for inspecting signals without spending tokens.

Use `--save-icon-crops` or `--keep-crops` to retain matcher input crops in `icon_crops/` inside the same run folder as the JSON artifact. Without either flag, the pipeline creates the crops in a temporary directory and removes them after matching.

## Configuration

`pipeline_config.yaml` contains all component settings. The switches are:

- `pipeline.ocr.enabled`: include OCR text evidence.
- `pipeline.ocr.detection_threshold`: minimum OCR confidence for VLM text evidence; defaults to `0.90`.
- `pipeline.icon_matching.enabled`: include matched icon-name evidence.
- `candidate_queries.enabled`: call the configured VLM provider to generate retrieval queries.
- `candidate_queries.input_mode`: choose `vanilla` (original screenshot), `segmented` (numbered accepted-icon overlay), or `hybrid` (original followed by overlay). It defaults to `vanilla`.

The VLM setting defaults to `true`. Set it to `false` to inspect OCR/icon evidence without initializing a provider client or spending API tokens. The script prints the exact normalized signal block in either mode.

Enabled OCR or icon stages fail the run if their dependencies, inputs, or configured backend fail. Disabled stages are recorded as disabled and do not prevent VLM generation.

OCR evidence includes only non-empty detections whose confidence is at least `pipeline.ocr.detection_threshold`. Detections without a confidence score are excluded. The JSON signal records the threshold plus total, accepted, and rejected detection counts.

Icon matching accepts a candidate only when its configured top-match score is at least `matching.detection_threshold`. Set `matching.detection_score` to `final`, `primary`, or `secondary` to choose the score: `secondary` is the configured tie-breaker score and requires an available tie breaker. Rejected candidates do not appear in VLM icon evidence.

## VLM evidence and JSON output

The VLM receives the screenshot, optional user question, and this compact additional prompt text:

```text
OCR-visible text:
- Design
- Auto layout

Detected icons (top-left to bottom-right):
- move
- pen
```

OCR detections remain sorted by top-left position. Icon candidates use `pipeline.icon_matching.ordering`: by default `center_cluster` groups visually aligned boxes into rows using `0.4` times the median final-candidate height as its center tolerance, then orders every row left-to-right. Set `provider: strict_top_left` for the legacy `(y, x, width, height)` order. Candidate IDs and the accepted icon-name list both use this selected icon-candidate order; no coordinates are included in the prompt or pipeline JSON signal format.

The JSON contains source input, normalized signals, the exact evidence block, VLM output and optional reasoning summary, plus `icon_candidate_to_detected_name`. The icon signal records the configured threshold and score metric, alongside accepted and rejected counts. That candidate mapping is deliberately `{}` unless `--save-icon-crops` is supplied. With crop retention, every candidate crop remains inspectable with its acceptance status, selected threshold score, all matcher scores, and detected name (or `null` when rejected).

`segmented` and `hybrid` additionally write `annotated_screenshot.png` beside the JSON. The overlay contains only accepted icon matches, numbered contiguously in the configured ordering. Their VLM evidence maps each number to its detected name; `hybrid` supplies the original screenshot first and the annotation second. The `vlm_input` JSON records the selected mode, ordered image roles, annotation path, mapping, a separate `input_description`, and the separate `system_prompt` and `user_prompt` fields. Images remain separate API content parts. See [VLM input modes](vlm-input-modes.md) for the comparison workflow.
