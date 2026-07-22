# Query-generation evaluation artifacts

`scripts/generate_query_generation_evaluation_set.py` runs every visual-query
annotation through the same `VisualSignalPipeline` used by
`scripts/run_pipeline_example.py`. It creates an inspectable batch artifact for
evaluating generated documentation-retrieval queries against the annotated target
element and intent.

## Run

Use the project interpreter:

```powershell
C:\Users\samue\miniconda3\envs\figma-navigator\python.exe scripts/generate_query_generation_evaluation_set.py
```

The defaults read `annotation/visual_query_annotations.jsonl` and create:

```text
eval/results/query-generation/visual_query_annotations_<YYYYMMDD_HHMMSS>/
  config.yaml
  generated_queries.jsonl
  query_annotations.jsonl  # created by the human-review UI
```

`config.yaml` is an unchanged snapshot of the configuration selected for the
run. `generated_queries.jsonl` contains one record per input annotation in the
same order. The default folder name references the input filename and appends a
timestamp to seconds.

## Parameters

- `--input-path`: annotation JSONL to process; defaults to
  `annotation/visual_query_annotations.jsonl`.
- `--config-path`: unified pipeline configuration to execute and snapshot;
  defaults to `pipeline_config.yaml`.
- `--output-dir`: optional explicit destination folder. It must not already
  exist, and the script creates `generated_queries.jsonl` and `config.yaml`
  inside it.
- `--save-icon-crops` or `--keep-crops`: additionally retain inspectable icon
  crops in per-annotation `artifacts/<index>_<screenshot-id>/icon_crops/`
  directories inside the result folder.

For example:

```powershell
C:\Users\samue\miniconda3\envs\figma-navigator\python.exe scripts/generate_query_generation_evaluation_set.py `
  --input-path annotation/visual_query_annotations.jsonl `
  --config-path pipeline_config.yaml `
  --output-dir eval/results/query-generation/manual_run
```

## Record contents and failures

Each JSONL record includes the complete source annotation, original and resolved
image paths, input/config provenance, the normalized effective configuration,
and either the full serialized pipeline result or a structured pipeline error.
Successful pipeline results retain OCR and icon signals, exact VLM prompt and
evidence, provider metadata, and generated retrieval queries. The only external
asset is the screenshot itself, which remains path-referenced rather than being
embedded in JSONL.

If one screenshot cannot be processed, the script writes a `failed` record for
that annotation and continues with later records. It returns a nonzero status
after completing the artifact when any records failed. The output folder is
created atomically only after both required files have been written. Segmented
and hybrid input modes also retain their per-annotation overlays under
`artifacts/`, preventing artifacts from one screenshot overwriting another.

## Human query review

`annotation/annotate_generated_queries_ui.py` launches a local Gradio reviewer
for the candidate queries in one `generated_queries.jsonl` file:

```powershell
C:\Users\samue\miniconda3\envs\figma-navigator\python.exe annotation/annotate_generated_queries_ui.py `
  --input-path eval/results/query-generation/<run-folder>/generated_queries.jsonl
```

`--input-path` selects the generated-query artifact to review. `--output-path`
optionally selects the annotation JSONL; otherwise the UI writes the sibling
`query_annotations.jsonl`. Use `--port` to request a specific local port.

Each screen shows the source screenshot, user query, annotated target element
and intent, generated candidate queries, and a collapsible view of the exact VLM
prompt, visual evidence, generator metadata, and any supplied reasoning summary.
Hover the information icon on
each annotation field to read its criterion:

- `is_target_correct`: whether the queries refer to the UI element in the user query.
- `is_intent_preserved`: whether the suggested queries preserve the user intent.
- `is_grounded`: `true`, `false`, or `acceptable`, based on whether the queries
  introduce claims not deducible from the supplied context.
- `unsupported_claims`: one unsupported claim per line; required when grounding
  is `false` or `acceptable`.
- `is_standalone`: whether the suggested queries are understandable without the
  surrounding context.
- `general_comment`: optional overall reviewer comment about the generated query set.

Failed pipeline records and successful records with no generated queries remain
visible but are unavailable for annotation. Every saved reviewer record embeds
the complete generated-query source record, its source path and one-based source
record index, plus the ratings. The UI rewrites annotations atomically in source
order, resumes existing work, and pre-fills previously saved ratings.
