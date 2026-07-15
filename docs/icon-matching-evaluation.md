# Icon matching evaluation

This workflow creates a flat set of icon-candidate crops from screenshots, labels those crops manually, and evaluates the configured template matcher.

## Create a candidate evaluation set

`scripts/generate_icon_candidate_evaluation_set.py` converts a `.lst` file of screenshot paths into a flat collection of candidate-crop PNGs. It uses the same `IconCandidateGenerator` and `icon_matching.yaml` settings as `generate_icon_candidates.py`.

Each output folder also includes `crop_provenance.jsonl`. Every line identifies one crop PNG, its source screenshot, original candidate ID, bounding boxes, and proposal metadata.

### Input list format

Use one screenshot path per line. Blank lines and lines beginning with `#` are ignored. Relative paths are resolved from the `.lst` file's parent directory.

```text
# toolbar examples
../../screenshots_example/aaaa0000.png
../../screenshots_example/bbbbb.png
```

### Run

```powershell
conda run -n figma-navigator python scripts/generate_icon_candidate_evaluation_set.py `
  --input-list-path path/to/screenshots.lst
```

Optional parameters:

* `--config-path`: shared candidate-generation configuration; defaults to `icon_matching.yaml`.
* `--output-dir`: destination for the flat crop PNGs and JSONL manifest; defaults to `outputs/icon-candidates/evaluation_candidates`.
* `--clear-output`: removes prior output-folder contents before generating a replacement set. Without it, the script refuses a non-empty destination so the manifest always covers every crop in the folder.

Crop names use `<source-number>_<screenshot-stem>_<candidate-id>.png`, avoiding collisions when multiple screenshots emit the same candidate IDs.

## Label crops

Copy or rename `crop_provenance.jsonl` to the evaluation manifest, then populate every record's `label`. A non-empty label must exactly match a configured library template. An empty label means that the crop is not an icon and should be rejected.

By default, the evaluator reads `outputs/icon-candidates/evaluation_candidates/crop_labeling.jsonl`. Crop paths can be absolute or relative to the manifest.

## Evaluate matching

```powershell
conda run -n figma-navigator python scripts/evaluate_icon_matching.py `
  --detection-threshold 0.9 `
  --threshold-score final `
  --json-output-path outputs/icon-candidates/evaluation_candidates/matching-evaluation.json
```

`--manifest-path` chooses a different labeling JSONL, `--config-path` chooses matcher settings, and `--json-output-path` is optional. Without it, results are printed only to the terminal.

`--threshold-score` selects the top result score used only for thresholding: `final` (the configured reranked score), `primary` (Chamfer), or `secondary` (the tie-breaker score). It does not change the matcher ranking or predicted label. Secondary scoring requires an enabled tie breaker. A top score below `--detection-threshold` produces no detection.

Pass `--mine-threshold` to first score the complete manifest, then re-run the evaluation with the smallest representable threshold above the highest score among empty-label crops. This guarantees that every labeled non-icon is rejected. The optional JSON report keeps the same schema and contains only the re-evaluated results.

## Metrics

Precision and recall are end-to-end: a true positive must be a labeled crop that is detected with the correct top-1 label. A wrong emitted label counts as both a false positive and a false negative. Emitting a label for an empty-label crop is a false positive.

MRR is calculated over labeled crops. A threshold-rejected crop contributes zero; otherwise it contributes the reciprocal of the correct label's rank in the full library result list. Detailed JSON reports retain the selected `threshold_score` and also expose the top match's `final_score`, `primary_score`, and `secondary_score` for threshold analysis.

End-to-end accuracy counts all fully correct outcomes, including correctly rejecting non-icons. Detection accuracy only evaluates the icon versus non-icon decision and ignores the emitted label's correctness.
