# Icon Candidate Generation

`src/icon_matching/candidate_generation` proposes likely icon regions from a user-selected Figma screenshot crop. It is a classical computer-vision experiment, not an icon recognizer: its output is a set of inspectable image regions for a later icon-matching module or a VLM to evaluate.

## Pipeline

The pipeline runs enabled proposal methods across the full screenshot crop:

1. **Morphology and connected components** use local contrast plus adaptive thresholding, then lightly dilate foreground pixels so disconnected strokes can become one icon proposal.
2. **Canny contours** find visible edges and group nearby contour boxes into larger proposals.
3. **Geometric filters** first remove regions outside configured size, aspect-ratio, and compactness limits. This stops oversized toolbar-level proposals from affecting smaller icon proposals.
4. **Merging** joins only highly overlapping morphology/contour proposals. It uses intersection over union (IoU) and never merges two proposals from the same detector.
5. The same **geometric filters** run once more as a safety check after merging.
6. **Optional OCR suppression** runs the repository's configured PaddleOCR or EasyOCR engine internally and removes candidates mostly covered by recognized text.
7. **Optional containment deduplication** removes a final candidate fully contained by another final candidate. The larger box is retained and receives the removed proposal's detector evidence.

All stages are configured in [icon_matching.yaml](../icon_matching.yaml). Set any stage's `enabled` field to `false` to compare approaches. OCR suppression is disabled by default because recall is more important than aggressively removing ambiguous candidates at this stage.

Final candidates are ordered before their IDs are assigned. The default `ordering.provider: center_cluster` calculates box centers, groups candidates whose vertical centers fall within `ordering.row_center_tolerance_height_multiplier` (default `0.4`) times the median final-candidate height of the row's first center, then sorts each row left-to-right. This keeps a visually aligned toolbar row together despite small detector-box jitter. Set `ordering.provider: strict_top_left` to restore the legacy deterministic `(y, x, width, height)` ordering.

The `filters.square_shape` block is enabled by default to remove clearly text-like, elongated boxes. Its `minimum_compactness` is the shorter side divided by the longer side: `1.0` accepts only exact squares, while lower values accept progressively more rectangular candidates. Set `filters.square_shape.enabled: false` to retain all shapes during an experiment.

Set `merging.deduplicate_contained_regions: false` to retain fully contained final regions. When enabled (the default), this cleanup runs after geometric filtering and optional OCR suppression, independently of the IoU-based cross-detector merge; equal boxes keep the first proposal deterministically.

## Run the script

Run from the repository root in the `figma-navigator` Python 3.11 environment:

```bash
python scripts/generate_icon_candidates.py \
  --image-path path/to/figma-crop.png \
  --config-path icon_matching.yaml \
  --output-dir outputs/icon-candidates
```

`--image-path` is the cropped Figma screenshot. `--config-path` selects all detector and output settings. `--output-dir` receives the run artifacts. By default, the directory receives `candidates.json`, `overlay.png` when visualization is enabled, and `crops/candidate-###.png` for each final candidate.

Use `--no-crop-files` to skip the `crops/` directory. In that mode the script writes only `candidates.json` and, when enabled in YAML, `overlay.png`. The JSON still contains source-image metadata and `crop_bbox`, allowing a future matcher to crop pixels itself.

## Comparing configurations

To test morphology alone, set `detectors.contours.enabled: false`. To test contours alone, set `detectors.morphology.enabled: false`. Enable both detector blocks and `merging.enabled: true` for the combined experiment. Enable `ocr_suppression.enabled: true` and select `paddle` or `easy` when testing optional text suppression.

The overlay colors indicate provenance: orange is morphology, blue is contours, and green is a candidate supported by both methods. Compare overlays and the `counts` field in JSON before tuning thresholds.

## Matching handoff format

`candidates.json` has schema version `1.0` and contains source image metadata, the effective settings, per-stage counts, optional OCR metadata, and final candidates. Each candidate includes:

```json
{
  "id": "candidate-001",
  "content_bbox": {"x": 12, "y": 24, "width": 16, "height": 16},
  "crop_bbox": {"x": 10, "y": 22, "width": 20, "height": 20},
  "detector_sources": ["morphology", "contours"],
  "detector_evidence": {"morphology": [], "contours": []},
  "proposal_score": 1.0,
  "crop_path": ".../crops/candidate-001.png",
  "crop_dimensions": {"width": 20, "height": 20}
}
```

`content_bbox` is the detector's box; `crop_bbox` includes configured padding and is always available. `crop_path` and `crop_dimensions` are present only when crop-file output is enabled. This stage never assigns an icon identity or a match score.
