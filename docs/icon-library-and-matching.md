# Icon Library and Matching

This experiment turns manually collected Figma icon screenshots into normalized templates, then compares a single icon candidate against that library. Candidate-region generation remains a separate step documented in [icon-candidate-generation.md](icon-candidate-generation.md).

## Shared preprocessing

Both library construction and query matching use `icon_matching.template_preprocessing` in [icon_matching.yaml](../icon_matching.yaml). The input screenshot's dimensions and aspect ratio are ignored. The shared normalizer assumes a light glyph on a dark background, uses Otsu thresholding to create a binary mask, derives a soft grayscale mask, tightly crops the glyph, and centers it without distortion on the configured square canvas.

Keeping this logic shared is important: a matcher compares like-for-like masks rather than two subtly different crop or resize procedures.

## Build the library

Source screenshots live in `assets/icon_library/raw_screenshots`. Nested folders are supported; every filename stem becomes a label and labels must be unique across those folders.

```powershell
conda run -n figma-navigator python scripts/build_icon_library.py
```

`--input-dir` selects manually collected source images. `--output-dir` selects the generated library directory. `--config-path` selects the YAML file containing the shared canvas configuration.

The default output directory, `assets/icon_library/templates`, contains:

- `soft/crops/` and `binary/crops/`: tightly cropped glyph masks.
- `soft/canvases/` and `binary/canvases/`: standardized matching canvases.
- `templates.json`: template labels, crop metadata, preprocessing threshold, and artifact paths.

## Match one icon

Use the example script with a PNG containing one icon candidate crop. It does not identify candidate regions itself; run candidate generation first when starting with a larger screenshot.

```powershell
conda run -n figma-navigator python scripts/match_icon_example.py `
  --image-path path/to/icon-candidate.png `
  --output-path outputs/icon-match.json
```

`--config-path` chooses the shared settings. `--top-k` overrides the configured number of displayed matches. `--output-path` is optional and writes the complete preprocessing metadata plus ranked matches as JSON; the ranked table is always printed to the terminal.

## Matching strategy

The configured primary provider is `chamfer`. It converts binary masks to edge maps and uses symmetric distance-transform Chamfer similarity, checking small translations to tolerate imperfect candidate alignment. Scores are normalized from `0` through `1`, where higher is more similar.

The optional `soft_ncc` tie breaker reranks only the configured leading candidate pool using normalized cross-correlation on the soft grayscale canvases. Set `matching.tie_breaker.enabled: false` to return the pure primary-Chamfer order and omit tie-breaker scores.

All providers implement the same matcher contract, so a future CNN can consume normalized canvases and return the same ranked-score model without changing the CLI or the surrounding pipeline.
