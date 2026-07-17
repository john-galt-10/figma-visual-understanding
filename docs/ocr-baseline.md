# OCR Baseline

The reusable `src/ocr` package scans a raster screenshot through a common interface and normalizes the output from PaddleOCR or EasyOCR. It preserves image metadata, recognized text, confidence, pixel-coordinate polygons, and JSON-safe engine-specific detection metadata. This makes the OCR evidence inspectable before it is provided to a VLM or a later parsing stage.

Supported inputs are PNG, JPEG, WebP, BMP, and TIFF images. The initial default recognition language is English, which matches the expected Figma UI screenshots.

## Installation

Install the experiment dependencies in the `figma-navigator` Python 3.11 environment:

```bash
pip install -r requirements.txt
```

## Run an OCR scan

Run a scan by providing the screenshot path and OCR engine:

```bash
python scripts/run_ocr_example.py --image-path path/to/figma-screenshot.png --engine paddle
python scripts/run_ocr_example.py --image-path path/to/figma-screenshot.png --engine easy
```

Pass `--detection-threshold` with a value from `0.0` through `1.0` to print only non-empty detections meeting that confidence. Omitting the option preserves the raw normalized provider output:

```powershell
python scripts/run_ocr_example.py --image-path path/to/figma-screenshot.png --engine easy --detection-threshold 0.90
```

When filtering is enabled, detections without confidence scores are excluded. The output metadata records the applied threshold and the original detection count.

Pass `--output-path` to write the same final JSON that is printed to standard output. Parent directories are created when needed:

```powershell
python scripts/run_ocr_example.py --image-path path/to/figma-screenshot.png --engine easy --output-path outputs/ocr-scan.json
```

`--image-path` is the source raster screenshot and `--engine` selects either `paddle` or `easy`. The script prints an indented JSON scan result to standard output; errors are printed to standard error. Its output contains shared image and engine metadata, one structured object per detected text region, complete visible text, and timing information.

The PaddleOCR adapter disables oneDNN/MKLDNN acceleration by default because PaddlePaddle 3.3.x has a CPU inference regression. This prioritizes reliable baseline scans over that optional acceleration.
