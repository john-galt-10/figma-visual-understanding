"""PaddleOCR and EasyOCR adapters implementing the shared OCR interface."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .base import OcrDependencyError, OcrEngine, OcrError
from .models import TextDetection
from .serialization import to_json_safe, to_polygon


class EasyOcrEngine(OcrEngine):
    """Run EasyOCR and convert its detail output into shared OCR models."""

    engine_name = "easy"

    def _package_name(self) -> str:
        """Return EasyOCR's Python distribution name."""
        return "easyocr"

    def _scan_image(self, image_path: Path) -> tuple[list[TextDetection], dict[str, Any]]:
        """Read all EasyOCR regions from the supplied screenshot."""
        try:
            import easyocr
        except ImportError as error:
            raise OcrDependencyError(
                "EasyOCR is not installed. Install dependencies with: pip install -r requirements.txt"
            ) from error

        reader = easyocr.Reader([self.language])
        raw_detections = reader.readtext(str(image_path), detail=1, paragraph=False)
        detections = [
            TextDetection(
                text=str(text),
                confidence=float(confidence),
                polygon=to_polygon(box),
                metadata={"provider": "easyocr", "raw_detection": to_json_safe(raw)},
            )
            for raw in raw_detections
            for box, text, confidence in [raw]
        ]
        return detections, {"provider": "easyocr", "raw_detection_count": len(raw_detections)}


class PaddleOcrEngine(OcrEngine):
    """Run PaddleOCR, supporting its current and legacy Python response shapes."""

    engine_name = "paddle"

    def _package_name(self) -> str:
        """Return PaddleOCR's Python distribution name."""
        return "paddleocr"

    def _scan_image(self, image_path: Path) -> tuple[list[TextDetection], dict[str, Any]]:
        """Read all PaddleOCR regions and normalize supported response formats."""
        try:
            from paddleocr import PaddleOCR
        except ImportError as error:
            raise OcrDependencyError(
                "PaddleOCR is not installed. Install dependencies with: pip install -r requirements.txt"
            ) from error

        try:
            # PaddlePaddle 3.3.x has a CPU oneDNN/PIR regression. Disabling
            # MKLDNN selects the regular Paddle inference path instead.
            engine = PaddleOCR(lang=self.language, enable_mkldnn=False)
            if hasattr(engine, "predict"):
                raw_results = list(engine.predict(str(image_path)))
            else:
                raw_results = engine.ocr(str(image_path), cls=True)
        except Exception as error:
            raise OcrError(f"PaddleOCR failed to scan '{image_path}': {error}") from error

        detections = self._normalize_results(raw_results)
        return detections, {
            "provider": "paddleocr",
            "raw_result_count": len(raw_results),
        }

    def _normalize_results(self, raw_results: list[Any]) -> list[TextDetection]:
        """Extract text evidence from PaddleOCR v2 and v3 result structures."""
        detections: list[TextDetection] = []
        for result in raw_results:
            payload = self._result_payload(result)
            if isinstance(payload, dict) and "rec_texts" in payload:
                texts = payload.get("rec_texts", [])
                scores = payload.get("rec_scores", [])
                boxes = payload.get("rec_polys", payload.get("dt_polys", []))
                for index, text in enumerate(texts):
                    score = scores[index] if index < len(scores) else None
                    box = boxes[index] if index < len(boxes) else []
                    detections.append(
                        TextDetection(
                            text=str(text),
                            confidence=float(score) if score is not None else None,
                            polygon=to_polygon(box),
                            metadata={
                                "provider": "paddleocr",
                                "result_index": index,
                                "raw_result": to_json_safe(payload),
                            },
                        )
                    )
                continue

            # PaddleOCR 2.x returns one list per image: [box, (text, score)].
            entries = result if isinstance(result, list) else []
            for entry in entries:
                if not isinstance(entry, (list, tuple)) or len(entry) < 2:
                    continue
                box, recognition = entry[0], entry[1]
                if not isinstance(recognition, (list, tuple)) or len(recognition) < 2:
                    continue
                detections.append(
                    TextDetection(
                        text=str(recognition[0]),
                        confidence=float(recognition[1]),
                        polygon=to_polygon(box),
                        metadata={"provider": "paddleocr", "raw_detection": to_json_safe(entry)},
                    )
                )
        return detections

    @staticmethod
    def _result_payload(result: Any) -> Any:
        """Turn PaddleOCR result wrappers into a mapping when their API permits it."""
        if isinstance(result, dict):
            return result
        if hasattr(result, "json"):
            try:
                json_value = result.json
                json_value = json_value() if callable(json_value) else json_value
                return json.loads(json_value) if isinstance(json_value, str) else json_value
            except Exception:
                pass
        if hasattr(result, "to_dict"):
            try:
                return result.to_dict()
            except Exception:
                pass
        return result
