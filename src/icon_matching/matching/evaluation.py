"""Evaluation of ranked icon matches against manually labeled candidate crops."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from .base import IconMatchingEvaluationError
from .pipeline import IconLibraryMatcher


DetectionScoreName = Literal["final", "primary", "secondary"]


@dataclass(frozen=True)
class CropEvaluation:
    """Record the thresholded matching outcome for one labeled crop."""

    crop_path: str
    expected_label: str
    predicted_label: str | None
    final_score: float
    primary_score: float
    secondary_score: float | None
    threshold_score: float
    detected: bool
    true_label_rank: int | None
    reciprocal_rank: float
    outcome: str


@dataclass(frozen=True)
class IconMatchingEvaluationResult:
    """Contain aggregate metrics and inspectable outcomes for one evaluation run."""

    manifest_path: str
    detection_threshold: float
    detection_score_name: DetectionScoreName
    total_crops: int
    labeled_crops: int
    unlabeled_crops: int
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int
    precision: float
    recall: float
    mean_reciprocal_rank: float
    end_to_end_accuracy: float
    detection_accuracy: float
    crops: list[CropEvaluation]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready representation of metrics and per-crop outcomes."""
        payload = asdict(self)
        for key in ("precision", "recall", "mean_reciprocal_rank", "end_to_end_accuracy", "detection_accuracy"):
            payload[key] = round(float(payload[key]), 6)
        return payload


class IconMatchingEvaluator:
    """Evaluate thresholded icon matches using a JSONL manifest with manual labels."""

    def __init__(self, matcher: IconLibraryMatcher) -> None:
        """Store the configured matcher whose library defines valid non-empty labels."""
        self.matcher = matcher

    def evaluate(
        self,
        manifest_path: str | Path,
        detection_threshold: float,
        detection_score_name: DetectionScoreName,
    ) -> IconMatchingEvaluationResult:
        """Match manifest crops and calculate end-to-end detection and ranking metrics."""
        if not 0.0 <= detection_threshold <= 1.0:
            raise IconMatchingEvaluationError("Detection threshold must be between 0 and 1.")
        manifest_file = Path(manifest_path).expanduser().resolve()
        records = _load_manifest_records(manifest_file)
        _validate_labels(records, {template.label for template in self.matcher.templates})
        if detection_score_name == "secondary" and self.matcher.tie_breaker is None:
            raise IconMatchingEvaluationError(
                "Secondary detection scoring requires an enabled matching.tie_breaker."
            )

        evaluations: list[CropEvaluation] = []
        true_positives = false_positives = false_negatives = true_negatives = 0
        correct_detection_count = 0
        for record in records:
            expected_label = record["label"]
            crop_path = _resolve_crop_path(manifest_file, record["crop_path"])
            if not crop_path.is_file():
                raise IconMatchingEvaluationError(f"Crop file does not exist: {crop_path}")
            match_run = self.matcher.match(crop_path, top_k=len(self.matcher.templates))
            top_match = match_run.results[0]
            detection_score = _select_detection_score(top_match, detection_score_name)
            detected = detection_score >= detection_threshold
            predicted_label = top_match.label if detected else None
            true_label_rank = _find_label_rank(match_run.results, expected_label) if expected_label else None
            reciprocal_rank = 0.0
            if bool(expected_label) == detected:
                correct_detection_count += 1

            if not expected_label:
                if detected:
                    false_positives += 1
                    outcome = "false_positive_non_icon"
                else:
                    true_negatives += 1
                    outcome = "true_negative"
            elif not detected:
                false_negatives += 1
                outcome = "false_negative_rejected"
            elif predicted_label == expected_label:
                true_positives += 1
                reciprocal_rank = 1.0 / true_label_rank if true_label_rank is not None else 0.0
                outcome = "true_positive"
            else:
                false_positives += 1
                false_negatives += 1
                reciprocal_rank = 1.0 / true_label_rank if true_label_rank is not None else 0.0
                outcome = "misclassified"

            evaluations.append(
                CropEvaluation(
                    crop_path=str(crop_path),
                    expected_label=expected_label,
                    predicted_label=predicted_label,
                    final_score=top_match.final_score,
                    primary_score=top_match.primary_score,
                    secondary_score=top_match.tie_breaker_score,
                    threshold_score=detection_score,
                    detected=detected,
                    true_label_rank=true_label_rank,
                    reciprocal_rank=reciprocal_rank,
                    outcome=outcome,
                )
            )

        labeled_crops = sum(bool(record["label"]) for record in records)
        total_crops = len(records)
        return IconMatchingEvaluationResult(
            manifest_path=str(manifest_file),
            detection_threshold=detection_threshold,
            detection_score_name=detection_score_name,
            total_crops=total_crops,
            labeled_crops=labeled_crops,
            unlabeled_crops=total_crops - labeled_crops,
            true_positives=true_positives,
            false_positives=false_positives,
            false_negatives=false_negatives,
            true_negatives=true_negatives,
            precision=_safe_divide(true_positives, true_positives + false_positives),
            recall=_safe_divide(true_positives, true_positives + false_negatives),
            mean_reciprocal_rank=_safe_divide(
                sum(evaluation.reciprocal_rank for evaluation in evaluations), labeled_crops
            ),
            end_to_end_accuracy=_safe_divide(true_positives + true_negatives, total_crops),
            detection_accuracy=_safe_divide(correct_detection_count, total_crops),
            crops=evaluations,
        )


def mine_detection_threshold(result: IconMatchingEvaluationResult) -> float:
    """Return the smallest representable threshold that rejects every labeled non-icon crop."""
    non_icon_scores = [crop.threshold_score for crop in result.crops if not crop.expected_label]
    if not non_icon_scores:
        raise IconMatchingEvaluationError(
            "Cannot mine a threshold because the evaluation manifest has no non-icon crops."
        )
    threshold = math.nextafter(max(non_icon_scores), math.inf)
    if threshold > 1.0:
        raise IconMatchingEvaluationError(
            "Cannot mine a threshold in the supported 0 through 1 range because a non-icon scored 1.0."
        )
    return threshold


def _load_manifest_records(manifest_path: Path) -> list[dict[str, str]]:
    """Load required crop and label fields from non-empty JSONL manifest lines."""
    if not manifest_path.is_file():
        raise IconMatchingEvaluationError(f"Evaluation manifest does not exist: {manifest_path}")
    records: list[dict[str, str]] = []
    for line_number, line in enumerate(manifest_path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            crop_path = payload["crop_path"]
            label = payload["label"]
        except (json.JSONDecodeError, KeyError, TypeError) as error:
            raise IconMatchingEvaluationError(
                f"Invalid JSONL record on line {line_number} of '{manifest_path}': {error}"
            ) from error
        if not isinstance(crop_path, str) or not crop_path:
            raise IconMatchingEvaluationError(f"Manifest line {line_number} has no usable 'crop_path'.")
        if not isinstance(label, str):
            raise IconMatchingEvaluationError(f"Manifest line {line_number} has a non-string 'label'.")
        records.append({"crop_path": crop_path, "label": label})
    if not records:
        raise IconMatchingEvaluationError(f"Evaluation manifest contains no crop records: {manifest_path}")
    return records


def _validate_labels(records: list[dict[str, str]], library_labels: set[str]) -> None:
    """Reject manual labels that cannot be evaluated against the active template library."""
    unknown_labels = sorted({record["label"] for record in records if record["label"] and record["label"] not in library_labels})
    if unknown_labels:
        raise IconMatchingEvaluationError(
            "Manifest contains labels absent from the configured template library: " + ", ".join(unknown_labels)
        )


def _resolve_crop_path(manifest_path: Path, crop_path: str) -> Path:
    """Resolve an absolute crop path or a manifest-relative crop filename."""
    path = Path(crop_path).expanduser()
    return path.resolve() if path.is_absolute() else (manifest_path.parent / path).resolve()


def _select_detection_score(match, score_name: DetectionScoreName) -> float:
    """Return the configured top-match score or reject unavailable secondary scores."""
    if score_name == "final":
        return match.final_score
    if score_name == "primary":
        return match.primary_score
    if match.tie_breaker_score is None:
        raise IconMatchingEvaluationError(
            "Secondary detection scoring is unavailable for the top result. "
            "Increase matching.tie_breaker.candidate_pool_size or select final/primary."
        )
    return match.tie_breaker_score


def _find_label_rank(matches, expected_label: str) -> int | None:
    """Return the one-based rank of an expected label in the full library result list."""
    for match in matches:
        if match.label == expected_label:
            return match.rank
    return None


def _safe_divide(numerator: float, denominator: float) -> float:
    """Return a stable zero when a metric has no applicable denominator."""
    return numerator / denominator if denominator else 0.0
