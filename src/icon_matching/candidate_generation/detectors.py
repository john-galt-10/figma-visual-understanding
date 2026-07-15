"""Classical-CV icon-region proposal detectors."""

from __future__ import annotations

from itertools import combinations

import numpy as np

from .base import CandidateGenerationDependencyError, RegionDetector
from .config import ContourSettings, MorphologySettings
from .models import BoundingBox, RegionProposal


def _cv2():
    """Import OpenCV lazily so configuration work has no native dependency cost."""
    try:
        import cv2
    except ImportError as error:
        raise CandidateGenerationDependencyError(
            "OpenCV is not installed. Install dependencies with: pip install -r requirements.txt"
        ) from error
    return cv2


class MorphologyConnectedComponentsDetector(RegionDetector):
    """Find locally contrasting pixel clusters and group nearby icon strokes."""

    name = "morphology"

    def __init__(self, settings: MorphologySettings) -> None:
        """Store the detector-specific settings selected for this run."""
        self.settings = settings

    def detect(self, grayscale_image: np.ndarray) -> list[RegionProposal]:
        """Return components from a local-contrast foreground mask."""
        cv2 = _cv2()
        local_background = cv2.GaussianBlur(
            grayscale_image,
            (self.settings.adaptive_block_size, self.settings.adaptive_block_size),
            0,
        )
        contrast_mask = cv2.absdiff(grayscale_image, local_background)
        _, local_mask = cv2.threshold(
            contrast_mask,
            self.settings.local_contrast_threshold,
            255,
            cv2.THRESH_BINARY,
        )
        adaptive_mask = cv2.adaptiveThreshold(
            grayscale_image,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            self.settings.adaptive_block_size,
            self.settings.adaptive_constant,
        )
        mask = cv2.bitwise_or(local_mask, adaptive_mask)
        if self.settings.dilation_iterations:
            kernel = np.ones(
                (self.settings.dilation_kernel_size, self.settings.dilation_kernel_size), dtype=np.uint8
            )
            mask = cv2.dilate(mask, kernel, iterations=self.settings.dilation_iterations)
        component_count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        proposals: list[RegionProposal] = []
        for index in range(1, component_count):
            x, y, width, height, area = (int(value) for value in stats[index])
            if area < self.settings.minimum_component_area:
                continue
            proposals.append(
                RegionProposal(
                    bbox=BoundingBox(x=x, y=y, width=width, height=height),
                    detector=self.name,
                    evidence={"component_area": area},
                )
            )
        return proposals


class CannyContourDetector(RegionDetector):
    """Find Canny contours, then join close contour boxes into icon proposals."""

    name = "contours"

    def __init__(self, settings: ContourSettings) -> None:
        """Store the detector-specific settings selected for this run."""
        self.settings = settings

    def detect(self, grayscale_image: np.ndarray) -> list[RegionProposal]:
        """Return grouped bounding boxes for qualifying Canny contours."""
        cv2 = _cv2()
        edges = cv2.Canny(
            grayscale_image,
            self.settings.canny_low_threshold,
            self.settings.canny_high_threshold,
        )
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < self.settings.minimum_contour_area:
                continue
            x, y, width, height = cv2.boundingRect(contour)
            boxes.append(BoundingBox(x=int(x), y=int(y), width=int(width), height=int(height)))
        grouped_boxes = _group_nearby_boxes(boxes, self.settings.grouping_distance)
        return [
            RegionProposal(
                bbox=box,
                detector=self.name,
                evidence={"grouped_contour_count": _contained_box_count(box, boxes)},
            )
            for box in grouped_boxes
        ]


def _group_nearby_boxes(boxes: list[BoundingBox], distance: int) -> list[BoundingBox]:
    """Iteratively union boxes whose horizontal and vertical gaps are both small."""
    groups = list(boxes)
    changed = True
    while changed:
        changed = False
        for first_index, second_index in combinations(range(len(groups)), 2):
            if _box_gap(groups[first_index], groups[second_index]) <= distance:
                groups[first_index] = _union_box(groups[first_index], groups[second_index])
                del groups[second_index]
                changed = True
                break
    return groups


def _box_gap(first: BoundingBox, second: BoundingBox) -> int:
    """Return the largest axis gap between two rectangles, or zero when touching."""
    horizontal_gap = max(first.x - second.right, second.x - first.right, 0)
    vertical_gap = max(first.y - second.bottom, second.y - first.bottom, 0)
    return max(horizontal_gap, vertical_gap)


def _union_box(first: BoundingBox, second: BoundingBox) -> BoundingBox:
    """Return the smallest rectangle covering both input rectangles."""
    x = min(first.x, second.x)
    y = min(first.y, second.y)
    right = max(first.right, second.right)
    bottom = max(first.bottom, second.bottom)
    return BoundingBox(x=x, y=y, width=right - x, height=bottom - y)


def _contained_box_count(container: BoundingBox, boxes: list[BoundingBox]) -> int:
    """Count original contour boxes contained in one grouped contour region."""
    return sum(
        box.x >= container.x
        and box.y >= container.y
        and box.right <= container.right
        and box.bottom <= container.bottom
        for box in boxes
    )
