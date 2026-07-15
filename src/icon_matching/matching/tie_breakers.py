"""Optional secondary ranking signals for icon-library matches."""

from __future__ import annotations

import cv2

from icon_matching.normalization import NormalizedIcon

from .base import IconTieBreaker
from .models import TemplateRecord


class SoftNccTieBreaker(IconTieBreaker):
    """Compare soft canvases with normalized cross-correlation."""

    name = "soft_ncc"

    def score(self, query: NormalizedIcon, template: TemplateRecord) -> float:
        """Return soft-canvas correlation mapped from OpenCV's -1..1 range to 0..1."""
        correlation = float(cv2.matchTemplate(query.soft_canvas, template.soft_canvas, cv2.TM_CCOEFF_NORMED)[0, 0])
        if correlation != correlation:
            correlation = -1.0
        return max(0.0, min(1.0, (correlation + 1.0) / 2.0))
