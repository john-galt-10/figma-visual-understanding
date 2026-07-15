"""Classical-CV proposals for icon regions in cropped Figma screenshots."""

from .config import IconMatchingSettings, load_settings
from .pipeline import IconCandidateGenerator

__all__ = ["IconCandidateGenerator", "IconMatchingSettings", "load_settings"]
