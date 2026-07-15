"""Provider-neutral matching of normalized icon crops against a template library."""

from .evaluation import IconMatchingEvaluator, mine_detection_threshold
from .pipeline import IconLibraryMatcher

__all__ = ["IconLibraryMatcher", "IconMatchingEvaluator", "mine_detection_threshold"]
