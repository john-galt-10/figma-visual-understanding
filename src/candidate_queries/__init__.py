"""Provider-neutral candidate retrieval-query generation utilities."""

from .base import CandidateQueryGenerator
from .factory import create_candidate_query_generator
from .models import CandidateQueryResult, FocusBox

__all__ = [
    "CandidateQueryGenerator",
    "CandidateQueryResult",
    "FocusBox",
    "create_candidate_query_generator",
]
