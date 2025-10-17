"""Ray intersection puzzle package."""

from .generator import (
    MidpointGenerator,
    CandidatePoint,
)
from .evaluator import (
    MidpointEvaluator,
)

__all__ = [
    "MidpointGenerator",
    "MidpointEvaluator",
    "CandidatePoint",
]
