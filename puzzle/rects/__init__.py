"""Colored-rectangles stacking order puzzle."""

from .generator import RectsGenerator, RectsPuzzleRecord
from .evaluator import RectsEvaluator, RectsEvaluationResult, OrderPosition

__all__ = [
    "RectsGenerator",
    "RectsEvaluator",
    "RectsPuzzleRecord",
    "RectsEvaluationResult",
    "OrderPosition",
]

