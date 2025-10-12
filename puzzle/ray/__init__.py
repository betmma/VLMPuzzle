"""Ray-and-mirrors puzzle (trace the ray to a labeled point)."""

from .generator import RayGenerator, RayPuzzleRecord, MirrorSpec, PointSpec, RayStart
from .evaluator import RayEvaluator, RayEvaluationResult

__all__ = [
    "RayGenerator",
    "RayEvaluator",
    "RayPuzzleRecord",
    "RayEvaluationResult",
    "MirrorSpec",
    "PointSpec",
    "RayStart",
]

