"""Puzzle generation and evaluation toolkit."""

__all__ = [
    "AbstractPuzzleGenerator",
    "AbstractPuzzleEvaluator",
    "JigsawGenerator",
    "JigsawEvaluator",
    "JigsawPuzzleRecord",
    "JigsawEvaluationResult",
    "SudokuGenerator",
    "SudokuEvaluator",
    "SudokuPuzzleRecord",
    "SudokuEvaluationResult",
    "MirrorGenerator",
    "MirrorEvaluator",
    "MirrorPuzzleRecord",
    "MirrorEvaluationResult",
    "MirrorCellEvaluation",
    "PieceEvaluation",
    "CellEvaluation",
]

from .base import AbstractPuzzleGenerator, AbstractPuzzleEvaluator
from .jigsaw import (
    JigsawGenerator,
    JigsawEvaluator,
    JigsawPuzzleRecord,
    JigsawEvaluationResult,
    PieceEvaluation,
)
from .sudoku import (
    SudokuGenerator,
    SudokuEvaluator,
    SudokuPuzzleRecord,
    SudokuEvaluationResult,
    CellEvaluation,
)

from .mirror import (
    MirrorGenerator,
    MirrorEvaluator,
    MirrorPuzzleRecord,
    MirrorEvaluationResult,
    MirrorCellEvaluation,
)
