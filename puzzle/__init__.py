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
