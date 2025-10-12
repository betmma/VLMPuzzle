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
    "ArcPuzzleGenerator",
    "ArcPuzzleEvaluator",
    "ArcPuzzleRecord",
    "ArcEvaluationResult",
    "ArcCellEvaluation",
    "MazeGenerator",
    "MazeEvaluator",
    "MazePuzzleRecord",
    "MazeEvaluationResult",
    "PieceEvaluation",
    "CellEvaluation",
    "RayGenerator",
    "RayEvaluator",
    "RayPuzzleRecord",
    "RayEvaluationResult",
    "ArcConnectGenerator",
    "ArcConnectEvaluator",
    "ArcConnectPuzzleRecord",
    "ArcConnectEvaluationResult",
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
from .arcagi import (
    ArcPuzzleGenerator,
    ArcPuzzleEvaluator,
    ArcPuzzleRecord,
    ArcEvaluationResult,
    ArcCellEvaluation,
)
from .maze import (
    MazeGenerator,
    MazeEvaluator,
    MazePuzzleRecord,
    MazeEvaluationResult,
)
from .rects import (
    RectsGenerator,
    RectsEvaluator,
    RectsPuzzleRecord,
    RectsEvaluationResult,
)

# Ray-and-mirrors (speak option via NATO)
from .ray import (
    RayGenerator,
    RayEvaluator,
    RayPuzzleRecord,
    RayEvaluationResult,
)

# Arc connection (speak option via NATO)
from .arc_connect import (
    ArcConnectGenerator,
    ArcConnectEvaluator,
    ArcConnectPuzzleRecord,
    ArcConnectEvaluationResult,
)

