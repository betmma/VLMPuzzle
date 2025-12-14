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
    "MazeLabyrinthGenerator",
    "MazeLabyrinthEvaluator",
    "MazeHexagonGenerator",
    "MazeHexagonEvaluator",
    "MazePuzzleRecord",
    "MazeEvaluationResult",
    "PieceEvaluation",
    "CellEvaluation",
    "RayGenerator",
    "RayEvaluator",
    "RayPuzzleRecord",
    "RayEvaluationResult",
    "RayIntersectionGenerator",
    "RayIntersectionEvaluator",
    "RayIntersectionPuzzleRecord",
    "RayIntersectionEvaluationResult",
    "MidpointGenerator",
    "MidpointEvaluator",
    "Segment",
    "ArcConnectGenerator",
    "ArcConnectEvaluator",
    "ArcConnectPuzzleRecord",
    "ArcConnectEvaluationResult",
    "CircleCountGenerator",
    "CircleCountEvaluator",
    "CircleCountPuzzleRecord",
    "CircleCountEvaluationResult",
    "transcribe_video",
]

from .base import AbstractPuzzleEvaluator, AbstractPuzzleGenerator
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
from .maze_square import (
    MazeGenerator,
    MazeEvaluator,
    MazePuzzleRecord,
    MazeEvaluationResult,
)
from .maze_labyrinth import (
    MazeLabyrinthGenerator,
    MazeLabyrinthEvaluator,
)
from .maze_hexagon import (
    MazeHexagonGenerator,
    MazeHexagonEvaluator,
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

from .ray_intersection import (
    RayIntersectionGenerator,
    RayIntersectionEvaluator,
)


# Arc connection (speak option via NATO)
from .arc_connect import (
    ArcConnectGenerator,
    ArcConnectEvaluator,
    ArcConnectPuzzleRecord,
)

from .circle_count import (
    CircleCountGenerator,
    CircleCountEvaluator,
    CircleCountPuzzleRecord,
    CircleCountEvaluationResult,
)

