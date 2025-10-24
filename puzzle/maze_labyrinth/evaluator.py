"""Evaluator for circular labyrinth maze puzzles."""

from __future__ import annotations

from typing import Optional

from ..maze_base import MazeEvaluationResult, MazePuzzleEvaluator


class MazeLabyrinthEvaluator(MazePuzzleEvaluator):
    """Reuse the shared pixel-based evaluation with adjusted color sensitivity."""

    RED_DOMINANCE = 75


__all__ = ["MazeLabyrinthEvaluator", "MazeEvaluationResult"]


def main(argv: Optional[list[str]] = None) -> None:
    MazeLabyrinthEvaluator.main(argv)


if __name__ == "__main__":
    MazeLabyrinthEvaluator.main()
