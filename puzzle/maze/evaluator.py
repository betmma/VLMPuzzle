"""Maze puzzle evaluator for path-following tasks."""

from __future__ import annotations

from typing import Optional

from ..maze_base import MazeEvaluationResult, MazePuzzleEvaluator


class MazeEvaluator(MazePuzzleEvaluator):
    """Evaluate maze solutions by using the shared pixel-based maze pipeline."""

    RED_DOMINANCE = 80


__all__ = ["MazeEvaluator", "MazeEvaluationResult"]


def main(argv: Optional[list[str]] = None) -> None:
    MazeEvaluator.main(argv)


if __name__ == "__main__":
    MazeEvaluator.main()
