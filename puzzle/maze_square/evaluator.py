"""Maze puzzle evaluator for path-following tasks."""

from __future__ import annotations

from typing import Any, Dict, Optional

from .generator import MazeGenerator
from ..maze_base import MazeEvaluationResult, MazePuzzleEvaluator


class MazeEvaluator(MazePuzzleEvaluator):
    """Evaluate maze solutions by using the shared pixel-based maze pipeline."""

    RED_DOMINANCE = 80

    def _build_generator(self, record: Dict[str, Any]) -> MazeGenerator:
        rows, cols = record["grid_size"]
        cell_size = record["cell_size"]
        width, height = record["canvas_dimensions"]
        aspect = width / height
        return MazeGenerator(
            output_dir=self.base_dir,
            canvas_width=width,
            aspect=aspect,
            size=cell_size,
            rows=rows,
            cols=cols,
            cell_size=cell_size,
            prompt=record["prompt"],
            show_cell_id=False,
            video=False,
        )


__all__ = ["MazeEvaluator", "MazeEvaluationResult"]


def main(argv: Optional[list[str]] = None) -> None:
    MazeEvaluator.main(argv)


if __name__ == "__main__":
    MazeEvaluator.main()
