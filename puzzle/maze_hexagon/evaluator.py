"""Evaluator for hexagonal maze puzzles."""

from __future__ import annotations

from typing import Any, Dict, Optional

from .generator import MazeHexagonGenerator
from ..maze_base import MazeEvaluationResult, MazePuzzleEvaluator


class MazeHexagonEvaluator(MazePuzzleEvaluator):
    """Reuse the shared maze evaluation while tweaking color sensitivity for thin walls."""

    RED_DOMINANCE = 75

    def _build_generator(self, record: Dict[str, Any]) -> MazeHexagonGenerator:
        width, height = record["canvas_dimensions"]
        aspect = width / height
        return MazeHexagonGenerator(
            output_dir=self.base_dir,
            radius=record["radius"],
            cell_radius=record["cell_radius"],
            wall_thickness=record["wall_thickness"],
            canvas_width=width,
            aspect=aspect,
            prompt=record["prompt"],
            show_cell_id=False,
            video=False,
        )


__all__ = ["MazeHexagonEvaluator", "MazeEvaluationResult"]


def main(argv: Optional[list[str]] = None) -> None:
    MazeHexagonEvaluator.main(argv)


if __name__ == "__main__":
    MazeHexagonEvaluator.main()
