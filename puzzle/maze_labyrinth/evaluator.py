"""Evaluator for circular labyrinth maze puzzles."""

from __future__ import annotations

from typing import Any, Dict, Optional

from .generator import MazeLabyrinthGenerator
from ..maze_base import MazeEvaluationResult, MazePuzzleEvaluator


class MazeLabyrinthEvaluator(MazePuzzleEvaluator):
    """Reuse the shared pixel-based evaluation with adjusted color sensitivity."""

    RED_DOMINANCE = 75

    def _build_generator(self, record: Dict[str, Any]) -> MazeLabyrinthGenerator:
        width, height = record["canvas_dimensions"]
        aspect = width / height
        return MazeLabyrinthGenerator(
            output_dir=self.base_dir,
            rings=record["rings"],
            segments=record["segments"],
            ring_width=record["ring_width"],
            wall_thickness=record["wall_thickness"],
            canvas_width=width,
            aspect=aspect,
            prompt=record["prompt"],
            show_cell_id=False,
            video=False,
        )


__all__ = ["MazeLabyrinthEvaluator", "MazeEvaluationResult"]


def main(argv: Optional[list[str]] = None) -> None:
    MazeLabyrinthEvaluator.main(argv)


if __name__ == "__main__":
    MazeLabyrinthEvaluator.main()
