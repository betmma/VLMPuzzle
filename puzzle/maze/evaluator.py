"""Maze puzzle evaluator for path-following tasks."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

try:
    RESAMPLE_LANCZOS = Image.Resampling.LANCZOS
except AttributeError:  # pragma: no cover
    RESAMPLE_LANCZOS = Image.LANCZOS

from ..base import AbstractPuzzleEvaluator, PathLike

RED_THRESHOLD = 150
RED_DOMINANCE = 80


@dataclass
class MazeEvaluationResult:
    puzzle_id: str
    connected: bool
    touches_goal: bool
    stray_in_walls: bool
    red_cells: List[Tuple[int, int]]
    message: str

    def to_dict(self) -> dict:
        return {
            "puzzle_id": self.puzzle_id,
            "connected": self.connected,
            "touches_goal": self.touches_goal,
            "stray_in_walls": self.stray_in_walls,
            "red_cells": [list(cell) for cell in self.red_cells],
            "message": self.message,
        }


class MazeEvaluator(AbstractPuzzleEvaluator):
    """Evaluate maze solutions by checking a red path from start to goal."""

    def evaluate(
        self,
        puzzle_id: str,
        candidate_image: PathLike,
        *,
        trim_tolerance: int = 12,
    ) -> MazeEvaluationResult:
        record = self.get_record(puzzle_id)
        candidate_path = Path(candidate_image)
        if not candidate_path.exists():
            raise FileNotFoundError(f"Candidate image not found: {candidate_path}")

        solution_path = self.resolve_path(record["solution_image_path"])
        solution_image = Image.open(solution_path).convert("RGB")
        candidate_image_obj = Image.open(candidate_path).convert("RGB")
        candidate_processed = self._align(candidate_image_obj, solution_image.size, trim_tolerance)

        maze_grid: List[List[int]] = [list(map(int, row)) for row in record["maze_grid"]]
        rows = len(maze_grid)
        cols = len(maze_grid[0]) if rows else 0
        cell_bboxes = [
            [tuple(map(int, bbox)) for bbox in row]
            for row in record["cell_bboxes"]
        ]
        start = tuple(map(int, record["start"]))
        goal = tuple(map(int, record["goal"]))

        candidate_arr = np.asarray(candidate_processed)

        red_cells: List[Tuple[int, int]] = []
        stray_in_walls = False

        for r in range(rows):
            for c in range(cols):
                left, top, right, bottom = cell_bboxes[r][c]
                margin = max(1, (right - left) // 6)
                x0 = left + margin
                y0 = top + margin
                x1 = right - margin
                y1 = bottom - margin
                cell_slice = candidate_arr[y0:y1, x0:x1]
                if cell_slice.size == 0:
                    continue
                if self._is_red(cell_slice):
                    if maze_grid[r][c] == 1:
                        stray_in_walls = True
                    else:
                        red_cells.append((r, c))

        connected, touches_goal = self._check_connectivity(red_cells, start, goal)

        success = connected and touches_goal and not stray_in_walls
        if not red_cells:
            message = "No red path detected."
        elif stray_in_walls:
            message = "Red path overlaps walls."
        elif not touches_goal:
            message = "Red path does not reach the goal."
        elif not connected:
            message = "Red path is not continuous from start to goal."
        else:
            message = "Red path successfully connects start to goal."

        return MazeEvaluationResult(
            puzzle_id=puzzle_id,
            connected=connected,
            touches_goal=touches_goal,
            stray_in_walls=stray_in_walls,
            red_cells=red_cells,
            message=message,
        )

    # ------------------------------------------------------------------

    def _align(
        self,
        image: Image.Image,
        reference_size: Tuple[int, int],
        trim_tolerance: int,
    ) -> Image.Image:
        if image.size == reference_size:
            return image
        trimmed = self._trim_borders(image, tolerance=trim_tolerance)
        if trimmed.size != reference_size:
            trimmed = trimmed.resize(reference_size, RESAMPLE_LANCZOS)
        return trimmed

    @staticmethod
    def _trim_borders(image: Image.Image, *, tolerance: int = 12) -> Image.Image:
        arr = np.asarray(image)
        if arr.size == 0:
            return image
        if arr.ndim == 3:
            diff = np.max(np.abs(arr - arr[0, 0]), axis=2)
        else:
            diff = np.abs(arr - arr[0, 0])
        mask = diff > tolerance
        if not np.any(mask):
            return image
        ys, xs = np.where(mask)
        top, bottom = int(ys.min()), int(ys.max())
        left, right = int(xs.min()), int(xs.max())
        return image.crop((left, top, right + 1, bottom + 1))

    @staticmethod
    def _is_red(pixels: np.ndarray) -> bool:
        flat = pixels.reshape(-1, 3).astype(np.float32)
        if flat.size == 0:
            return False
        r = flat[:, 0]
        g = flat[:, 1]
        b = flat[:, 2]
        dominance = r - np.maximum(g, b)
        return bool(np.any((r >= RED_THRESHOLD) & (dominance >= RED_DOMINANCE)))

    def _check_connectivity(
        self,
        red_cells: Sequence[Tuple[int, int]],
        start: Tuple[int, int],
        goal: Tuple[int, int],
    ) -> Tuple[bool, bool]:
        red_set = set(red_cells)
        if start not in red_set or goal not in red_set:
            return False, goal in red_set
        queue = [start]
        visited = {start}
        while queue:
            r, c = queue.pop(0)
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if (nr, nc) in red_set and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append((nr, nc))
        return goal in visited, goal in red_set


__all__ = ["MazeEvaluator", "MazeEvaluationResult"]


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate maze puzzles")
    parser.add_argument("metadata", type=Path, help="Path to maze puzzles metadata JSON")
    parser.add_argument("puzzle_id", type=str, help="Identifier of the puzzle to evaluate")
    parser.add_argument("candidate", type=Path, help="Image containing the candidate solution")
    parser.add_argument("--base-dir", type=Path, default=None)
    parser.add_argument("--trim-tolerance", type=int, default=12)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)
    evaluator = MazeEvaluator(args.metadata, base_dir=args.base_dir)
    result = evaluator.evaluate(args.puzzle_id, args.candidate, trim_tolerance=args.trim_tolerance)
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
