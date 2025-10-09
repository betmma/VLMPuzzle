"""Evaluate ARC-AGI composite puzzles by reading the predicted test output grid."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

from ..base import AbstractPuzzleEvaluator, PathLike

ARC_PALETTE = {
    0: (0, 0, 0),
    1: (0, 0, 255),
    2: (255, 0, 0),
    3: (0, 255, 0),
    4: (255, 255, 0),
    5: (128, 128, 128),
    6: (255, 0, 255),
    7: (255, 128, 0),
    8: (0, 255, 255),
    9: (128, 64, 0),
}

try:
    RESAMPLE_LANCZOS = Image.Resampling.LANCZOS
except AttributeError:  # pragma: no cover
    RESAMPLE_LANCZOS = Image.LANCZOS


@dataclass
class ArcCellEvaluation:
    row: int
    col: int
    expected_value: int
    predicted_value: int
    expected_color: Tuple[int, int, int]
    predicted_color: Tuple[int, int, int]
    color_distance: float
    is_correct: bool

    def to_dict(self) -> dict:
        return {
            "row": self.row,
            "col": self.col,
            "expected_value": self.expected_value,
            "predicted_value": self.predicted_value,
            "expected_color": list(self.expected_color),
            "predicted_color": list(self.predicted_color),
            "color_distance": self.color_distance,
            "is_correct": self.is_correct,
        }


@dataclass
class ArcEvaluationResult:
    puzzle_id: str
    task_id: str
    accuracy: float
    correct_cells: int
    total_cells: int
    predicted_grid: List[List[int]]
    cell_breakdown: List[ArcCellEvaluation]

    def to_dict(self) -> dict:
        return {
            "puzzle_id": self.puzzle_id,
            "task_id": self.task_id,
            "accuracy": self.accuracy,
            "correct_cells": self.correct_cells,
            "total_cells": self.total_cells,
            "predicted_grid": self.predicted_grid,
            "cell_breakdown": [cell.to_dict() for cell in self.cell_breakdown],
        }


class ArcPuzzleEvaluator(AbstractPuzzleEvaluator):
    """Evaluate ARC-AGI puzzles by parsing the predicted test output grid."""

    def evaluate(
        self,
        puzzle_id: str,
        candidate_image: PathLike,
        *,
        trim_tolerance: int = 12,
    ) -> ArcEvaluationResult:
        record = self.get_record(puzzle_id)
        candidate_path = Path(candidate_image)
        if not candidate_path.exists():
            raise FileNotFoundError(f"Candidate image not found: {candidate_path}")

        solution_path = self.resolve_path(record["solution_image_path"])
        solution_image = Image.open(solution_path).convert("RGB")
        candidate_image_obj = Image.open(candidate_path)
        candidate_aligned = self._align(candidate_image_obj, solution_image.size, trim_tolerance)
        candidate_arr = np.asarray(candidate_aligned)

        placements = record.get("placements")
        test_bbox = self._find_test_bbox(placements)
        x0, y0, x1, y1 = test_bbox
        cell_size = int(record["cell_size"])
        rows = int(record["test_rows"])
        cols = int(record["test_cols"])
        expected_grid: List[List[int]] = record["test_output"]

        predicted_grid: List[List[int]] = []
        breakdown: List[ArcCellEvaluation] = []
        correct = 0
        total = rows * cols

        for r in range(rows):
            row_values: List[int] = []
            for c in range(cols):
                cell_bbox = self._cell_bbox(x0, y0, cell_size, r, c)
                sample = self._sample_cell(candidate_arr, cell_bbox)
                mean_rgb = sample.mean(axis=(0, 1)) if sample.size else np.array([0, 0, 0])
                predicted_value, predicted_color = self._closest_color(mean_rgb)

                expected_value = int(expected_grid[r][c])
                expected_color = ARC_PALETTE.get(expected_value, ARC_PALETTE[0])
                color_distance = self._color_distance(mean_rgb, expected_color)
                is_correct = predicted_value == expected_value
                if is_correct:
                    correct += 1

                breakdown.append(
                    ArcCellEvaluation(
                        row=r,
                        col=c,
                        expected_value=expected_value,
                        predicted_value=predicted_value,
                        expected_color=expected_color,
                        predicted_color=predicted_color,
                        color_distance=color_distance,
                        is_correct=is_correct,
                    )
                )
                row_values.append(predicted_value)
            predicted_grid.append(row_values)

        accuracy = correct / total if total else 0.0
        return ArcEvaluationResult(
            puzzle_id=puzzle_id,
            task_id=str(record.get("task_id", "")),
            accuracy=accuracy,
            correct_cells=correct,
            total_cells=total,
            predicted_grid=predicted_grid,
            cell_breakdown=breakdown,
        )

    def _align(
        self,
        image: Image.Image,
        reference_size: Tuple[int, int],
        trim_tolerance: int,
    ) -> Image.Image:
        """Resize candidate image to match the solution size without trimming.

        The trim_tolerance parameter is ignored to preserve interface compatibility.
        """
        image = image.convert("RGB")
        if image.size == reference_size:
            return image
        return image.resize(reference_size, RESAMPLE_LANCZOS)

    @staticmethod
    def _trim_borders(image: Image.Image, *, tolerance: int = 12) -> Image.Image:
        arr = np.asarray(image)
        if arr.ndim == 3:
            ref = arr[0, 0]
            diff = np.max(np.abs(arr - ref), axis=2)
        else:
            ref = arr[0, 0]
            diff = np.abs(arr - ref)
        mask = diff > tolerance
        if not np.any(mask):
            return image
        ys, xs = np.where(mask)
        top, bottom = int(ys.min()), int(ys.max())
        left, right = int(xs.min()), int(xs.max())
        return image.crop((left, top, right + 1, bottom + 1))

    def _find_test_bbox(self, placements: List[dict]) -> Tuple[int, int, int, int]:
        if placements is None:
            raise KeyError("Puzzle metadata is missing placements")
        for placement in placements:
            if placement.get("kind") == "test_output":
                bbox = placement.get("bbox")
                if not bbox or len(bbox) != 4:
                    break
                return tuple(int(v) for v in bbox)
        raise KeyError("Metadata does not include a test_output placement")

    def _cell_bbox(self, x0: int, y0: int, cell_size: int, row: int, col: int) -> Tuple[int, int, int, int]:
        left = x0 + col * cell_size
        top = y0 + row * cell_size
        right = left + cell_size
        bottom = top + cell_size
        shrink = max(1, cell_size // 6)
        return (
            left + shrink,
            top + shrink,
            right - shrink,
            bottom - shrink,
        )

    def _sample_cell(self, array: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        x0, y0, x1, y1 = bbox
        x0_clamped = max(0, min(array.shape[1] - 1, x0))
        y0_clamped = max(0, min(array.shape[0] - 1, y0))
        x1_clamped = max(x0_clamped + 1, min(array.shape[1], x1))
        y1_clamped = max(y0_clamped + 1, min(array.shape[0], y1))
        return array[y0_clamped:y1_clamped, x0_clamped:x1_clamped]

    def _closest_color(self, rgb: np.ndarray) -> Tuple[int, Tuple[int, int, int]]:
        best_value = 0
        best_distance = math.inf
        best_color = ARC_PALETTE[0]
        for value, color in ARC_PALETTE.items():
            distance = self._color_distance(rgb, color)
            if distance < best_distance:
                best_distance = distance
                best_value = value
                best_color = color
        return best_value, best_color

    @staticmethod
    def _color_distance(rgb: np.ndarray, color: Tuple[int, int, int]) -> float:
        diff = rgb - np.array(color, dtype=np.float32)
        return float(np.linalg.norm(diff))


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ARC-AGI composite puzzles")
    parser.add_argument("metadata", type=Path, help="Path to ARC puzzle metadata JSON")
    parser.add_argument("puzzle_id", type=str, help="Identifier of the puzzle record")
    parser.add_argument("candidate", type=Path, help="Image containing the model prediction")
    parser.add_argument("--base-dir", type=Path, default=None, help="Override base directory for relative paths")
    parser.add_argument("--trim-tolerance", type=int, default=12)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)
    evaluator = ArcPuzzleEvaluator(args.metadata, base_dir=args.base_dir)
    result = evaluator.evaluate(args.puzzle_id, args.candidate, trim_tolerance=args.trim_tolerance)
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
