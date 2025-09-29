"""Mirror puzzle evaluator for symmetry completion tasks."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

from ..base import AbstractPuzzleEvaluator, PathLike

try:
    RESAMPLE_LANCZOS = Image.Resampling.LANCZOS
except AttributeError:  # pragma: no cover
    RESAMPLE_LANCZOS = Image.LANCZOS


@dataclass
class MirrorCellEvaluation:
    row: int
    col: int
    expected_color: Tuple[int, int, int]
    actual_color: Tuple[int, int, int]
    distance: float
    is_correct: bool

    def to_dict(self) -> dict:
        return {
            "row": self.row,
            "col": self.col,
            "expected_color": list(self.expected_color),
            "actual_color": list(self.actual_color),
            "distance": self.distance,
            "is_correct": self.is_correct,
        }


@dataclass
class MirrorEvaluationResult:
    puzzle_id: str
    correct_cells: int
    total_cells: int
    accuracy: float
    cell_breakdown: List[MirrorCellEvaluation]

    def to_dict(self) -> dict:
        return {
            "puzzle_id": self.puzzle_id,
            "correct_cells": self.correct_cells,
            "total_cells": self.total_cells,
            "accuracy": self.accuracy,
            "cell_breakdown": [cell.to_dict() for cell in self.cell_breakdown],
        }


class MirrorEvaluator(AbstractPuzzleEvaluator):
    """Evaluate mirrored-color puzzles by comparing right-half cell averages."""

    def evaluate(
        self,
        puzzle_id: str,
        candidate_image: PathLike,
        *,
        trim_tolerance: int = 12,
        color_tolerance: float = 20.0,
    ) -> MirrorEvaluationResult:
        record = self.get_record(puzzle_id)
        candidate_path = Path(candidate_image)
        if not candidate_path.exists():
            raise FileNotFoundError(f"Candidate image not found: {candidate_path}")

        solution_image = Image.open(self.resolve_path(record["solution_image_path"]))
        candidate_image_obj = Image.open(candidate_path)

        candidate_processed = self._align(candidate_image_obj, solution_image.size, trim_tolerance)

        rows, cols = map(int, record["grid_size"])
        cell_size = int(record["cell_size"])
        colored_cells = record["colored_cells"]
        left_colors = {(cell["row"], cell["col"]): tuple(cell["color"]) for cell in colored_cells}
        half_cols = cols // 2

        solution_arr = np.asarray(solution_image)
        candidate_arr = np.asarray(candidate_processed)

        breakdown: List[MirrorCellEvaluation] = []
        correct = 0
        total = rows * half_cols

        for row in range(rows):
            for right_col in range(half_cols, cols):
                mirror_col = half_cols - 1 - (right_col - half_cols)
                expected_color = left_colors.get((row, mirror_col), (255, 255, 255))
                expected_rgb = np.array(expected_color, dtype=np.float32)

                y0 = row * cell_size
                y1 = y0 + cell_size
                x0 = right_col * cell_size
                x1 = x0 + cell_size
                shrink = max(1, cell_size // 8)
                y0s = min(max(y0 + shrink, 0), candidate_arr.shape[0] - 1)
                y1s = max(min(y1 - shrink, candidate_arr.shape[0]), y0s + 1)
                x0s = min(max(x0 + shrink, 0), candidate_arr.shape[1] - 1)
                x1s = max(min(x1 - shrink, candidate_arr.shape[1]), x0s + 1)

                cell_candidate = candidate_arr[y0s:y1s, x0s:x1s]
                actual_rgb = cell_candidate.mean(axis=(0, 1))

                distance = float(np.linalg.norm(actual_rgb - expected_rgb))
                is_correct = distance <= color_tolerance
                if is_correct:
                    correct += 1
                breakdown.append(
                    MirrorCellEvaluation(
                        row=row,
                        col=right_col,
                        expected_color=tuple(map(int, expected_color)),
                        actual_color=tuple(map(int, actual_rgb.round().astype(int))),
                        distance=distance,
                        is_correct=is_correct,
                    )
                )

        accuracy = correct / total if total else 0.0
        return MirrorEvaluationResult(
            puzzle_id=puzzle_id,
            correct_cells=correct,
            total_cells=total,
            accuracy=accuracy,
            cell_breakdown=breakdown,
        )

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


__all__ = ["MirrorEvaluator", "MirrorEvaluationResult", "MirrorCellEvaluation"]


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate mirror puzzles")
    parser.add_argument("metadata", type=Path)
    parser.add_argument("puzzle_id", type=str)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--base-dir", type=Path, default=None)
    parser.add_argument("--color-tolerance", type=float, default=20.0)
    parser.add_argument("--trim-tolerance", type=int, default=12)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)
    evaluator = MirrorEvaluator(args.metadata, base_dir=args.base_dir)
    result = evaluator.evaluate(
        args.puzzle_id,
        args.candidate,
        color_tolerance=args.color_tolerance,
        trim_tolerance=args.trim_tolerance,
    )
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
