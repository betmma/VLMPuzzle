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

        solution_path = self.resolve_path(record["solution_image_path"])
        if not solution_path.exists():
            raise FileNotFoundError(f"Solution image not found: {solution_path}")
        with Image.open(solution_path) as solution_image_obj:
            solution_size = solution_image_obj.size

        candidate_image_obj = Image.open(candidate_path).convert("RGB")
        _ = trim_tolerance  # retained for backwards compatibility

        rows, cols = map(int, record["grid_size"])
        cell_size = int(record["cell_size"])
        padding = list(record.get("cell_padding", []))
        while len(padding) < 4:
            padding.append(0)
        pad_left, pad_top, _, _ = (int(value) for value in padding[:4])

        inner_bounds = record.get("cell_inner_bounds")
        if inner_bounds and len(inner_bounds) == 4:
            inner_left, inner_top, inner_right, inner_bottom = (int(v) for v in inner_bounds)
        else:
            inner_left = inner_top = 1
            inner_right = cell_size - 1
            inner_bottom = cell_size - 1

        inner_width = max(1, inner_right - inner_left)
        inner_height = max(1, inner_bottom - inner_top)
        shrink_x_src = max(0, inner_width // 8)
        shrink_y_src = max(0, inner_height // 8)

        source_bboxes = [
            [
                (
                    pad_left + c * cell_size,
                    pad_top + r * cell_size,
                    pad_left + (c + 1) * cell_size,
                    pad_top + (r + 1) * cell_size,
                )
                for c in range(cols)
            ]
            for r in range(rows)
        ]

        cell_bboxes = self.scale_cell_bboxes(
            source_bboxes,
            source_size=solution_size,
            target_size=candidate_image_obj.size,
            margin_px=0,
        )

        colored_cells = record["colored_cells"]
        left_colors = {(cell["row"], cell["col"]): tuple(cell["color"]) for cell in colored_cells}
        half_cols = cols // 2

        candidate_arr = np.asarray(candidate_image_obj)
        candidate_height, candidate_width = candidate_arr.shape[:2]

        breakdown: List[MirrorCellEvaluation] = []
        correct = 0
        total = rows * half_cols

        for row in range(rows):
            for right_col in range(half_cols, cols):
                mirror_col = half_cols - 1 - (right_col - half_cols)
                expected_color = left_colors.get((row, mirror_col), (255, 255, 255))
                expected_rgb = np.array(expected_color, dtype=np.float32)

                left, top, right, bottom = cell_bboxes[row][right_col]
                cell_width_scaled = max(1, right - left)
                cell_height_scaled = max(1, bottom - top)
                cell_scale_x = cell_width_scaled / cell_size
                cell_scale_y = cell_height_scaled / cell_size

                x0 = left + int(round(inner_left * cell_scale_x))
                x1 = left + int(round(inner_right * cell_scale_x))
                y0 = top + int(round(inner_top * cell_scale_y))
                y1 = top + int(round(inner_bottom * cell_scale_y))

                shrink_x = max(0, int(round(shrink_x_src * cell_scale_x)))
                shrink_y = max(0, int(round(shrink_y_src * cell_scale_y)))

                x0 = max(0, min(candidate_width - 1, x0))
                y0 = max(0, min(candidate_height - 1, y0))
                x1 = max(x0 + 1, min(candidate_width, x1))
                y1 = max(y0 + 1, min(candidate_height, y1))

                x0s = min(max(x0 + shrink_x, 0), candidate_width - 1)
                y0s = min(max(y0 + shrink_y, 0), candidate_height - 1)
                x1s = max(x0s + 1, min(candidate_width, x1 - shrink_x))
                y1s = max(y0s + 1, min(candidate_height, y1 - shrink_y))
                if x1s <= x0s or y1s <= y0s:
                    x0s, y0s, x1s, y1s = x0, y0, x1, y1

                cell_candidate = candidate_arr[y0s:y1s, x0s:x1s]
                actual_rgb = cell_candidate.mean(axis=(0, 1)) if cell_candidate.size else np.zeros(3)

                distance = float(np.linalg.norm(actual_rgb - expected_rgb))
                is_correct = distance <= color_tolerance
                if is_correct:
                    correct += 1
                breakdown.append(
                    MirrorCellEvaluation(
                        row=row,
                        col=right_col,
                        expected_color=tuple(int(value) for value in expected_color),
                        actual_color=tuple(int(value) for value in np.clip(np.round(actual_rgb), 0, 255)),
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


__all__ = ["MirrorEvaluator", "MirrorEvaluationResult", "MirrorCellEvaluation"]


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate mirror puzzles")
    parser.add_argument("metadata", type=Path)
    parser.add_argument("puzzle_id", type=str)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--base-dir", type=Path, default=None)
    parser.add_argument("--color-tolerance", type=float, default=60.0)
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
