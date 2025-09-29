"""Sudoku puzzle evaluator implementation using OCR."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pytesseract
from PIL import Image, ImageOps, ImageFilter

from ..base import AbstractPuzzleEvaluator, PathLike

try:  # Pillow 9/10 compatibility
    RESAMPLE_LANCZOS = Image.Resampling.LANCZOS
except AttributeError:  # pragma: no cover - older Pillow
    RESAMPLE_LANCZOS = Image.LANCZOS


@dataclass
class CellEvaluation:
    """Per-cell OCR result."""

    row: int
    col: int
    expected: int
    predicted: Optional[int]
    confidence: float
    is_correct: bool
    is_clue: bool

    def to_dict(self) -> dict:
        return {
            "row": self.row,
            "col": self.col,
            "expected": self.expected,
            "predicted": self.predicted,
            "confidence": self.confidence,
            "is_correct": self.is_correct,
            "is_clue": self.is_clue,
        }


@dataclass
class SudokuEvaluationResult:
    """Aggregate evaluation for a Sudoku puzzle."""

    puzzle_id: str
    correct_cells: int
    total_cells: int
    accuracy: float
    is_valid_solution: bool
    cell_breakdown: List[CellEvaluation]

    def to_dict(self) -> dict:
        return {
            "puzzle_id": self.puzzle_id,
            "correct_cells": self.correct_cells,
            "total_cells": self.total_cells,
            "accuracy": self.accuracy,
            "is_valid_solution": self.is_valid_solution,
            "cell_breakdown": [cell.to_dict() for cell in self.cell_breakdown],
        }


class SudokuEvaluator(AbstractPuzzleEvaluator):
    """Evaluate Sudoku solutions by reading digits from images with OCR."""

    def evaluate(
        self,
        puzzle_id: str,
        candidate_image: PathLike,
        *,
        trim_tolerance: int = 12,
    ) -> SudokuEvaluationResult:
        record = self.get_record(puzzle_id)

        solution_path = self.resolve_path(record["solution_image_path"])
        candidate_path = Path(candidate_image)
        if not solution_path.exists():
            raise FileNotFoundError(f"Solution image missing: {solution_path}")
        if not candidate_path.exists():
            raise FileNotFoundError(f"Candidate image missing: {candidate_path}")

        solution_image = Image.open(solution_path).convert("RGB")
        candidate_image_obj = Image.open(candidate_path).convert("RGB")

        processed_candidate = self._align_candidate(candidate_image_obj, solution_image.size, trim_tolerance)

        solution_grid = [[int(value) for value in row] for row in record["solution_grid"]]
        puzzle_grid = [[int(value) for value in row] for row in record["puzzle_grid"]]
        grid_size = len(solution_grid)
        cell_bboxes = self._load_cell_bboxes(record, solution_image.size, grid_size)

        predicted_grid: List[List[int]] = [[0 for _ in range(grid_size)] for _ in range(grid_size)]
        breakdown: List[CellEvaluation] = []
        correct = 0

        for row_idx, row in enumerate(solution_grid):
            for col_idx, expected in enumerate(row):
                raw_bbox = cell_bboxes[row_idx][col_idx]
                bbox = self._shrink_bbox(raw_bbox, processed_candidate.size, shrink=2)
                ref_tile = solution_image.crop(bbox)
                tile = processed_candidate.crop(bbox)
                similarity = self._tile_similarity(ref_tile, tile)
                if similarity >= 0.985:
                    predicted, confidence = expected, float(similarity)
                else:
                    predicted, confidence = self._extract_digit(tile)
                is_correct = predicted == expected
                if is_correct:
                    correct += 1
                predicted_grid[row_idx][col_idx] = predicted or 0
                breakdown.append(
                    CellEvaluation(
                        row=row_idx,
                        col=col_idx,
                        expected=expected,
                        predicted=predicted,
                        confidence=confidence,
                        is_correct=is_correct,
                        is_clue=puzzle_grid[row_idx][col_idx] != 0,
                    )
                )

        total_cells = grid_size * grid_size if grid_size else 0
        accuracy = correct / total_cells if total_cells else 0.0
        is_valid = self._is_valid_solution(predicted_grid)

        return SudokuEvaluationResult(
            puzzle_id=puzzle_id,
            correct_cells=correct,
            total_cells=total_cells,
            accuracy=accuracy,
            is_valid_solution=is_valid,
            cell_breakdown=breakdown,
        )

    def _tile_similarity(self, reference: Image.Image, candidate: Image.Image) -> float:
        ref_arr = np.asarray(reference.convert("L"), dtype=np.float32) / 255.0
        cand_arr = np.asarray(candidate.convert("L"), dtype=np.float32) / 255.0
        if ref_arr.shape != cand_arr.shape:
            candidate = candidate.resize(reference.size, RESAMPLE_LANCZOS)
            cand_arr = np.asarray(candidate.convert("L"), dtype=np.float32) / 255.0
        mae = float(np.mean(np.abs(ref_arr - cand_arr)))
        return float(max(0.0, 1.0 - mae))

    # ------------------------------------------------------------------

    def _align_candidate(
        self,
        candidate: Image.Image,
        reference_size: Tuple[int, int],
        trim_tolerance: int,
    ) -> Image.Image:
        if candidate.size == reference_size:
            return candidate
        trimmed = self._trim_borders(candidate, tolerance=trim_tolerance)
        if trimmed.size != reference_size:
            trimmed = trimmed.resize(reference_size, RESAMPLE_LANCZOS)
        return trimmed

    def _extract_digit(self, tile: Image.Image) -> Tuple[Optional[int], float]:
        gray = tile.convert("L")
        arr = np.asarray(gray, dtype=np.uint8)
        dark_ratio = float((arr < 215).sum()) / arr.size
        if dark_ratio < 0.008:
            return None, 0.0

        processed = ImageOps.autocontrast(gray)
        scale = 4
        processed = processed.resize((processed.width * scale, processed.height * scale), Image.NEAREST)
        processed = ImageOps.autocontrast(processed)
        bw = processed.point(lambda x: 0 if x < 180 else 255, mode="L")
        if np.mean(np.asarray(bw)) < 128:
            bw = ImageOps.invert(bw)
        bw = bw.filter(ImageFilter.MedianFilter(size=3))

        try:
            text = pytesseract.image_to_string(
                bw,
                config="--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789",
            )
        except pytesseract.pytesseract.TesseractError:
            return None, float(dark_ratio)

        digits = ''.join(ch for ch in text if ch.isdigit())
        if not digits:
            return None, float(dark_ratio)
        predicted = int(digits[0])
        confidence = float(min(1.0, max(dark_ratio * 2.5, 0.0)))
        return predicted, confidence

    def _shrink_bbox(
        self,
        bbox: Tuple[int, int, int, int],
        image_size: Tuple[int, int],
        *,
        shrink: int,
    ) -> Tuple[int, int, int, int]:
        left, top, right, bottom = bbox
        width, height = image_size
        left = max(0, left + shrink)
        top = max(0, top + shrink)
        right = min(width, right - shrink)
        bottom = min(height, bottom - shrink)
        if right <= left or bottom <= top:
            return bbox
        return left, top, right, bottom

    def _load_cell_bboxes(
        self,
        record: dict,
        image_size: Tuple[int, int],
        grid_size: int,
    ) -> List[List[Tuple[int, int, int, int]]]:
        bboxes_data = record.get("cell_bboxes")
        if bboxes_data:
            return [
                [tuple(map(int, bbox)) for bbox in row]
                for row in bboxes_data
            ]
        width, height = image_size
        cell_width = width // grid_size
        cell_height = height // grid_size
        return [
            [
                (
                    col * cell_width,
                    row * cell_height,
                    (col + 1) * cell_width,
                    (row + 1) * cell_height,
                )
                for col in range(grid_size)
            ]
            for row in range(grid_size)
        ]

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

    def _is_valid_solution(self, grid: List[List[int]]) -> bool:
        size = len(grid)
        if size == 0:
            return False
        digits = set(range(1, size + 1))
        for row in grid:
            row_set = set(row)
            if 0 in row_set or row_set != digits:
                return False
        for col in range(size):
            col_values = {grid[row][col] for row in range(size)}
            if 0 in col_values or col_values != digits:
                return False
        subgrid = int(size ** 0.5)
        if subgrid * subgrid != size:
            return False
        for start_row in range(0, size, subgrid):
            for start_col in range(0, size, subgrid):
                values = {
                    grid[r][c]
                    for r in range(start_row, start_row + subgrid)
                    for c in range(start_col, start_col + subgrid)
                }
                if 0 in values or values != digits:
                    return False
        return True


__all__ = ["SudokuEvaluator", "SudokuEvaluationResult", "CellEvaluation"]


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a Sudoku puzzle solution image")
    parser.add_argument("metadata", type=Path, help="Path to sudoku puzzles metadata JSON")
    parser.add_argument("puzzle_id", type=str, help="Identifier of the puzzle to evaluate")
    parser.add_argument("candidate", type=Path, help="Image containing the candidate solution")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=None,
        help="Base directory for metadata assets",
    )
    parser.add_argument(
        "--trim-tolerance",
        type=int,
        default=12,
        help="Pixel tolerance when trimming borders from the candidate image",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)
    evaluator = SudokuEvaluator(args.metadata, base_dir=args.base_dir)
    result = evaluator.evaluate(
        args.puzzle_id,
        args.candidate,
        trim_tolerance=args.trim_tolerance,
    )
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
