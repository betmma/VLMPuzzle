"""Sudoku puzzle evaluator implementation using OCR."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import cv2
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
    is_correct: bool
    is_clue: bool

    def to_dict(self) -> dict:
        return {
            "row": self.row,
            "col": self.col,
            "expected": self.expected,
            "predicted": self.predicted,
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
        debug_dir: Optional[PathLike] = None,
    ) -> SudokuEvaluationResult:
        record = self.get_record(puzzle_id)

        solution_path = self.resolve_path(record["solution_image_path"])
        candidate_path = Path(candidate_image)
        if not solution_path.exists():
            raise FileNotFoundError(f"Solution image missing: {solution_path}")
        if not candidate_path.exists():
            raise FileNotFoundError(f"Candidate image missing: {candidate_path}")

        with Image.open(solution_path) as solution_image_obj:
            solution_size = solution_image_obj.size
        candidate_image_obj = Image.open(candidate_path).convert("RGB")

        solution_grid = [[int(value) for value in row] for row in record["solution_grid"]]
        puzzle_grid = [[int(value) for value in row] for row in record["puzzle_grid"]]
        grid_size = len(solution_grid)
        cell_bboxes = self.map_cell_bboxes_to_image(
            record,
            target_size=candidate_image_obj.size,
            reference_size=solution_size,
        )

        debug_path: Optional[Path]
        debug_records: Optional[List[dict]]
        if debug_dir is not None:
            debug_path = Path(debug_dir)
            debug_path.mkdir(parents=True, exist_ok=True)
            (debug_path / "cells").mkdir(parents=True, exist_ok=True)
            debug_records = []
        else:
            debug_path = None
            debug_records = None

        predicted_grid: List[List[int]] = [[0 for _ in range(grid_size)] for _ in range(grid_size)]
        breakdown: List[CellEvaluation] = []
        correct = 0

        for row_idx, row in enumerate(solution_grid):
            for col_idx, expected in enumerate(row):
                bbox = cell_bboxes[row_idx][col_idx]
                tile = candidate_image_obj.crop(bbox)
                predicted = self._extract_digit(
                    tile,
                    debug_dir=debug_path,
                    cell_label=f"r{row_idx:02d}_c{col_idx:02d}",
                    debug_records=debug_records,
                    expected=expected,
                )
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
                        is_correct=is_correct,
                        is_clue=puzzle_grid[row_idx][col_idx] != 0,
                    )
                )

        total_cells = grid_size * grid_size if grid_size else 0
        accuracy = correct / total_cells if total_cells else 0.0
        is_valid = self._is_valid_solution(predicted_grid)

        if debug_records is not None and debug_path is not None:
            debug_log_path = debug_path / "ocr_debug.json"
            debug_log_path.write_text(json.dumps(debug_records, indent=2), encoding="utf-8")

        return SudokuEvaluationResult(
            puzzle_id=puzzle_id,
            correct_cells=correct,
            total_cells=total_cells,
            accuracy=accuracy,
            is_valid_solution=is_valid,
            cell_breakdown=breakdown,
        )
    # ------------------------------------------------------------------

    def _extract_digit(
        self,
        tile: Image.Image,
        *,
        debug_dir: Optional[Path] = None,
        cell_label: Optional[str] = None,
        debug_records: Optional[List[dict]] = None,
        expected: Optional[int] = None,
    ) -> Optional[int]:
        gray = tile.convert("L")
        arr = np.asarray(gray, dtype=np.uint8)
        dark_ratio = float((arr < 215).sum()) / arr.size
        debug_entry: Optional[dict] = None
        raw_path: Optional[str] = None
        processed_path: Optional[str] = None
        if debug_dir is not None and cell_label is not None:
            cell_dir = debug_dir / "cells"
            cell_dir.mkdir(parents=True, exist_ok=True)
            raw_file = cell_dir / f"{cell_label}_raw.png"
            tile.save(raw_file)
            raw_path = raw_file.as_posix()
        if debug_records is not None:
            debug_entry = {
                "cell": cell_label,
                "expected": expected,
                "dark_ratio": float(dark_ratio),
                "path_raw": raw_path,
            }
        if dark_ratio < 0.008:
            if debug_entry is not None:
                debug_entry.update(
                    {
                        "reason": "low_dark_ratio",
                        "ocr_text": None,
                        "predicted": None,
                        "path_processed": None,
                    }
                )
                debug_records.append(debug_entry)
            return None

        scale = 4
        resized = cv2.resize(
            arr,
            (arr.shape[1] * scale, arr.shape[0] * scale),
            interpolation=cv2.INTER_CUBIC,
        )
        blurred = cv2.GaussianBlur(resized, (3, 3), 0)
        _, thresh = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        if thresh.mean() < 128:
            thresh = cv2.bitwise_not(thresh)

        kernel = np.ones((2, 2), np.uint8)
        digit_mask = cv2.bitwise_not(thresh)
        digit_mask = cv2.morphologyEx(digit_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        digit_mask = cv2.dilate(digit_mask, kernel, iterations=1)
        thresh = cv2.bitwise_not(digit_mask)

        coords = np.column_stack(np.where(thresh < 200))
        if coords.size:
            y0, x0 = coords.min(axis=0)
            y1, x1 = coords.max(axis=0) + 1
            pad = 6
            y0 = max(y0 - pad, 0)
            x0 = max(x0 - pad, 0)
            y1 = min(y1 + pad, thresh.shape[0])
            x1 = min(x1 + pad, thresh.shape[1])
            cropped = thresh[y0:y1, x0:x1]
            thresh = cv2.copyMakeBorder(
                cropped,
                4,
                4,
                4,
                4,
                borderType=cv2.BORDER_CONSTANT,
                value=255,
            )

        bw = Image.fromarray(thresh)
        bw = ImageOps.autocontrast(bw)

        processed_paths = {"normal": processed_path}
        processed_variants = [("normal", bw), ("invert", ImageOps.invert(bw))]
        ocr_configs = [
            (
                "psm8",
                "--psm 8 --oem 3 -c tessedit_char_whitelist=0123456789 "
                "-c classify_bln_numeric_mode=1 -c tessedit_single_char=1",
            ),
            (
                "psm10",
                "--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789 "
                "-c classify_bln_numeric_mode=1 -c tessedit_single_char=1",
            ),
        ]

        if debug_dir is not None and cell_label is not None:
            bw_file = (debug_dir / "cells") / f"{cell_label}_ocr.png"
            bw.save(bw_file)
            processed_paths["normal"] = bw_file.as_posix()
            processed_path = processed_paths["normal"]

        attempts = []
        predicted: Optional[int] = None
        chosen_text: Optional[str] = None
        chosen_variant: Optional[str] = None
        chosen_config: Optional[str] = None
        final_processed_path = processed_paths.get("normal")

        for variant_name, image_variant in processed_variants:
            if variant_name not in processed_paths:
                if debug_dir is not None and cell_label is not None:
                    variant_file = (debug_dir / "cells") / f"{cell_label}_ocr_{variant_name}.png"
                    image_variant.save(variant_file)
                    processed_paths[variant_name] = variant_file.as_posix()
                else:
                    processed_paths[variant_name] = processed_paths.get("normal")
            last_processed_path = processed_paths.get(variant_name, processed_paths.get("normal"))
            final_processed_path = last_processed_path
            for config_name, config_value in ocr_configs:
                try:
                    text = pytesseract.image_to_string(image_variant, config=config_value)
                except pytesseract.pytesseract.TesseractError as exc:
                    attempts.append(
                        {
                            "variant": variant_name,
                            "config": config_name,
                            "error": str(exc),
                        }
                    )
                    continue
                digits = "".join(ch for ch in text if ch.isdigit())
                if digits:
                    predicted = int(digits[0])
                    chosen_text = text
                    chosen_variant = variant_name
                    chosen_config = config_name
                    final_processed_path = last_processed_path
                    break
                attempts.append(
                    {
                        "variant": variant_name,
                        "config": config_name,
                        "ocr_text": text,
                    }
                )
            if predicted is not None:
                break

        if predicted is None:
            if debug_entry is not None:
                debug_entry.update(
                    {
                        "reason": "no_digits",
                        "ocr_text": None,
                        "predicted": None,
                        "path_processed": final_processed_path,
                        "attempts": attempts,
                    }
                )
                debug_records.append(debug_entry)
            return None

        if debug_entry is not None:
            debug_entry.update(
                {
                    "reason": "ocr",
                    "ocr_text": chosen_text,
                    "predicted": predicted,
                    "path_processed": final_processed_path,
                }
            )
            if chosen_variant and chosen_variant != "normal":
                debug_entry["fallback"] = chosen_variant
            if chosen_config is not None:
                debug_entry["ocr_config"] = chosen_config
            if attempts:
                debug_entry["prior_attempts"] = attempts
            debug_records.append(debug_entry)
        return predicted

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
    parser.add_argument(
        "--debug-dir",
        type=Path,
        default=None,
        help="Optional directory to dump OCR debug crops",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)
    evaluator = SudokuEvaluator(args.metadata, base_dir=args.base_dir)
    result = evaluator.evaluate(
        args.puzzle_id,
        args.candidate,
        trim_tolerance=args.trim_tolerance,
        debug_dir=args.debug_dir,
    )
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()

