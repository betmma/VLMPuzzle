"""Evaluator for colored-rectangles stacking order puzzles."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from ..base import AbstractPuzzleEvaluator, PathLike


Color = Tuple[int, int, int]


@dataclass
class OrderPosition:
    index: int
    expected_color: Color
    predicted_color: Optional[Color]
    is_correct: bool

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "expected_color": list(self.expected_color),
            "predicted_color": (list(self.predicted_color) if self.predicted_color else None),
            "is_correct": self.is_correct,
        }


@dataclass
class RectsEvaluationResult:
    puzzle_id: str
    expected_order: List[Color]
    predicted_order: List[Optional[Color]]
    correct: int
    total: int
    order_breakdown: List[OrderPosition]

    def to_dict(self) -> dict:
        return {
            "puzzle_id": self.puzzle_id,
            "expected_order": [list(c) for c in self.expected_order],
            "predicted_order": [list(c) if c else None for c in self.predicted_order],
            "correct": self.correct,
            "total": self.total,
            "order_breakdown": [pos.to_dict() for pos in self.order_breakdown],
        }


class RectsEvaluator(AbstractPuzzleEvaluator):
    """Evaluate by extracting a top-to-bottom color order from the candidate image."""

    def evaluate(self, puzzle_id: str, candidate_image: PathLike) -> RectsEvaluationResult:
        record = self.get_record(puzzle_id)
        candidate_path = Path(candidate_image)
        if not candidate_path.exists():
            raise FileNotFoundError(f"Candidate image not found: {candidate_path}")

        rects = record.get("rectangles", [])
        # Order from top-most to bottom-most based on z
        rect_entries: List[Tuple[int, Color]] = []
        for r in rects:
            try:
                z = int(r.get("z"))
                color_list = r.get("color")
                if not (isinstance(color_list, (list, tuple)) and len(color_list) == 3):
                    continue
                color = (int(color_list[0]), int(color_list[1]), int(color_list[2]))
                rect_entries.append((z, color))
            except Exception:
                continue
        expected_order = [color for z, color in sorted(rect_entries, key=lambda t: t[0], reverse=True)]

        with Image.open(candidate_path) as img:
            cand = np.asarray(img.convert("RGB"))

        predicted_order = self._extract_order(cand, expected_order)

        correct = 0
        breakdown: List[OrderPosition] = []
        for idx, exp_color in enumerate(expected_order):
            pred_color = predicted_order[idx] if idx < len(predicted_order) else None
            is_correct = self._is_same_color(pred_color, exp_color)
            breakdown.append(
                OrderPosition(index=idx, expected_color=exp_color, predicted_color=pred_color, is_correct=is_correct)
            )
            if is_correct:
                correct += 1

        return RectsEvaluationResult(
            puzzle_id=puzzle_id,
            expected_order=expected_order,
            predicted_order=predicted_order,
            correct=correct,
            total=len(expected_order),
            order_breakdown=breakdown,
        )

    # --- Helpers --------------------------------------------------------------------

    @staticmethod
    def _is_same_color(pred: Optional[Color], target: Color, tol: float = 24.0) -> bool:
        if pred is None:
            return False
        diff = np.array(pred, dtype=np.float32) - np.array(target, dtype=np.float32)
        return float(np.linalg.norm(diff)) <= tol

    def _extract_order(self, arr: np.ndarray, palette: Sequence[Color]) -> List[Optional[Color]]:
        H, W = arr.shape[:2]
        # Downsample rows for speed
        step = max(1, H // (len(palette) * 6))
        # Build palette as numpy array
        pal = np.array(palette, dtype=np.float32)
        # Track which palette entries are still available to assign
        available = np.ones(len(palette), dtype=bool) if len(palette) else np.array([], dtype=bool)
        # Thresholds
        assign_margin: float = 48.0           # max distance to accept a row match
        used_pixel_margin: float = 48.0       # mask pixels near already-assigned colors

        row_assignments: List[int] = []  # index into palette for each sampled row
        for y in range(0, H, step):
            row = arr[y, :, :].astype(np.float32)
            # Exclude near-white and near-black pixels from row color estimation
            # Thresholds allow for minor compression noise around pure colors
            is_white = np.all(row >= 240.0, axis=1)
            is_black = np.all(row <= 15.0, axis=1)
            mask = ~(is_white | is_black)

            # Further exclude pixels whose color matches any already-assigned palette color
            if pal.size and np.any(~available):
                used_colors = pal[~available]
                # Compute distance of each pixel to the set of used colors
                diffs = row[:, None, :] - used_colors[None, :, :]
                dists_used = np.linalg.norm(diffs, axis=2)
                suppress = np.any(dists_used <= used_pixel_margin, axis=1)
                mask = mask & (~suppress)

            if np.any(mask):
                filtered = row[mask]
            else:
                # If the row is only white/black or already-used colors, skip it
                continue

            mean_rgb = filtered.mean(axis=0)
            if not pal.size or not np.any(available):
                break
            dists = np.linalg.norm(pal - mean_rgb[None, :], axis=1)
            # Exclude already seen colors by setting their distance to infinity
            dists = np.where(available, dists, np.inf)
            nearest = int(np.argmin(dists)) if np.any(available) else -1
            if nearest >= 0:
                min_dist = float(dists[nearest])
                if np.isfinite(min_dist) and min_dist <= assign_margin:
                    row_assignments.append(nearest)
                    available[nearest] = False
                    # Stop early if we've assigned all colors
                    if not np.any(available):
                        break
                # Otherwise, distance too large: skip this row without assignment

        # Collapse consecutive duplicates and map to colors, keep unique sequentially
        order_idx: List[int] = []
        last = None
        for idx in row_assignments:
            if idx < 0:
                continue
            if last is None or idx != last:
                order_idx.append(idx)
                last = idx
        # Deduplicate while preserving first occurrence order, limited to palette size
        seen = set()
        dedup_idx: List[int] = []
        for idx in order_idx:
            if idx not in seen:
                dedup_idx.append(idx)
                seen.add(idx)
            if len(dedup_idx) >= len(palette):
                break
        return [palette[i] if 0 <= i < len(palette) else None for i in dedup_idx]


__all__ = ["RectsEvaluator", "RectsEvaluationResult", "OrderPosition"]


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate rectangles-order puzzles")
    parser.add_argument("metadata", type=Path)
    parser.add_argument("puzzle_id", type=str)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--base-dir", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)
    evaluator = RectsEvaluator(args.metadata, base_dir=args.base_dir)
    result = evaluator.evaluate(args.puzzle_id, args.candidate)
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
