"""Evaluator for ray intersection puzzles.

The evaluator performs two checks:
1. Transcribe the attempt video to capture the spoken option letter.
2. Use the provided candidate image (expected to be the last frame) to detect
    the reddened intersection and compare it against the recorded ground truth.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..base import AbstractPuzzleEvaluator, PathLike


class RayIntersectionEvaluator(AbstractPuzzleEvaluator):
    VIDEO_GLOBS = ("video_*.mp4", "video_*.webm", "video_*.mov", "*.mp4", "*.webm", "*.mov")

    def evaluate(
        self,
        puzzle_id: str,
        candidate_image: PathLike,
        *,
        min_red_ratio: float = 0.12,
    ) -> AbstractPuzzleEvaluator.OptionEvaluationResult:
        record = self.get_record(puzzle_id)
        correct = str(record.get("correct_option", "")).strip().upper()
        if correct not in ("A", "B", "C", "D", "E"):
            raise ValueError("Puzzle record missing valid 'correct_option' (A–E)")

        candidate_path = Path(candidate_image)
        attempt_dir = candidate_path.parent
        
        transcript_result = self.transcribe_video(attempt_dir)
        transcript_option = transcript_result.get("first_nato_word")
        text_path = attempt_dir / "content.txt"
        if not text_path.exists() or not text_path.is_file():
            raise FileNotFoundError(f"Text not found: {text_path}")
        text_text = text_path.read_text(encoding="utf-8")
        text_result = self.extract_first_nato_word(text_text)

        red_scores: Dict[str, float] = {}
        loaded_frame = cv2.imread(candidate_path.as_posix(), cv2.IMREAD_COLOR)
        if loaded_frame is not None:
            rgb_frame = cv2.cvtColor(loaded_frame, cv2.COLOR_BGR2RGB)
            red_option, red_scores = self._score_red_hits(rgb_frame, record)
        else:
            red_option = None
            red_scores = {}
        red_score_value = red_scores.get(correct, 0.0)
        red_is_correct = red_score_value >= min_red_ratio if red_scores else False

        return AbstractPuzzleEvaluator.OptionEvaluationResult(
            puzzle_id=puzzle_id,
            correct_option=correct,
            transcribe_option=transcript_option,
            image_option=red_option,
            text_option=text_result,
            attempt_dir=attempt_dir.as_posix(),
        )

    def _score_red_hits(self, frame: np.ndarray, record: Dict[str, object]) -> Tuple[Optional[str], Dict[str, float]]:
        height = frame.shape[0]
        width = frame.shape[1]
        if height <= 0 or width <= 0:
            return None, {}
        canvas_dims = record.get("canvas_dimensions")
        if not isinstance(canvas_dims, Sequence) or len(canvas_dims) < 2:
            return None, {}
        canvas_width = float(canvas_dims[0])
        canvas_height = float(canvas_dims[1])
        if canvas_width <= 0 or canvas_height <= 0:
            return None, {}
        scale_x = width / canvas_width
        scale_y = height / canvas_height
        point_radius = float(record.get("point_radius", 20))
        search_radius = point_radius * 1.6
        scaled_radius = int(round(search_radius * 0.5 * (scale_x + scale_y)))
        if scaled_radius < 3:
            scaled_radius = 3

        red_mask = self._red_mask(frame)
        candidates_raw = record.get("candidates")
        scores: Dict[str, float] = {}
        if isinstance(candidates_raw, Sequence):
            for entry in candidates_raw:
                if not isinstance(entry, dict):
                    continue
                label = entry.get("label")
                if not isinstance(label, str) or len(label) != 1:
                    continue
                cx_raw = float(entry.get("x", 0.0))
                cy_raw = float(entry.get("y", 0.0))
                cx = int(round(cx_raw * scale_x))
                cy = int(round(cy_raw * scale_y))
                x0 = max(0, cx - scaled_radius)
                x1 = min(width, cx + scaled_radius)
                y0 = max(0, cy - scaled_radius)
                y1 = min(height, cy + scaled_radius)
                if x1 <= x0 or y1 <= y0:
                    scores[label] = 0.0
                    continue
                region = red_mask[y0:y1, x0:x1]
                if region.size == 0:
                    scores[label] = 0.0
                    continue
                ratio = float(region.mean())
                scores[label] = ratio
        predicted_label: Optional[str] = None
        if scores:
            sorted_items = sorted(scores.items(), key=lambda item: item[1], reverse=True)
            best_label, best_score = sorted_items[0]
            if best_score > 0.0:
                predicted_label = best_label
        return predicted_label, scores

    def _red_mask(self, frame: np.ndarray) -> np.ndarray:
        red = frame[:, :, 0].astype(np.float32)
        green = frame[:, :, 1].astype(np.float32)
        blue = frame[:, :, 2].astype(np.float32)
        dominance = red - np.maximum(green, blue)
        mask = (red >= 140.0) & (dominance >= 40.0) & (green <= 130.0) & (blue <= 130.0)
        return mask.astype(np.float32)


__all__ = ["RayIntersectionEvaluator", "RayIntersectionEvaluationResult"]


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ray intersection puzzles")
    parser.add_argument("metadata", type=Path)
    parser.add_argument("puzzle_id", type=str)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--base-dir", type=Path, default=None)
    parser.add_argument("--min-red-ratio", dest="min_red_ratio", type=float, default=0.12)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)
    evaluator = RayIntersectionEvaluator(args.metadata, base_dir=args.base_dir)
    result = evaluator.evaluate(
        args.puzzle_id,
        args.candidate,
        min_red_ratio=args.min_red_ratio,
    )
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
