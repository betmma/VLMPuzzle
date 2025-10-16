"""Evaluator for ray intersection puzzles.

The evaluator performs two checks:
1. Transcribe the attempt video to capture the spoken option letter.
2. Use the provided candidate image (expected to be the last frame) to detect
    the reddened intersection and compare it against the recorded ground truth.
"""

from __future__ import annotations

import argparse
import json
import math
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
        video_sample_stride: int = 5,
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

        video_option = self._video_majority_option(attempt_dir, record, video_sample_stride)

        red_option: Optional[str] = None
        red_pixel_count = 0
        red_centroid: Optional[Tuple[float, float]] = None
        loaded_frame = cv2.imread(candidate_path.as_posix(), cv2.IMREAD_COLOR)
        if loaded_frame is not None:
            rgb_frame = cv2.cvtColor(loaded_frame, cv2.COLOR_BGR2RGB)
            red_option, red_pixel_count, red_centroid = self._score_red_hits(rgb_frame, record)

        result = AbstractPuzzleEvaluator.OptionEvaluationResult(
            puzzle_id=puzzle_id,
            correct_option=correct,
            transcribe_option=transcript_option,
            video_option=video_option,
            image_option=red_option,
            text_option=text_result,
            attempt_dir=attempt_dir.as_posix(),
        )
        result.red_pixel_count = red_pixel_count
        result.red_centroid = red_centroid
        result.min_red_ratio = min_red_ratio
        return result

    def _score_red_hits(
        self,
        frame: np.ndarray,
        record: Dict[str, object],
    ) -> Tuple[Optional[str], int, Optional[Tuple[float, float]]]:
        height = frame.shape[0]
        width = frame.shape[1]
        if height <= 0 or width <= 0:
            return None, 0, None
        canvas_dims = record.get("canvas_dimensions")
        if not isinstance(canvas_dims, Sequence) or len(canvas_dims) < 2:
            return None, 0, None
        canvas_width = float(canvas_dims[0])
        canvas_height = float(canvas_dims[1])
        if canvas_width <= 0 or canvas_height <= 0:
            return None, 0, None
        scale_x = width / canvas_width
        scale_y = height / canvas_height
        red_mask = self._red_mask(frame)
        red_pixels = np.column_stack(np.nonzero(red_mask > 0.5))
        red_count = int(red_pixels.shape[0])
        if red_count < 20:
            return None, red_count, None

        mean_y = float(red_pixels[:, 0].mean())
        mean_x = float(red_pixels[:, 1].mean())
        red_point = (mean_x, mean_y)

        candidates_raw = record.get("candidates")
        best_label: Optional[str] = None
        best_distance: Optional[float] = None
        if isinstance(candidates_raw, Sequence):
            for entry in candidates_raw:
                if not isinstance(entry, dict):
                    continue
                label = entry.get("label")
                if not isinstance(label, str) or len(label) != 1:
                    continue
                cx_raw = entry.get("x")
                cy_raw = entry.get("y")
                if not isinstance(cx_raw, (int, float)) or not isinstance(cy_raw, (int, float)):
                    continue
                cx = float(cx_raw) * scale_x
                cy = float(cy_raw) * scale_y
                distance = math.hypot(cx - mean_x, cy - mean_y)
                if best_distance is None or distance < best_distance:
                    best_distance = distance
                    best_label = label

        return best_label, red_count, red_point

    def _iter_video_files(self, attempt_dir: Path) -> List[Path]:
        seen = set()
        videos: List[Path] = []
        for pattern in self.VIDEO_GLOBS:
            for candidate in attempt_dir.glob(pattern):
                if not candidate.is_file():
                    continue
                if candidate in seen:
                    continue
                seen.add(candidate)
                videos.append(candidate)
        videos.sort(key=lambda path: path.name)
        return videos

    def _video_majority_option(
        self,
        attempt_dir: Path,
        record: Dict[str, object],
        sample_stride: int,
    ) -> Optional[str]:
        stride = int(sample_stride) if sample_stride > 0 else 1
        counts: Dict[str, int] = {}
        for video_path in self._iter_video_files(attempt_dir):
            capture = cv2.VideoCapture(video_path.as_posix())
            if not capture.isOpened():
                capture.release()
                continue
            frame_index = 0
            while True:
                success, frame = capture.read()
                if not success:
                    break
                if frame_index % stride == 0:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    label, _, _ = self._score_red_hits(rgb_frame, record)
                    if label:
                        key = label.upper()
                        counts[key] = counts.get(key, 0) + 1
                frame_index += 1
            capture.release()
        if not counts:
            return None
        best_count = max(counts.values())
        best_labels = [label for label, count in counts.items() if count == best_count]
        best_labels.sort()
        return best_labels[0]

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
