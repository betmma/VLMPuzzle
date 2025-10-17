"""Evaluator for midpoint puzzles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from ..base import AbstractPuzzleEvaluator, PathLike
from ..point_target_base import PointTargetPuzzleEvaluator


class MidpointEvaluator(PointTargetPuzzleEvaluator):
    VIDEO_GLOBS = PointTargetPuzzleEvaluator.VIDEO_GLOBS

    def evaluate(
        self,
        puzzle_id: str,
        candidate_image: PathLike,
        *,
        video_sample_stride: int = 5,
    ) -> AbstractPuzzleEvaluator.OptionEvaluationResult:
        record = self.get_record(puzzle_id)
        correct = str(record.get("correct_option", "")).strip().upper()
        if not correct or len(correct) != 1:
            raise ValueError("Puzzle record missing valid 'correct_option' (single letter)")

        candidate_path = Path(candidate_image)
        attempt_dir = candidate_path.parent

        transcript_option = self.transcript_option_from_attempt(attempt_dir)
        text_option = self.text_option_from_attempt(attempt_dir)
        video_option = self.video_option_from_attempt(attempt_dir, record, video_sample_stride)
        image_option, red_pixel_count, red_centroid = self.image_option_from_path(candidate_path, record)

        result = AbstractPuzzleEvaluator.OptionEvaluationResult(
            puzzle_id=puzzle_id,
            correct_option=correct,
            transcribe_option=transcript_option,
            video_option=video_option,
            image_option=image_option,
            text_option=text_option,
            attempt_dir=attempt_dir.as_posix(),
        )
        result.red_pixel_count = red_pixel_count
        result.red_centroid = red_centroid
        return result


__all__ = ["MidpointEvaluator"]


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate midpoint puzzles")
    parser.add_argument("metadata", type=Path)
    parser.add_argument("puzzle_id", type=str)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--base-dir", type=Path, default=None)
    parser.add_argument("--video-stride", dest="video_sample_stride", type=int, default=5)
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = _parse_args(argv)
    evaluator = MidpointEvaluator(args.metadata, base_dir=args.base_dir)
    result = evaluator.evaluate(
        args.puzzle_id,
        args.candidate,
        video_sample_stride=args.video_sample_stride,
    )
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
