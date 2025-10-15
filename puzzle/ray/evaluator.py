"""Evaluator for ray-and-mirrors puzzles.

Given a generated video output folder (from scripts/veo3.py), mirrorVote.py
passes the path to its extracted last frame (result.png). This evaluator locates
the corresponding video file(s) in that same folder, runs scripts/transcribe_video.py
to extract the first NATO code word as an option letter (A–E), and compares it
against the recorded correct option in puzzle metadata.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from ..base import AbstractPuzzleEvaluator, PathLike


@dataclass
class RayEvaluationResult:
    puzzle_id: str
    predicted_option: Optional[str]
    correct_option: str
    is_correct: bool
    video_path: Optional[str]
    transcript_json_path: Optional[str]

    def to_dict(self) -> dict:
        return {
            "puzzle_id": self.puzzle_id,
            "predicted_option": self.predicted_option,
            "correct_option": self.correct_option,
            "is_correct": self.is_correct,
            "video_path": self.video_path,
            "transcript_json_path": self.transcript_json_path,
        }


class RayEvaluator(AbstractPuzzleEvaluator):
    """Transcribe the attempt's video and check the spoken option."""

    VIDEO_GLOBS = ("video_*.mp4", "video_*.webm", "video_*.mov", "*.mp4", "*.webm", "*.mov")

    def evaluate(
        self,
        puzzle_id: str,
        candidate_image: PathLike,
        *,
        engine: str = "local",  # or "api"
        model: str = "whisper-1",  # used if engine==api
        base_url: Optional[str] = None,  # used if engine==api
    ) -> RayEvaluationResult:
        record = self.get_record(puzzle_id)
        correct = str(record.get("correct_option", "")).strip().upper() or ""
        if correct not in ("A", "B", "C", "D", "E"):
            raise ValueError("Puzzle record missing valid 'correct_option' (A–E)")

        candidate_path = Path(candidate_image)
        if not candidate_path.exists():
            raise FileNotFoundError(f"Candidate image not found: {candidate_path}")
        attempt_dir = candidate_path.parent

        # Find a video file in the same folder
        video_path: Optional[Path] = None
        for pattern in self.VIDEO_GLOBS:
            for p in attempt_dir.glob(pattern):
                if p.is_file():
                    video_path = p
                    break
            if video_path is not None:
                break

        predicted: Optional[str] = None
        transcript_json_path: Optional[Path] = None

        if video_path is not None:
            # Run transcriber
            json_out = attempt_dir / "transcription.json"
            cmd: List[str] = [
                str(Path.cwd() / "scripts" / "transcribe_video.py"),
                video_path.as_posix(),
                "--output-json",
                json_out.as_posix(),
            ]
            if engine == "api":
                cmd.extend(["--engine", "api", "--model", model])
                if base_url:
                    cmd.extend(["--base-url", base_url])
            else:
                cmd.extend(["--engine", "local"])  # default whisper

            completed = subprocess.run([str(Path().resolve() / cmd[0])] + cmd[1:], capture_output=True, text=True)
            # Best-effort parse: transcriber prints JSON path on success when --output-json is provided
            out_path = Path(completed.stdout.strip().splitlines()[-1].strip())
            if out_path.exists():
                transcript_json_path = out_path
                payload = json.loads(out_path.read_text(encoding="utf-8"))
                nato_letter = payload.get("first_nato_word")
                if isinstance(nato_letter, str) and nato_letter:
                    predicted = nato_letter.strip().upper()[0]

        is_correct = (predicted == correct) if predicted else False
        return RayEvaluationResult(
            puzzle_id=puzzle_id,
            predicted_option=predicted,
            correct_option=correct,
            is_correct=is_correct,
            video_path=video_path.as_posix() if video_path else None,
            transcript_json_path=transcript_json_path.as_posix() if transcript_json_path else None,
        )


__all__ = ["RayEvaluator", "RayEvaluationResult"]


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ray puzzles via video transcription")
    parser.add_argument("metadata", type=Path)
    parser.add_argument("puzzle_id", type=str)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--base-dir", type=Path, default=None)
    parser.add_argument("--engine", choices=["local", "api"], default="local")
    parser.add_argument("--model", type=str, default="whisper-1")
    parser.add_argument("--base-url", dest="base_url", type=str, default=None)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)
    evaluator = RayEvaluator(args.metadata, base_dir=args.base_dir)
    result = evaluator.evaluate(
        args.puzzle_id,
        args.candidate,
        engine=args.engine,
        model=args.model,
        base_url=args.base_url,
    )
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()

