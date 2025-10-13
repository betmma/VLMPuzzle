"""Evaluator for arc connection puzzles.

Transcribes the attempt video in the output folder to detect the spoken option
letter (A–E) using scripts/transcribe_video.py, then compares with the
generated record's correct option.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import re

from ..base import AbstractPuzzleEvaluator, PathLike


@dataclass
class ArcConnectEvaluationResult:
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


class ArcConnectEvaluator(AbstractPuzzleEvaluator):
    """Transcribe the attempt's video and check the spoken option."""

    VIDEO_GLOBS = ("video_*.mp4", "video_*.webm", "video_*.mov", "*.mp4", "*.webm", "*.mov")

    def evaluate(
        self,
        puzzle_id: str,
        candidate_image: PathLike,
        *,
        engine: str = "local",
        model: str = "whisper-1",
        base_url: Optional[str] = None,
    ) -> ArcConnectEvaluationResult:
        record = self.get_record(puzzle_id)
        correct = str(record.get("correct_option", "")).strip().upper() or ""
        if correct not in ("A", "B", "C", "D", "E"):
            raise ValueError("Puzzle record missing valid 'correct_option' (A–E)")

        candidate_path = Path(candidate_image)
        attempt_dir = candidate_path.parent
        
        text_path = attempt_dir / "content.txt"
        if not text_path.exists() or not text_path.is_file():
            raise FileNotFoundError(f"Text not found: {text_path}")
        text_response = text_path.read_text(encoding="utf-8")

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
                cmd.extend(["--engine", "local"])  # whisper

            # On Windows, invoke the Python interpreter explicitly for .py scripts
            import sys as _sys
            py_cmd = [_sys.executable, cmd[0]] + cmd[1:]
            completed = subprocess.run(py_cmd, capture_output=True, text=True)
            if completed.returncode == 0:
                try:
                    out_path = Path(completed.stdout.strip().splitlines()[-1].strip())
                    if out_path.exists():
                        transcript_json_path = out_path
                        payload = json.loads(out_path.read_text(encoding="utf-8"))
                        nato_word = payload.get("first_nato_word")
                        if isinstance(nato_word, str) and nato_word:
                            predicted = nato_word.strip().upper()[0]
                except Exception:
                    pass
            else:
                cmd2: List[str] = [
                    str(Path.cwd() / "scripts" / "transcribe_video.py"),
                    video_path.as_posix(),
                    "--nato-only",
                ]
                import sys as _sys
                py_cmd2 = [_sys.executable, cmd2[0]] + cmd2[1:]
                completed2 = subprocess.run(py_cmd2, capture_output=True, text=True)
                if completed2.returncode == 0:
                    val = completed2.stdout.strip().upper()
                    if val:
                        predicted = val[0]
        else:
            options = re.findall(r'\b([A-E])\b', text_response.upper())
            predicted = options[-1] if options else None

        is_correct = (predicted == correct) if predicted else False
        return ArcConnectEvaluationResult(
            puzzle_id=puzzle_id,
            predicted_option=predicted,
            correct_option=correct,
            is_correct=is_correct,
            video_path=video_path.as_posix() if video_path else None,
            transcript_json_path=transcript_json_path.as_posix() if transcript_json_path else None,
        )


__all__ = ["ArcConnectEvaluator", "ArcConnectEvaluationResult"]


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate arc connection puzzles via video transcription")
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
    evaluator = ArcConnectEvaluator(args.metadata, base_dir=args.base_dir)
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
