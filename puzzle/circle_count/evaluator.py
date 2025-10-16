"""Evaluator for circle count puzzles.

Determines whether the attempt reported the correct number of circles by
transcribing an attempt video when present, otherwise by reading content.txt.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..base import AbstractPuzzleEvaluator, PathLike


@dataclass
class CircleCountEvaluationResult:
    puzzle_id: str
    predicted_option: Optional[int]
    correct_option: int
    is_correct: bool
    video_path: Optional[str]
    transcript_json_path: Optional[str]
    transcript_text: Optional[str]

    def to_dict(self) -> Dict[str, Optional[object]]:
        return {
            "puzzle_id": self.puzzle_id,
            "predicted_option": self.predicted_option,
            "correct_option": self.correct_option,
            "is_correct": self.is_correct,
            "video_path": self.video_path,
            "transcript_json_path": self.transcript_json_path,
            "transcript_text": self.transcript_text,
        }


class CircleCountEvaluator(AbstractPuzzleEvaluator):
    VIDEO_GLOBS = ("video_*.mp4", "video_*.webm", "video_*.mov", "*.mp4", "*.webm", "*.mov")
    NUMBER_WORDS: Dict[str, int] = {
        "zero": 0,
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
        "eleven": 11,
        "twelve": 12,
        "thirteen": 13,
        "fourteen": 14,
        "fifteen": 15
    }

    def evaluate(
        self,
        puzzle_id: str,
        candidate_image: PathLike,
        *,
        engine: str = "local",
        model: str = "whisper-1",
        base_url: Optional[str] = None,
        whisper_model: str = "base",
    ) -> CircleCountEvaluationResult:
        record = self.get_record(puzzle_id)
        circle_count = int(record.get("circle_count", 0))
        if circle_count <= 0:
            raise ValueError("Puzzle record missing positive 'circle_count'")

        candidate_path = Path(candidate_image)
        attempt_dir = candidate_path.parent

        video_path = self._find_video(attempt_dir)
        transcript_json_path: Optional[Path] = None
        transcript_text: Optional[str] = None
        predicted: Optional[int] = None

        if video_path is not None:
            transcript_json_path, transcript_text = self._transcribe_video(
                video_path,
                attempt_dir,
                engine=engine,
                model=model,
                base_url=base_url,
                whisper_model=whisper_model,
            )
            if transcript_text:
                predicted = self._extract_number(transcript_text)
        else:
            text_path = attempt_dir / "content.txt"
            if not text_path.exists():
                raise FileNotFoundError(f"Text not found: {text_path}")
            transcript_text = text_path.read_text(encoding="utf-8")
            predicted = self._extract_number(transcript_text)

        is_correct = predicted == circle_count

        return CircleCountEvaluationResult(
            puzzle_id=puzzle_id,
            predicted_option=predicted,
            correct_option=circle_count,
            is_correct=is_correct,
            video_path=video_path.as_posix() if video_path is not None else None,
            transcript_json_path=transcript_json_path.as_posix() if transcript_json_path is not None else None,
            transcript_text=transcript_text,
        )

    def _find_video(self, attempt_dir: Path) -> Optional[Path]:
        for pattern in self.VIDEO_GLOBS:
            candidates = list(attempt_dir.glob(pattern))
            for path in candidates:
                if path.is_file():
                    return path
        return None

    def _transcribe_video(
        self,
        video_path: Path,
        attempt_dir: Path,
        *,
        engine: str,
        model: str,
        base_url: Optional[str],
        whisper_model: str,
    ) -> Tuple[Optional[Path], Optional[str]]:
        script_path = Path.cwd() / "scripts" / "transcribe_video.py"
        json_out = attempt_dir / "transcription.json"
        cmd: List[str] = [
            sys.executable,
            script_path.as_posix(),
            video_path.as_posix(),
            "--output-json",
            json_out.as_posix(),
        ]
        if engine == "api":
            cmd.extend(["--engine", "api", "--model", model])
            if base_url:
                cmd.extend(["--base-url", base_url])
        else:
            cmd.extend(["--engine", "local", "--whisper-model", whisper_model])
        completed = subprocess.run(cmd, capture_output=True, text=True)
        if completed.returncode != 0:
            return None, None
        if not json_out.exists():
            lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
            if lines:
                last = Path(lines[-1])
                if last.exists():
                    json_out = last
        if json_out.exists():
            payload = json.loads(json_out.read_text(encoding="utf-8"))
            transcript = payload.get("transcript")
            if isinstance(transcript, str):
                return json_out, transcript
            alt = payload.get("text")
            if isinstance(alt, str):
                return json_out, alt
        return json_out if json_out.exists() else None, None

    def _extract_number(self, text: str) -> Optional[int]:
        if '**' in text:
            text = ' '.join(re.findall(r'\*\*(.*?)\*\*', text))
        digits = re.findall(r"\b\d+\b", text)
        if digits:
            return int(digits[-1])
        words = re.findall(r"[a-zA-Z]+", text.lower())
        for word in reversed(words):
            if word in self.NUMBER_WORDS:
                return self.NUMBER_WORDS[word]
        return None


__all__ = [
    "CircleCountEvaluator",
    "CircleCountEvaluationResult",
]


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate circle count puzzles")
    parser.add_argument("metadata", type=Path)
    parser.add_argument("puzzle_id", type=str)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--base-dir", type=Path, default=None)
    parser.add_argument("--engine", choices=["local", "api"], default="local")
    parser.add_argument("--model", type=str, default="whisper-1")
    parser.add_argument("--base-url", dest="base_url", type=str, default=None)
    parser.add_argument("--whisper-model", dest="whisper_model", type=str, default="base")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)
    evaluator = CircleCountEvaluator(args.metadata, base_dir=args.base_dir)
    result = evaluator.evaluate(
        args.puzzle_id,
        args.candidate,
        engine=args.engine,
        model=args.model,
        base_url=args.base_url,
        whisper_model=args.whisper_model,
    )
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
