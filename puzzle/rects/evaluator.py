"""Evaluator for colored-rectangles stacking order puzzles."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from ..base import AbstractPuzzleEvaluator, PathLike


Color = Tuple[int, int, int]


CANONICAL_COLOR_TUPLES: Dict[str, Color] = {
    "blue": (68, 229, 229),
    "green": (149, 229, 68),
    "purple": (149, 68, 229),
    "red": (229, 68, 68),
}

CANONICAL_COLOR_VECTORS: Dict[str, np.ndarray] = {
    name: np.array(rgb, dtype=np.float32) for name, rgb in CANONICAL_COLOR_TUPLES.items()
}

COLOR_SYNONYMS: Dict[str, str] = {
    "blue": "blue",
    "cyan": "blue",
    "teal": "blue",
    "turquoise": "blue",
    "green": "green",
    "lime": "green",
    "emerald": "green",
    "purple": "purple",
    "violet": "purple",
    "magenta": "purple",
    "red": "red",
    "crimson": "red",
    "scarlet": "red",
}

VIDEO_GLOBS: Tuple[str, ...] = ("video_*.mp4", "video_*.webm", "video_*.mov", "*.mp4", "*.webm", "*.mov", "*.mkv")
AUDIO_GLOBS: Tuple[str, ...] = ("*.wav", "*.mp3", "*.m4a", "*.aac", "*.flac", "*.ogg")
SPEECH_TRANSCRIPT_FILENAME = "speech_transcription.json"


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
    expected_names: List[Optional[str]]
    predicted_names: List[Optional[str]]
    spoken_transcript: Optional[str]
    spoken_color_names: List[str]
    spoken_color_rgb: List[Color]
    speech_source: Optional[str]
    speech_engine: Optional[str]
    speech_transcript_path: Optional[str]
    spoken_correct: int
    spoken_total: int

    def to_dict(self) -> dict:
        return {
            "puzzle_id": self.puzzle_id,
            "expected_order": [list(c) for c in self.expected_order],
            "predicted_order": [list(c) if c else None for c in self.predicted_order],
            "correct": self.correct,
            "total": self.total,
            "order_breakdown": [pos.to_dict() for pos in self.order_breakdown],
            "expected_names": self.expected_names,
            "predicted_names": self.predicted_names,
            "spoken_transcript": self.spoken_transcript,
            "spoken_color_names": self.spoken_color_names,
            "spoken_color_rgb": [list(c) for c in self.spoken_color_rgb],
            "speech_source": self.speech_source,
            "speech_engine": self.speech_engine,
            "speech_transcript_path": self.speech_transcript_path,
            "spoken_correct": self.spoken_correct,
            "spoken_total": self.spoken_total,
        }


class RectsEvaluator(AbstractPuzzleEvaluator):
    """Evaluate by extracting a top-to-bottom color order from the candidate image.

    Also supports transcribing spoken answers from co-located audio or video files
    to capture the verbalized color order (blue, green, purple, red).
    """

    def evaluate(
        self,
        puzzle_id: str,
        candidate_image: PathLike,
        *,
        speech_media: Optional[PathLike] = None,
        speech_mode: str = "auto",
        speech_engine: str = "local",
        speech_model: str = "whisper-1",
        speech_base_url: Optional[str] = None,
        whisper_model: str = "base",
        speech_language: Optional[str] = "en",
    ) -> RectsEvaluationResult:
        record = self.get_record(puzzle_id)
        candidate_path = Path(candidate_image)
        if not candidate_path.exists():
            raise FileNotFoundError(f"Candidate image not found: {candidate_path}")
        attempt_dir = candidate_path.parent

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

        expected_names: List[Optional[str]] = [self._color_to_name(color) for color in expected_order]
        predicted_names: List[Optional[str]] = [self._color_to_name(color) if color else None for color in predicted_order]

        speech_media_path = Path(speech_media) if speech_media is not None else None
        speech_source_path = self._resolve_speech_source(attempt_dir, speech_media_path, mode=speech_mode)
        speech_transcript: Optional[str] = None
        speech_engine_used: Optional[str] = None
        speech_json_path: Optional[str] = None
        spoken_names: List[str] = []
        spoken_rgb: List[Color] = []
        if speech_source_path is not None:
            transcription = self._transcribe_media(
                speech_source_path,
                attempt_dir=attempt_dir,
                engine=speech_engine,
                model=speech_model,
                base_url=speech_base_url,
                whisper_model=whisper_model,
                language=speech_language,
            )
            if transcription is not None:
                speech_transcript, spoken_names, spoken_rgb, transcript_path = transcription
                speech_engine_used = speech_engine
                speech_json_path = transcript_path.as_posix()

        expected_valid_sequence = [name for name in expected_names if name is not None]
        spoken_correct = 0
        if spoken_names and expected_valid_sequence:
            limit = min(len(expected_valid_sequence), len(spoken_names))
            for idx in range(limit):
                if spoken_names[idx] == expected_valid_sequence[idx]:
                    spoken_correct += 1
        spoken_total = len(expected_valid_sequence)

        return RectsEvaluationResult(
            puzzle_id=puzzle_id,
            expected_order=expected_order,
            predicted_order=predicted_order,
            correct=correct,
            total=len(expected_order),
            order_breakdown=breakdown,
            expected_names=expected_names,
            predicted_names=predicted_names,
            spoken_transcript=speech_transcript,
            spoken_color_names=spoken_names,
            spoken_color_rgb=spoken_rgb,
            speech_source=speech_source_path.as_posix() if speech_source_path is not None else None,
            speech_engine=speech_engine_used,
            speech_transcript_path=speech_json_path,
            spoken_correct=spoken_correct,
            spoken_total=spoken_total,
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

    # --- Speech + color-word helpers -------------------------------------------

    @staticmethod
    def _normalize_color_word(word: str) -> Optional[str]:
        if not word:
            return None
        lowered = word.strip().lower()
        if not lowered:
            return None
        if lowered in COLOR_SYNONYMS:
            return COLOR_SYNONYMS[lowered]
        if lowered in CANONICAL_COLOR_TUPLES:
            return lowered
        return None

    def _color_to_name(self, color: Optional[Color]) -> Optional[str]:
        if color is None:
            return None
        vec = np.array(color, dtype=np.float32)
        best_name: Optional[str] = None
        best_dist: Optional[float] = None
        for name, target in CANONICAL_COLOR_VECTORS.items():
            dist = float(np.linalg.norm(vec - target))
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_name = name
        return best_name

    def _extract_spoken_color_names(self, transcript: str) -> List[str]:
        if not transcript:
            return []
        tokens = re.findall(r"[A-Za-z]+", transcript)
        names: List[str] = []
        for token in tokens:
            normalized = self._normalize_color_word(token)
            if normalized is not None:
                names.append(normalized)
        return names

    def _resolve_speech_source(self, attempt_dir: Path, speech_media: Optional[Path], *, mode: str) -> Optional[Path]:
        if speech_media is not None:
            media_path = speech_media if speech_media.is_absolute() else attempt_dir / speech_media
            if media_path.exists() and media_path.is_file():
                return media_path
            return None
        preference = mode.lower().strip()
        patterns: List[str]
        if preference == "video":
            patterns = list(VIDEO_GLOBS) + list(AUDIO_GLOBS)
        elif preference == "audio":
            patterns = list(AUDIO_GLOBS) + list(VIDEO_GLOBS)
        elif preference == "off":
            return None
        else:
            patterns = list(AUDIO_GLOBS) + list(VIDEO_GLOBS)
        for pattern in patterns:
            for candidate in attempt_dir.glob(pattern):
                if candidate.is_file():
                    return candidate
        return None

    def _transcribe_media(
        self,
        media_path: Path,
        *,
        attempt_dir: Path,
        engine: str,
        model: str,
        base_url: Optional[str],
        whisper_model: str,
        language: Optional[str],
    ) -> Optional[Tuple[str, List[str], List[Color], Path]]:
        script_path = Path.cwd() / "scripts" / "transcribe_video.py"
        if not script_path.exists():
            return None
        json_out = attempt_dir / SPEECH_TRANSCRIPT_FILENAME
        json_out.parent.mkdir(parents=True, exist_ok=True)
        cmd: List[str] = [
            str(script_path),
            media_path.as_posix(),
            "--output-json",
            json_out.as_posix(),
        ]
        engine_lower = engine.lower().strip() if engine else "local"
        if engine_lower == "api":
            cmd.extend(["--engine", "api", "--model", model])
            if base_url:
                cmd.extend(["--base-url", base_url])
        else:
            cmd.extend(["--engine", "local", "--whisper-model", whisper_model])
            if language:
                cmd.extend(["--language", language])
        import sys as _sys
        py_cmd = [_sys.executable, cmd[0]] + cmd[1:]
        completed = subprocess.run(py_cmd, capture_output=True, text=True)
        if completed.returncode != 0:
            return None
        if not json_out.exists():
            return None
        payload_text = json_out.read_text(encoding="utf-8")
        if not payload_text.strip():
            return None
        payload = json.loads(payload_text)
        if not isinstance(payload, dict):
            return None
        transcript_raw = payload.get("transcript")
        transcript = transcript_raw if isinstance(transcript_raw, str) else None
        if not transcript:
            return None
        spoken_names = self._extract_spoken_color_names(transcript)
        spoken_rgb: List[Color] = []
        for name in spoken_names:
            color_tuple = CANONICAL_COLOR_TUPLES.get(name)
            if color_tuple is not None:
                spoken_rgb.append(color_tuple)
        return transcript, spoken_names, spoken_rgb, json_out


__all__ = ["RectsEvaluator", "RectsEvaluationResult", "OrderPosition"]


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate rectangles-order puzzles")
    parser.add_argument("metadata", type=Path)
    parser.add_argument("puzzle_id", type=str)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--base-dir", type=Path, default=None)
    parser.add_argument("--speech-media", dest="speech_media", type=Path, default=None, help="Explicit path to audio/video answer")
    parser.add_argument(
        "--speech-mode",
        dest="speech_mode",
        choices=["auto", "audio", "video", "off"],
        default="auto",
        help="How to search for speech media alongside the candidate image",
    )
    parser.add_argument(
        "--speech-engine",
        dest="speech_engine",
        choices=["local", "api"],
        default="local",
        help="Transcription engine to use (local Whisper or API)",
    )
    parser.add_argument("--speech-model", dest="speech_model", type=str, default="whisper-1", help="API transcription model name")
    parser.add_argument("--speech-base-url", dest="speech_base_url", type=str, default=None, help="Override API base URL for transcription")
    parser.add_argument(
        "--whisper-model",
        dest="whisper_model",
        type=str,
        default="base",
        help="Local Whisper model size when speech-engine=local",
    )
    parser.add_argument(
        "--speech-language",
        dest="speech_language",
        type=str,
        default="en",
        help="Language hint passed to Whisper when transcribing locally",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)
    evaluator = RectsEvaluator(args.metadata, base_dir=args.base_dir)
    result = evaluator.evaluate(
        args.puzzle_id,
        args.candidate,
        speech_media=args.speech_media,
        speech_mode=args.speech_mode,
        speech_engine=args.speech_engine,
        speech_model=args.speech_model,
        speech_base_url=args.speech_base_url,
        whisper_model=args.whisper_model,
        speech_language=args.speech_language,
    )
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
