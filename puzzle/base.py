"""Abstract interfaces for puzzle generation and evaluation."""

from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Generic, Iterable, List, Optional, Tuple, TypeVar, Union
from dataclasses import dataclass

PathLike = Union[str, Path]
RecordT = TypeVar("RecordT")


_NATO_WORDS = [
    # Variants first where applicable
    "ALPHA", "ALFA",
    "BRAVO",
    "CHARLIE",
    "DELTA",
    "ECHO",
    "FOXTROT",
    "GOLF",
    "HOTEL",
    "INDIA",
    "JULIET", "JULIETT",
    "KILO",
    "LIMA",
    "MIKE",
    "NOVEMBER",
    "OSCAR",
    "PAPA",
    "QUEBEC",
    "ROMEO",
    "SIERRA",
    "TANGO",
    "UNIFORM",
    "VICTOR",
    "WHISKEY",
    "XRAY", "X-RAY",
    "YANKEE",
    "ZULU",
]
def extract_first_nato_word(text: str) -> Optional[str]:
    """Return the first NATO phonetic code word found in `text`.

    - Matches are case-insensitive and must align to word boundaries.
    - Accepts common variants (e.g., ALFA/ALPHA, JULIET/JULIETT, XRAY/X-RAY).
    - Returns the normalized upper-case code word as found in `_NATO_WORDS`.
    - Returns None when no code word is present.
    """

    if not text:
        return None
    import re

    # Build a single regex from the word list with word boundaries.
    # Use alternation ordered as in _NATO_WORDS so variants keep priority.
    pattern = r"\b(?:(%s))\b" % ("|".join(map(re.escape, _NATO_WORDS)))
    regex = re.compile(pattern, flags=re.IGNORECASE)
    match = regex.findall(text)
    if not match:
        return None
    # Normalize to the canonical upper-case variant from the list
    found = match[0]
    found_upper = found.upper()
    return found_upper[0]

class AbstractPuzzleGenerator(ABC, Generic[RecordT]):
    """Base class for dataset builders that emit puzzle records."""

    def __init__(self, output_dir: PathLike) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def create_puzzle(self, *args, **kwargs) -> RecordT:
        """Create a puzzle from the provided resources."""

    def create_random_puzzle(self) -> RecordT:
        """Create a single randomized puzzle instance."""
        return self.create_puzzle()

    def generate_dataset(
        self,
        count: int,
        *,
        metadata_path: Optional[PathLike] = None,
        append: bool = True,
    ) -> List[RecordT]:
        """Generate a batch of puzzles and optionally persist metadata."""
        records = [self.create_random_puzzle() for _ in range(count)]
        if metadata_path is not None:
            self.write_metadata(records, metadata_path, append=append)
        return records

    def write_metadata(
        self,
        records: Iterable[RecordT],
        metadata_path: PathLike,
        *,
        append: bool = True,
    ) -> None:
        """Serialize puzzle records to JSON, appending if requested."""

        path = Path(metadata_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        existing: List[Dict[str, Any]] = []
        if append and path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
        payload = [self.record_to_dict(record) for record in records]
        path.write_text(json.dumps(existing + payload, indent=2), encoding="utf-8")

    def record_to_dict(self, record: RecordT) -> Dict[str, Any]:
        """Dictionary serialization hook for puzzle records."""

        if hasattr(record, "to_dict"):
            return getattr(record, "to_dict")()
        return dataclasses.asdict(record)

    def relativize_path(self, path: Path) -> str:
        """Map an absolute path into the generator output directory when possible."""

        try:
            return path.relative_to(self.output_dir).as_posix()
        except ValueError:
            return path.as_posix()


class AbstractPuzzleEvaluator(ABC):
    """Base class scaffolding for puzzle evaluators."""
    
    @dataclass
    class OptionEvaluationResult:
        puzzle_id: str
        correct_option: str
        transcribe_option: Optional[str]
        video_option: Optional[str]
        image_option: Optional[str]
        text_option: Optional[str]
        attempt_dir: str
        
        def to_dict(self) -> Dict[str, Optional[object]]:
            return {
                "puzzle_id": self.puzzle_id,
                "correct_option": self.correct_option,
                "transcribe_option": self.transcribe_option,
                "video_option": self.video_option,
                "image_option": self.image_option,
                "text_option": self.text_option,
                "attempt_dir": self.attempt_dir,
            }

    def __init__(
        self,
        metadata_path: PathLike,
        *,
        base_dir: Optional[PathLike] = None,
    ) -> None:
        self.metadata_path = Path(metadata_path)
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")
        self.base_dir = Path(base_dir) if base_dir is not None else self.metadata_path.parent
        self._records = self._load_metadata()

    @property
    def records(self) -> Dict[str, Dict[str, Any]]:
        """Return the loaded metadata keyed by puzzle id."""

        return self._records

    def _read_metadata(self) -> List[Dict[str, Any]]:
        raw = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise ValueError("Puzzle metadata must be a list of records")
        return raw

    def _load_metadata(self) -> Dict[str, Dict[str, Any]]:
        records: Dict[str, Dict[str, Any]] = {}
        for record in self._read_metadata():
            puzzle_id = record.get("id")
            if not puzzle_id:
                raise ValueError("Each puzzle record must include an 'id'")
            records[str(puzzle_id)] = record
        return records

    def get_record(self, puzzle_id: str) -> Dict[str, Any]:
        try:
            return self._records[puzzle_id]
        except KeyError as exc:
            raise KeyError(f"Puzzle id '{puzzle_id}' not found in metadata") from exc

    def resolve_path(self, path_value: object) -> Path:
        candidate = Path(str(path_value))
        if not candidate.is_absolute():
            candidate = self.base_dir / candidate
        return candidate

    @staticmethod
    def _coerce_dimension_pair(value: object) -> Optional[Tuple[int, int]]:
        """Convert a raw canvas dimension payload into a (width, height) pair."""

        if isinstance(value, (list, tuple)) and len(value) >= 2:
            raw_width, raw_height = value[0], value[1]
        elif isinstance(value, dict) and {"width", "height"} <= set(value):
            raw_width, raw_height = value["width"], value["height"]
        else:
            return None
        try:
            width = int(round(float(raw_width)))
            height = int(round(float(raw_height)))
        except (TypeError, ValueError):
            return None
        return (width, height) if width > 0 and height > 0 else None

    def _record_canvas_dimensions(
        self,
        record: Dict[str, Any],
        *,
        fallback: Optional[Tuple[int, int]] = None,
    ) -> Tuple[int, int]:
        """Best-effort lookup for the original canvas size stored in metadata."""

        dims = self._coerce_dimension_pair(record.get('canvas_dimensions'))
        if dims:
            return dims
        canvas_size = record.get('canvas_size')
        if isinstance(canvas_size, (int, float)) and canvas_size > 0:
            size_int = int(round(float(canvas_size)))
            return size_int, size_int
        cell_bboxes = record.get('cell_bboxes')
        if isinstance(cell_bboxes, Iterable):
            max_right = 0.0
            max_bottom = 0.0
            for row in cell_bboxes:
                if not isinstance(row, Iterable):
                    continue
                for bbox in row:
                    if not isinstance(bbox, Iterable):
                        continue
                    coords = list(bbox)
                    if len(coords) < 4:
                        continue
                    _, _, right, bottom = coords[:4]
                    try:
                        max_right = max(max_right, float(right))
                        max_bottom = max(max_bottom, float(bottom))
                    except (TypeError, ValueError):
                        continue
            if max_right > 0 and max_bottom > 0:
                return int(round(max_right)), int(round(max_bottom))
        if fallback is not None:
            return fallback
        raise ValueError('Unable to determine canvas dimensions from record')

    def scale_cell_bboxes(
        self,
        bboxes: Iterable[Iterable[Iterable[float]]],
        *,
        source_size: Tuple[int, int],
        target_size: Tuple[int, int],
        margin_px: int = 4,
    ) -> List[List[Tuple[int, int, int, int]]]:
        """Scale recorded cell bounding boxes into the coordinate space of an image."""

        source_width, source_height = source_size
        target_width, target_height = target_size
        if source_width <= 0 or source_height <= 0:
            raise ValueError('source_size must contain positive dimensions')
        if target_width <= 0 or target_height <= 0:
            raise ValueError('target_size must contain positive dimensions')
        scale_x = target_width / source_width
        scale_y = target_height / source_height

        def _clip(value: float, lower: int, upper: int) -> int:
            return max(lower, min(upper, int(round(value))))

        mapped: List[List[Tuple[int, int, int, int]]] = []
        for row in bboxes:
            if not isinstance(row, Iterable):
                continue
            mapped_row: List[Tuple[int, int, int, int]] = []
            for bbox in row:
                if not isinstance(bbox, Iterable):
                    continue
                coords = list(bbox)
                if len(coords) < 4:
                    continue
                left, top, right, bottom = coords[:4]
                try:
                    left_f = float(left)
                    top_f = float(top)
                    right_f = float(right)
                    bottom_f = float(bottom)
                except (TypeError, ValueError):
                    continue
                if margin_px > 0:
                    left_f = max(0.0, left_f + margin_px)
                    top_f = max(0.0, top_f + margin_px)
                    right_f = min(float(source_width), right_f - margin_px)
                    bottom_f = min(float(source_height), bottom_f - margin_px)
                    if right_f <= left_f:
                        left_f, right_f = float(left), float(right)
                    if bottom_f <= top_f:
                        top_f, bottom_f = float(top), float(bottom)
                scaled_left = _clip(left_f * scale_x, 0, target_width - 1)
                scaled_top = _clip(top_f * scale_y, 0, target_height - 1)
                scaled_right = _clip(right_f * scale_x, scaled_left + 1, target_width)
                scaled_bottom = _clip(bottom_f * scale_y, scaled_top + 1, target_height)
                mapped_row.append((scaled_left, scaled_top, scaled_right, scaled_bottom))
            if mapped_row:
                mapped.append(mapped_row)
        if not mapped:
            raise ValueError('No valid bounding boxes were produced during scaling')
        return mapped

    def map_cell_bboxes_to_image(
        self,
        record: Dict[str, Any],
        *,
        target_size: Tuple[int, int],
        margin_px: int = 4,
        reference_size: Optional[Tuple[int, int]] = None,
    ) -> List[List[Tuple[int, int, int, int]]]:
        """Convenience wrapper to scale recorded cell boxes to a new image size."""

        if 'cell_bboxes' not in record:
            raise KeyError("Puzzle record is missing 'cell_bboxes'")
        source_size = reference_size or self._record_canvas_dimensions(record, fallback=target_size)
        return self.scale_cell_bboxes(
            record['cell_bboxes'],
            source_size=source_size,
            target_size=target_size,
            margin_px=margin_px,
        )
        
    @staticmethod
    def transcribe_video(
        attempt_dir: PathLike,
        engine: str = "local",
        model: str = "whisper-1",
        base_url: Optional[str] = None,
        whisper_model: str = "base",
        script_path: Optional[PathLike] = None,
    ) -> Dict[str, Any]:
        """Run the shared transcription script and return the raw JSON payload string when available."""

        out_dir = Path(attempt_dir)
        video = out_dir / "video_1.mp4"
        if not video.exists() or not video.is_file():
            return {}
        script = Path(script_path) if script_path is not None else Path.cwd() / "scripts" / "transcribe_video.py"
        json_out = out_dir / "transcription.json"

        cmd: List[str] = [
            sys.executable,
            script.as_posix(),
            video.as_posix(),
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
        with open(json_out, "r", encoding="utf-8") as f:
            stdout_stripped = f.read().strip()
            return json.loads(stdout_stripped)
    
    @staticmethod
    def extract_first_nato_word(
        transcript: str
    ) -> Optional[str]:
        """Extract the first NATO code word from a transcript string."""
        nato=extract_first_nato_word(transcript)
        if nato:
            return nato
        remove_chars = '.,!?;:"\'()[]{}<>*/'
        
        for word in transcript.split()[::-1]:
            word = word.strip(remove_chars)
            if word in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
                return word
        return None


class EvaluationPayloadReader:
    """Helper to load evaluator payloads from attempt directories."""

    def __init__(self, *, filename: str = "evaluation.json") -> None:
        self.filename = filename

    def read_inner_payload(self, attempt_dir: Path) -> Optional[Dict[str, Any]]:
        evaluation_path = attempt_dir / self.filename
        if not evaluation_path.exists():
            return None
        try:
            with evaluation_path.open("r", encoding="utf-8") as handle:
                outer_payload = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return None
        stdout_blob = outer_payload.get("stdout")
        if not stdout_blob:
            return None
        try:
            return json.loads(stdout_blob)
        except json.JSONDecodeError:
            return None

    @abstractmethod
    def evaluate(self, puzzle_id: str, *args, **kwargs):
        """Evaluate a candidate solution for the given puzzle."""


class AbstractVoteSummarizer(ABC):
    """Interface for modules that summarize voting results across attempts.

    Concrete implementations should implement a single entrypoint `summarize`
    that scans a vote output directory and prints a human-readable summary.
    The method returns True if any summaries were produced, or False otherwise.
    """

    @abstractmethod
    def summarize(self, vote_root: Path, *, prefix_newline: bool = False) -> bool:
        """Summarize votes written under `vote_root`.

        Implementations may choose how to locate per-puzzle and per-attempt
        outputs within `vote_root` and what details to aggregate.

        Returns True if a summary was printed, False if nothing to report.
        """
        raise NotImplementedError


__all__ = [
    "AbstractPuzzleGenerator",
    "AbstractPuzzleEvaluator",
    "EvaluationPayloadReader",
    "AbstractVoteSummarizer",
    "PathLike",
    "transcribe_video",
]
