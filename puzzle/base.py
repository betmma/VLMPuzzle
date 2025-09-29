"""Abstract interfaces for puzzle generation and evaluation."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Generic, Iterable, List, Optional, TypeVar, Union

PathLike = Union[str, Path]
RecordT = TypeVar("RecordT")


class AbstractPuzzleGenerator(ABC, Generic[RecordT]):
    """Base class for dataset builders that emit puzzle records."""

    def __init__(self, output_dir: PathLike) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def create_puzzle(self, *args, **kwargs) -> RecordT:
        """Create a puzzle from the provided resources."""

    @abstractmethod
    def create_random_puzzle(self) -> RecordT:
        """Create a single randomized puzzle instance."""

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
        raise TypeError(
            "Puzzle record must implement to_dict() or override record_to_dict() in the generator."
        )

    def relativize_path(self, path: Path) -> str:
        """Map an absolute path into the generator output directory when possible."""

        try:
            return path.relative_to(self.output_dir).as_posix()
        except ValueError:
            return path.as_posix()


class AbstractPuzzleEvaluator(ABC):
    """Base class scaffolding for puzzle evaluators."""

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

    @abstractmethod
    def evaluate(self, puzzle_id: str, *args, **kwargs):
        """Evaluate a candidate solution for the given puzzle."""


__all__ = [
    "AbstractPuzzleGenerator",
    "AbstractPuzzleEvaluator",
    "PathLike",
]
