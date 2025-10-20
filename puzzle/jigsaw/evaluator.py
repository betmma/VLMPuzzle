"""Jigsaw puzzle evaluator implementation."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
from PIL import Image

from ..base import AbstractPuzzleEvaluator, PathLike

try:  # Pillow 9/10 compatibility
    RESAMPLE_LANCZOS = Image.Resampling.LANCZOS
except AttributeError:  # pragma: no cover - older Pillow
    RESAMPLE_LANCZOS = Image.LANCZOS


__all__ = [
    "JigsawEvaluator",
    "JigsawEvaluationResult",
    "PieceEvaluation",
]


@dataclass
class PieceEvaluation:
    """Result for a single puzzle piece."""

    piece_id: str
    similarity: float
    is_correct: bool

    def to_dict(self) -> Dict[str, Union[str, float, bool]]:
        return {
            "piece_id": self.piece_id,
            "similarity": self.similarity,
            "is_correct": self.is_correct,
        }


@dataclass
class JigsawEvaluationResult:
    """Aggregated evaluation score for an entire puzzle."""

    puzzle_id: str
    correct_pieces: int
    total_pieces: int
    accuracy: float
    per_piece: List[PieceEvaluation]

    def to_dict(self) -> Dict[str, Union[str, float, int, List[Dict[str, Union[str, float, bool]]]]]:
        return {
            "puzzle_id": self.puzzle_id,
            "correct_pieces": self.correct_pieces,
            "total_pieces": self.total_pieces,
            "accuracy": self.accuracy,
            "per_piece": [piece.to_dict() for piece in self.per_piece],
        }


class JigsawEvaluator(AbstractPuzzleEvaluator):
    """Evaluate candidate puzzle solutions against stored metadata."""

    def evaluate(
        self,
        puzzle_id: str,
        candidate_image: PathLike,
        *,
        similarity_threshold: float = 0.9,
        trim_tolerance: int = 8,
    ) -> JigsawEvaluationResult:
        record = self.get_record(puzzle_id)

        original_image_path = self.resolve_path(record["original_image_path"])
        candidate_path = Path(candidate_image)
        if not original_image_path.exists():
            raise FileNotFoundError(f"Original image missing: {original_image_path}")
        if not candidate_path.exists():
            raise FileNotFoundError(f"Candidate image missing: {candidate_path}")

        original_image = Image.open(original_image_path).convert("RGB")
        candidate_image_obj = Image.open(candidate_path).convert("RGB")

        trimmed_candidate = self._trim_borders(candidate_image_obj, tolerance=trim_tolerance)
        resized_candidate = trimmed_candidate.resize(original_image.size, RESAMPLE_LANCZOS)

        per_piece_results: List[PieceEvaluation] = []

        for piece in record["pieces"]:
            left, top, right, bottom = map(int, piece["bbox"])
            reference = original_image.crop((left, top, right, bottom))
            candidate_tile = resized_candidate.crop((left, top, right, bottom))
            similarity = self._piece_similarity(reference, candidate_tile)
            is_correct = similarity >= similarity_threshold
            per_piece_results.append(
                PieceEvaluation(
                    piece_id=str(piece["id"]),
                    similarity=similarity,
                    is_correct=is_correct,
                )
            )

        correct = sum(1 for piece in per_piece_results if piece.is_correct)
        total = len(per_piece_results)
        accuracy = correct / total if total else 0.0

        return JigsawEvaluationResult(
            puzzle_id=puzzle_id,
            correct_pieces=correct,
            total_pieces=total,
            accuracy=accuracy,
            per_piece=per_piece_results,
        )

    @staticmethod
    def _trim_borders(image: Image.Image, *, tolerance: int = 8) -> Image.Image:
        arr = np.asarray(image)
        if arr.size == 0:
            return image
        if arr.ndim == 3:
            diff = np.max(np.abs(arr - arr[0, 0]), axis=2)
        else:
            diff = np.abs(arr - arr[0, 0])
        mask = diff > tolerance
        if not np.any(mask):
            return image
        ys, xs = np.where(mask)
        top, bottom = int(ys.min()), int(ys.max())
        left, right = int(xs.min()), int(xs.max())
        return image.crop((left, top, right + 1, bottom + 1))

    @staticmethod
    def _piece_similarity(reference: Image.Image, candidate: Image.Image) -> float:
        reference_arr = np.asarray(reference).astype(np.float32) / 255.0
        candidate_arr = np.asarray(candidate).astype(np.float32) / 255.0
        if reference_arr.shape != candidate_arr.shape:
            candidate_arr = np.asarray(
                candidate.resize(reference.size, RESAMPLE_LANCZOS)
            ).astype(np.float32) / 255.0
        mae = np.mean(np.abs(reference_arr - candidate_arr))
        similarity = max(0.0, 1.0 - mae)
        return float(min(1.0, similarity))


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a puzzle solution image")
    parser.add_argument("metadata", type=Path, help="Path to data.json")
    parser.add_argument("puzzle_id", type=str, help="Puzzle identifier to evaluate")
    parser.add_argument("candidate", type=Path, help="Image produced by the model")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.9,
        help="Similarity threshold for considering a tile correct",
    )
    parser.add_argument(
        "--trim-tolerance",
        type=int,
        default=8,
        help="Pixel tolerance when trimming borders from the candidate image",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=None,
        help="Optional base directory to resolve stored image paths",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)
    evaluator = JigsawEvaluator(args.metadata, base_dir=args.base_dir)
    result = evaluator.evaluate(
        args.puzzle_id,
        args.candidate,
        similarity_threshold=args.threshold,
        trim_tolerance=args.trim_tolerance,
    )
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
