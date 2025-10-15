"""Summarize multiple-choice evaluation attempts stored under data/voteOutput.

This script scans ``evaluation.json`` payloads produced by vote pipelines for
multiple-choice puzzles such as arc_connect. It extracts the predicted and
correct options for each attempt and prints aggregate accuracy metrics by
puzzle type, puzzle id, and answer option.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VOTE_OUTPUT_ROOT = REPO_ROOT / "data" / "voteOutput"

KEYS=['predicted_option','transcribe_option','image_option','text_option']

@dataclass(frozen=True)
class AttemptRecord:
    puzzle_type: str
    puzzle_id: str
    attempt_index: int
    predicted_option: Optional[str]
    correct_option: str
    is_correct: bool
    output_directory: Path


def _coerce_attempt_index(value: object) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return 0


def _infer_puzzle_type(vote_run_directory: Optional[str]) -> str:
    if not vote_run_directory:
        return "unknown"
    path = Path(vote_run_directory)
    name = path.name
    if not name:
        return "unknown"
    prefix = name.split("_", 1)[0]
    return prefix or "unknown"


def _normalize_option(raw: Optional[object]) -> Optional[str]:
    if raw is None:
        return None
    text = str(raw).strip().upper()
    return text


def _parse_attempt(evaluation_path: Path, key:str) -> Optional[AttemptRecord]:
    payload = json.loads(evaluation_path.read_text(encoding="utf-8"))
    stdout_blob = payload.get("stdout")
    if not stdout_blob:
        return None
    inner = json.loads(stdout_blob)
    puzzle_id = str(inner.get("puzzle_id") or "").strip()
    correct_option = _normalize_option(inner.get("correct_option"))
    predicted_option = _normalize_option(inner.get(key))
    is_correct = correct_option == predicted_option
    if not puzzle_id or correct_option is None:
        return None
    attempt_index = _coerce_attempt_index(payload.get("attempt"))
    puzzle_type = _infer_puzzle_type(payload.get("vote_run_directory"))
    return AttemptRecord(
        puzzle_type=puzzle_type,
        puzzle_id=puzzle_id,
        attempt_index=attempt_index,
        predicted_option=predicted_option,
        correct_option=correct_option,
        is_correct=is_correct,
        output_directory=evaluation_path.parent,
    )


def _iter_attempts(vote_root: Path, key:str) -> Iterable[AttemptRecord]:
    if not vote_root.exists() or not vote_root.is_dir():
        return []
    attempts: List[AttemptRecord] = []
    for vote_run in sorted(vote_root.iterdir()):
        if not vote_run.is_dir():
            continue
        nested_attempts = False
        for attempt_dir in sorted(vote_run.iterdir()):
            if not attempt_dir.is_dir():
                continue
            evaluation_path = attempt_dir / "evaluation.json"
            if evaluation_path.exists() and evaluation_path.is_file():
                record = _parse_attempt(evaluation_path,key)
                if record is not None:
                    attempts.append(record)
                    nested_attempts = True
        if nested_attempts:
            continue
        evaluation_path = vote_run / "evaluation.json"
        if evaluation_path.exists() and evaluation_path.is_file():
            record = _parse_attempt(evaluation_path,key)
            if record is not None:
                attempts.append(record)
    return attempts


def _group_by_puzzle(records: Iterable[AttemptRecord]) -> Dict[str, List[AttemptRecord]]:
    grouped: Dict[str, List[AttemptRecord]] = defaultdict(list)
    for record in records:
        grouped[record.puzzle_id].append(record)
    return grouped


def _group_by_type(records: Iterable[AttemptRecord]) -> Dict[str, List[AttemptRecord]]:
    grouped: Dict[str, List[AttemptRecord]] = defaultdict(list)
    for record in records:
        grouped[record.puzzle_type].append(record)
    return grouped


def _print_option_breakdown(records: Iterable[AttemptRecord]) -> None:
    total_per_option: Dict[str, int] = {}
    correct_per_option: Dict[str, int] = {}
    predicted_counter: Counter[str] = Counter()
    predicted_none = 0

    for record in records:
        total_per_option[record.correct_option] = total_per_option.get(record.correct_option, 0) + 1
        if record.is_correct:
            correct_per_option[record.correct_option] = correct_per_option.get(record.correct_option, 0) + 1
        if record.predicted_option is None:
            predicted_none += 1
        else:
            predicted_counter[record.predicted_option] += 1

    print("Option accuracy (by correct answer):")
    for option in sorted(total_per_option.keys()):
        total = total_per_option[option]
        correct = correct_per_option.get(option, 0)
        rate = (correct / total) if total else 0.0
        print(f"  {option}: {correct}/{total} correct ({rate:.0%})")
    print("Predicted option distribution:")
    for option in sorted(predicted_counter.keys()):
        print(f"  {option}: {predicted_counter.get(option, 0)}")
    if predicted_none:
        print(f"  (unrecognized): {predicted_none}")


def _summarize_puzzles(records: Iterable[AttemptRecord], limit: int) -> None:
    grouped = _group_by_puzzle(records)
    worst_cases: List[tuple[int, int, str, str]] = []
    for puzzle_id, attempts in grouped.items():
        total = len(attempts)
        correct = sum(1 for record in attempts if record.is_correct)
        incorrect = total - correct
        worst_cases.append((incorrect, total, attempts[0].puzzle_type, puzzle_id))
    worst_cases.sort(reverse=True)
    trimmed = worst_cases[:limit]
    if not trimmed:
        return
    print("Most-missed puzzles:")
    for incorrect, total, puzzle_type, puzzle_id in trimmed:
        rate = ((total - incorrect) / total) if total else 0.0
        print(f"  {puzzle_type} {puzzle_id}: {total - incorrect}/{total} correct ({rate:.0%})")
    print()


def _summarize_types(records: Iterable[AttemptRecord]) -> None:
    grouped = _group_by_type(records)
    print("Performance by puzzle type:")
    for puzzle_type in sorted(grouped.keys()):
        attempts = grouped[puzzle_type]
        total = len(attempts)
        correct = sum(1 for record in attempts if record.is_correct)
        rate = (correct / total) if total else 0.0
        print(f"  {puzzle_type}: {correct}/{total} correct ({rate:.0%})")
    print()


def summarize_multiple_choice_attempts(vote_root: Path, key: str, top_misses: int) -> bool:
    attempts = list(_iter_attempts(vote_root,key))
    if not attempts:
        print(f"No multiple-choice evaluations found under {vote_root.as_posix()}.")
        return False

    total = len(attempts)
    correct = sum(1 for record in attempts if record.is_correct)
    accuracy = (correct / total) if total else 0.0

    print("Multiple-choice evaluation summary")
    print(f"Vote output root: {vote_root.as_posix()}")
    print(f"Total attempts: {total}")
    print(f"Correct attempts: {correct} ({accuracy:.0%})")
    print()

    _summarize_types(attempts)
    _print_option_breakdown(attempts)
    print()
    _summarize_puzzles(attempts, top_misses)
    return True


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize multiple-choice puzzle evaluation results."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
    default=DEFAULT_VOTE_OUTPUT_ROOT,
    help="Root directory containing vote outputs (default: data/voteOutput)",
    )
    parser.add_argument(
        "--top-misses",
        type=int,
        default=5,
        help="Number of lowest-accuracy puzzles to list",
    )
    return parser

def summarize_all(vote_root: Path, top_misses: int) -> bool:
    any_found = False
    for key in KEYS:
        print(f"Summary for key: {key}")
        found = summarize_multiple_choice_attempts(vote_root, key=key, top_misses=max(0, top_misses))
        any_found = any_found or found
    return any_found

def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    for key in KEYS:
        print(f"Summary for key: {key}")
        summarize_multiple_choice_attempts(args.output_root, key=key, top_misses=max(0, args.top_misses))


if __name__ == "__main__":
    main()
