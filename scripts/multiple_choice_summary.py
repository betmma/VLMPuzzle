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
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VOTE_OUTPUT_ROOT = REPO_ROOT / "data" / "voteOutput"

KEYS=['predicted_option','transcribe_option','video_option','image_option','text_option']

VOTE_IGNORE_NON_ALPHABETIC = True

@dataclass(frozen=True)
class AttemptRecord:
    puzzle_type: str
    puzzle_id: str
    attempt_index: int
    predicted_option: Optional[str]
    correct_option: str
    is_correct: bool
    output_directory: Path


@dataclass(frozen=True)
class MultiAttemptRecord:
    puzzle_type: str
    puzzle_id: str
    attempt_index: int
    correct_option: str
    predictions: Dict[str, Optional[str]]
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
    prefix = "_".join(name.split("_")[:-3]) # prefix_date_time
    return prefix or "unknown"


def _normalize_option(raw: Optional[object]) -> Optional[str]:
    if raw is None:
        return None
    text = str(raw).strip().upper()
    return text


def _parse_multi_attempt(evaluation_path: Path) -> Optional[MultiAttemptRecord]:
    payload = json.loads(evaluation_path.read_text(encoding="utf-8"))
    stdout_blob = payload.get("stdout")
    if not stdout_blob:
        return None
    inner = json.loads(stdout_blob)
    puzzle_id = str(inner.get("puzzle_id") or "").strip()
    correct_option = _normalize_option(inner.get("correct_option"))
    if not puzzle_id or correct_option is None:
        return None
    attempt_index = _coerce_attempt_index(payload.get("attempt"))
    puzzle_type = _infer_puzzle_type(payload.get("vote_run_directory"))
    predictions: Dict[str, Optional[str]] = {
        candidate_key: _normalize_option(inner.get(candidate_key)) for candidate_key in KEYS
    }
    return MultiAttemptRecord(
        puzzle_type=puzzle_type,
        puzzle_id=puzzle_id,
        attempt_index=attempt_index,
        correct_option=correct_option,
        predictions=predictions,
        output_directory=evaluation_path.parent,
    )


def _parse_attempt(evaluation_path: Path, key: str) -> Optional[AttemptRecord]:
    base_record = _parse_multi_attempt(evaluation_path)
    if base_record is None:
        return None
    predicted_option = base_record.predictions.get(key)
    is_correct = predicted_option == base_record.correct_option
    return AttemptRecord(
        puzzle_type=base_record.puzzle_type,
        puzzle_id=base_record.puzzle_id,
        attempt_index=base_record.attempt_index,
        predicted_option=predicted_option,
        correct_option=base_record.correct_option,
        is_correct=is_correct,
        output_directory=base_record.output_directory,
    )


def _iter_evaluation_paths(vote_root: Path) -> Iterator[Path]:
    if not vote_root.exists() or not vote_root.is_dir():
        return
    for vote_run in sorted(vote_root.iterdir()):
        if not vote_run.is_dir():
            continue
        nested_attempts = False
        for attempt_dir in sorted(vote_run.iterdir()):
            if not attempt_dir.is_dir():
                continue
            evaluation_path = attempt_dir / "evaluation.json"
            if evaluation_path.exists() and evaluation_path.is_file():
                nested_attempts = True
                yield evaluation_path
        if nested_attempts:
            continue
        evaluation_path = vote_run / "evaluation.json"
        if evaluation_path.exists() and evaluation_path.is_file():
            yield evaluation_path


def _iter_attempts(vote_root: Path, key:str) -> Iterable[AttemptRecord]:
    if not vote_root.exists() or not vote_root.is_dir():
        return []
    attempts: List[AttemptRecord] = []
    for evaluation_path in _iter_evaluation_paths(vote_root):
        record = _parse_attempt(evaluation_path,key)
        if record is not None:
            attempts.append(record)
    return attempts


def _iter_multi_attempts(vote_root: Path) -> List[MultiAttemptRecord]:
    if not vote_root.exists() or not vote_root.is_dir():
        return []
    records: List[MultiAttemptRecord] = []
    for evaluation_path in _iter_evaluation_paths(vote_root):
        record = _parse_multi_attempt(evaluation_path)
        if record is not None:
            records.append(record)
    return records


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


def _group_multi_by_type(records: Iterable[MultiAttemptRecord]) -> Dict[str, List[MultiAttemptRecord]]:
    grouped: Dict[str, List[MultiAttemptRecord]] = defaultdict(list)
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


def _summarize_voted_accuracy(records: Iterable[AttemptRecord]) -> None:
    grouped = _group_by_puzzle(records)
    eligible = 0
    voted_correct = 0
    tied = 0
    unresolved = 0
    per_type_correct: Dict[str, int] = defaultdict(int)
    per_type_total: Dict[str, int] = defaultdict(int)

    for attempts in grouped.values():
        if len(attempts) <= 1:
            continue
        eligible += 1
        counts = Counter(
            record.predicted_option for record in attempts if record.predicted_option is not None and (not VOTE_IGNORE_NON_ALPHABETIC or record.predicted_option.isalpha())
        )
        if not counts:
            unresolved += 1
            predicted = None
        else:
            most_common = counts.most_common()
            top_count = most_common[0][1]
            winners = [option for option, count in most_common if count == top_count]
            if len(winners) == 1:
                predicted = winners[0]
            else:
                tied += 1
                predicted = None
        correct_option = attempts[0].correct_option
        is_correct = predicted == correct_option
        if is_correct:
            voted_correct += 1
        puzzle_type = attempts[0].puzzle_type
        per_type_total[puzzle_type] += 1
        if is_correct:
            per_type_correct[puzzle_type] += 1

    if not eligible:
        return

    accuracy = voted_correct / eligible
    print("Voted accuracy across puzzles with multiple attempts:")
    print(f"  Eligible puzzles: {eligible}")
    print(f"  Voted correct: {voted_correct}/{eligible} correct ({accuracy:.0%})")
    if tied:
        print(f"  Ties (treated as incorrect): {tied}")
    if unresolved:
        print(f"  No predictions across attempts: {unresolved}")
    for puzzle_type in sorted(per_type_total):
        total = per_type_total[puzzle_type]
        correct = per_type_correct.get(puzzle_type, 0)
        rate = (correct / total) if total else 0.0
        print(f"    {puzzle_type}: {correct}/{total} correct ({rate:.0%})")
    print()


def _attempt_records_from_multi(records: Iterable[MultiAttemptRecord], key: str) -> List[AttemptRecord]:
    converted: List[AttemptRecord] = []
    for record in records:
        predicted_option = record.predictions.get(key)
        is_correct = predicted_option == record.correct_option
        converted.append(
            AttemptRecord(
                puzzle_type=record.puzzle_type,
                puzzle_id=record.puzzle_id,
                attempt_index=record.attempt_index,
                predicted_option=predicted_option,
                correct_option=record.correct_option,
                is_correct=is_correct,
                output_directory=record.output_directory,
            )
        )
    return converted


def summarize_multiple_choice_attempts(
    vote_root: Path,
    key: str,
    top_misses: int,
    attempts: Optional[List[AttemptRecord]] = None,
) -> bool:
    attempt_records = attempts if attempts is not None else list(_iter_attempts(vote_root, key))
    if not attempt_records:
        print(f"No multiple-choice evaluations found under {vote_root.as_posix()}.")
        return False
    
    all_none = all(record.predicted_option is None for record in attempt_records)
    if all_none:
        print(f"All attempts have no recognized '{key}' option.")
        return False

    total = len(attempt_records)
    correct = sum(1 for record in attempt_records if record.is_correct)
    accuracy = (correct / total) if total else 0.0

    print("Multiple-choice evaluation summary")
    print(f"Vote output root: {vote_root.as_posix()}")
    print(f"Total attempts: {total}")
    print(f"Correct attempts: {correct} ({accuracy:.0%})")
    print()

    _summarize_types(attempt_records)
    _print_option_breakdown(attempt_records)
    print()
    _summarize_voted_accuracy(attempt_records)
    _summarize_puzzles(attempt_records, top_misses)
    return True


def _summarize_key_correlations(multi_attempts: List[MultiAttemptRecord]) -> None:
    if not multi_attempts:
        print("No evaluation attempts available for key correlation analysis.")
        return

    existing_keys = [key for key in KEYS if any(record.predictions.get(key) is not None for record in multi_attempts)]
    if len(existing_keys) < 2:
        print("Key correlation analysis skipped: fewer than two prediction keys with data.")
        return

    print("Detected prediction keys with data:")
    print("  " + ", ".join(sorted(existing_keys)))
    print()

    grouped = _group_multi_by_type(multi_attempts)
    print("Prediction key correlation by puzzle type:")
    for puzzle_type in sorted(grouped.keys()):
        records = grouped[puzzle_type]
        pair_stats = []
        for key_a, key_b in combinations(existing_keys, 2):
            total = 0
            same_count = 0
            correct_when_same = 0
            diff_count = 0
            diff_correct_a = 0
            diff_correct_b = 0
            for record in records:
                pred_a = record.predictions.get(key_a)
                pred_b = record.predictions.get(key_b)
                total += 1
                if pred_a == pred_b:
                    same_count += 1
                    if pred_a == record.correct_option:
                        correct_when_same += 1
                    continue
                diff_count += 1
                if pred_a == record.correct_option:
                    diff_correct_a += 1
                if pred_b == record.correct_option:
                    diff_correct_b += 1
            if not total:
                continue
            same_rate = same_count / total
            same_accuracy = (correct_when_same / same_count) if same_count else 0.0
            diff_accuracy_a = (diff_correct_a / diff_count) if diff_count else None
            diff_accuracy_b = (diff_correct_b / diff_count) if diff_count else None
            pair_stats.append(
                (
                    key_a,
                    key_b,
                    total,
                    same_count,
                    same_rate,
                    correct_when_same,
                    same_accuracy,
                    diff_count,
                    diff_correct_a,
                    diff_accuracy_a,
                    diff_correct_b,
                    diff_accuracy_b,
                )
            )
        if not pair_stats:
            continue

        print(f"  {puzzle_type}:")
        for (
            key_a,
            key_b,
            total,
            same_count,
            same_rate,
            correct_same_count,
            same_accuracy,
            diff_count,
            diff_correct_a,
            diff_accuracy_a,
            diff_correct_b,
            diff_accuracy_b,
        ) in pair_stats:
            print(f"    {key_a} vs {key_b}:")
            print(f"      Comparable attempts: {total}")
            print(f"      Same predictions: {same_count}/{total} ({same_rate:.0%})")
            if same_count:
                print(f"      Accuracy when same: {correct_same_count}/{same_count} ({same_accuracy:.0%})")
            else:
                print("      Accuracy when same: n/a")
            if diff_count:
                print(
                    "      Accuracy when different -> "
                    + f"{key_a}: {diff_correct_a}/{diff_count} ({diff_accuracy_a:.0%}), "
                    + f"{key_b}: {diff_correct_b}/{diff_count} ({diff_accuracy_b:.0%})"
                )
            else:
                print("      Accuracy when different: n/a")
        print()


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
    multi_attempts = _iter_multi_attempts(vote_root)
    per_key_attempts = {key: _attempt_records_from_multi(multi_attempts, key) for key in KEYS}

    any_found = False
    for key in KEYS:
        print(f"Summary for key: {key}")
        found = summarize_multiple_choice_attempts(
            vote_root,
            key=key,
            top_misses=max(0, top_misses),
            attempts=per_key_attempts.get(key),
        )
        any_found = any_found or found

    print()
    # _summarize_key_correlations(multi_attempts)
    return any_found

def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    summarize_all(args.output_root, top_misses=max(0, args.top_misses))


if __name__ == "__main__":
    main()
