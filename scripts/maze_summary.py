"""Summarize evaluation results for maze-style puzzles.

This utility scans one or more directories for ``evaluation.json`` payloads
produced by maze puzzle evaluators. It aggregates connectivity diagnostics,
flag violations, and per-puzzle performance to help triage failure modes.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVALUATION_ROOT = REPO_ROOT / "data" / "output"


@dataclass(frozen=True)
class MazeAttempt:
    puzzle_type: str
    puzzle_id: str
    attempt_index: int
    red_pixel_count: Optional[int]
    overlaps_walls: Optional[bool]
    touches_start: Optional[bool]
    touches_goal: Optional[bool]
    connected: Optional[bool]
    success: bool
    message: str
    evaluation_path: Path


def _coerce_attempt_index(value: object) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return 0


def _coerce_int(value: object) -> Optional[int]:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if text.isdigit():
            return int(text)
        if text.startswith("-") and text[1:].isdigit():
            return int(text)
    return None


def _coerce_bool(value: object) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "y"}:
            return True
        if text in {"0", "false", "no", "n"}:
            return False
    return None


def _infer_puzzle_type(vote_run_directory: Optional[str]) -> str:
    if not vote_run_directory:
        return "unknown"
    name = Path(vote_run_directory).name
    if not name:
        return "unknown"
    parts = name.split("_")
    index = 0
    while index < len(parts) and "-" not in parts[index]:
        index += 1
    if index == 0:
        return "unknown"
    prefix = "_".join(parts[:index])
    return prefix or "unknown"


def _parse_stdout(payload: Dict[str, object]) -> Optional[Dict[str, object]]:
    stdout_blob = payload.get("stdout")
    if not isinstance(stdout_blob, str):
        return None
    trimmed = stdout_blob.strip()
    if not trimmed:
        return None
    return json.loads(trimmed)


def _parse_attempt(evaluation_path: Path) -> Optional[MazeAttempt]:
    raw_payload = evaluation_path.read_text(encoding="utf-8")
    payload = json.loads(raw_payload)
    stdout_data = _parse_stdout(payload)
    if stdout_data is None:
        return None

    puzzle_id_value = stdout_data.get("puzzle_id")
    puzzle_id = str(puzzle_id_value or "").strip()
    if not puzzle_id:
        return None

    red_pixel_count = _coerce_int(stdout_data.get("red_pixel_count"))
    overlaps_walls = _coerce_bool(stdout_data.get("overlaps_walls"))
    touches_start = _coerce_bool(stdout_data.get("touches_start"))
    touches_goal = _coerce_bool(stdout_data.get("touches_goal"))
    connected = _coerce_bool(stdout_data.get("connected"))
    message_value = stdout_data.get("message")
    message = str(message_value).strip() if isinstance(message_value, str) else ""

    success = False
    if overlaps_walls is False and touches_start is True and touches_goal is True and connected is True:
        success = True

    attempt_index = _coerce_attempt_index(payload.get("attempt"))
    puzzle_type = _infer_puzzle_type(payload.get("vote_run_directory"))

    return MazeAttempt(
        puzzle_type=puzzle_type,
        puzzle_id=puzzle_id,
        attempt_index=attempt_index,
        red_pixel_count=red_pixel_count,
        overlaps_walls=overlaps_walls,
        touches_start=touches_start,
        touches_goal=touches_goal,
        connected=connected,
        success=success,
        message=message,
        evaluation_path=evaluation_path,
    )


def _iter_attempts(roots: Sequence[Path]) -> List[MazeAttempt]:
    attempts: List[MazeAttempt] = []
    for root in roots:
        if not root.exists() or not root.is_dir():
            continue
        for evaluation_path in sorted(root.rglob("evaluation.json")):
            record = _parse_attempt(evaluation_path)
            if record is not None:
                attempts.append(record)
    return attempts


def _group_by_puzzle(records: Iterable[MazeAttempt]) -> Dict[str, List[MazeAttempt]]:
    grouped: Dict[str, List[MazeAttempt]] = defaultdict(list)
    for record in records:
        grouped[record.puzzle_id].append(record)
    return grouped


def _group_by_type(records: Iterable[MazeAttempt]) -> Dict[str, List[MazeAttempt]]:
    grouped: Dict[str, List[MazeAttempt]] = defaultdict(list)
    for record in records:
        grouped[record.puzzle_type].append(record)
    return grouped


def _average(values: Sequence[int]) -> float:
    if not values:
        return 0.0
    return float(sum(values)) / float(len(values))


def _print_type_summary(records: Sequence[MazeAttempt]) -> None:
    grouped = _group_by_type(records)
    if not grouped:
        return
    print("Performance by puzzle type:")
    for puzzle_type in sorted(grouped.keys()):
        attempts = grouped[puzzle_type]
        total = len(attempts)
        success = sum(1 for record in attempts if record.success)
        rate = (float(success) / float(total)) if total else 0.0
        print(f"  {puzzle_type}: {success}/{total} successful ({rate:.0%})")
    print()


def _print_flag_breakdown(records: Sequence[MazeAttempt]) -> None:
    total = len(records)
    overlaps = sum(1 for record in records if record.overlaps_walls is True)
    misses_start = sum(1 for record in records if record.touches_start is False)
    misses_goal = sum(1 for record in records if record.touches_goal is False)
    disconnected = sum(1 for record in records if record.connected is False)
    unknown_start = sum(1 for record in records if record.touches_start is None)
    unknown_goal = sum(1 for record in records if record.touches_goal is None)
    unknown_connected = sum(1 for record in records if record.connected is None)
    zero_red = sum(1 for record in records if record.red_pixel_count == 0)

    print("Constraint diagnostics:")
    print(f"  Overlaps walls: {overlaps}/{total}")
    print(f"  Misses start: {misses_start}/{total}")
    print(f"  Misses goal: {misses_goal}/{total}")
    print(f"  Not connected: {disconnected}/{total}")
    if zero_red:
        print(f"  No red path detected: {zero_red}/{total}")
    if unknown_start or unknown_goal or unknown_connected:
        print(
            "  Incomplete flags: "
            f"start? {unknown_start}, goal? {unknown_goal}, connected? {unknown_connected}"
        )
    print()


def _print_red_pixel_stats(records: Sequence[MazeAttempt]) -> None:
    overall: List[int] = [record.red_pixel_count for record in records if record.red_pixel_count is not None]
    success_values: List[int] = [
        record.red_pixel_count
        for record in records
        if record.red_pixel_count is not None and record.success
    ]
    failure_values: List[int] = [
        record.red_pixel_count
        for record in records
        if record.red_pixel_count is not None and not record.success
    ]
    if not overall:
        return
    print("Red pixel counts (average):")
    print(f"  Overall: {_average(overall):.1f}")
    if success_values:
        print(f"  Successful: {_average(success_values):.1f}")
    if failure_values:
        print(f"  Unsuccessful: {_average(failure_values):.1f}")
    print()


def _print_failure_reasons(records: Sequence[MazeAttempt]) -> None:
    counter = Counter()
    for record in records:
        if not record.success:
            label = record.message if record.message else "(no message)"
            counter[label] += 1
    if not counter:
        print("No maze failures detected across the scanned evaluations.")
        print()
        return
    print("Failure reasons:")
    for message, count in counter.most_common():
        print(f"  {message}: {count}")
    print()


def _print_top_failures(records: Sequence[MazeAttempt], limit: int) -> None:
    if limit <= 0:
        return
    grouped = _group_by_puzzle(records)
    worst: List[tuple[int, int, str, str]] = []
    for puzzle_id, attempts in grouped.items():
        total = len(attempts)
        success = sum(1 for record in attempts if record.success)
        failure = total - success
        if failure <= 0:
            continue
        puzzle_type = attempts[0].puzzle_type
        worst.append((failure, total, puzzle_type, puzzle_id))
    if not worst:
        return
    worst.sort(reverse=True)
    trimmed = worst[:limit]
    print("Most challenging puzzles:")
    for failure, total, puzzle_type, puzzle_id in trimmed:
        success = total - failure
        rate = (float(success) / float(total)) if total else 0.0
        print(f"  {puzzle_type} {puzzle_id}: {success}/{total} successful ({rate:.0%})")
    print()


def summarize_maze_attempts(roots: Sequence[Path], top_failures: int) -> bool:
    attempts = _iter_attempts(roots)
    if not attempts:
        joined_roots = ", ".join(path.as_posix() for path in roots)
        print(f"No maze evaluations found under: {joined_roots}.")
        return False

    grouped_by_puzzle = _group_by_puzzle(attempts)
    unique_puzzles = len(grouped_by_puzzle)
    solved_puzzles = sum(1 for attempts_for_puzzle in grouped_by_puzzle.values() if any(record.success for record in attempts_for_puzzle))
    fully_solved = sum(1 for attempts_for_puzzle in grouped_by_puzzle.values() if attempts_for_puzzle and all(record.success for record in attempts_for_puzzle))

    total_attempts = len(attempts)
    successful_attempts = sum(1 for record in attempts if record.success)
    accuracy = (float(successful_attempts) / float(total_attempts)) if total_attempts else 0.0

    print("Maze evaluation summary")
    print(f"Roots: {', '.join(path.as_posix() for path in roots)}")
    print(f"Total attempts: {total_attempts}")
    print(f"Successful attempts: {successful_attempts} ({accuracy:.0%})")
    print(f"Unique puzzles: {unique_puzzles}")
    print(f"Puzzles solved at least once: {solved_puzzles}/{unique_puzzles}")
    print(f"Puzzles solved in every attempt: {fully_solved}/{unique_puzzles}")
    print()

    _print_type_summary(attempts)
    _print_flag_breakdown(attempts)
    _print_red_pixel_stats(attempts)
    _print_failure_reasons(attempts)
    _print_top_failures(attempts, limit=top_failures)

    return True


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize maze puzzle evaluation results.")
    parser.add_argument(
        "--root",
        dest="roots",
        type=Path,
        action="append",
        help="Directory to scan recursively for evaluation.json files (can be repeated). Defaults to data/output.",
    )
    parser.add_argument(
        "--top-failures",
        type=int,
        default=5,
        help="Number of lowest-performing puzzles to list (default: 5)",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    roots = args.roots if args.roots else [DEFAULT_EVALUATION_ROOT]
    summarize_maze_attempts(roots, top_failures=max(0, args.top_failures))


if __name__ == "__main__":
    main()
