"""Voting utilities for Sudoku puzzles."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from puzzle.base import EvaluationPayloadReader, AbstractVoteSummarizer

Position = Tuple[int, int]


_reader = EvaluationPayloadReader()


def load_attempt(attempt_dir: Path) -> Optional[Dict[str, Any]]:
    """Parse a Sudoku evaluation attempt and return non-clue cell predictions."""

    inner_payload = _reader.read_inner_payload(attempt_dir)
    if inner_payload is None:
        return None

    cell_breakdown = inner_payload.get("cell_breakdown", [])
    non_clue_cells: Dict[Position, Dict[str, Any]] = {}
    for cell in cell_breakdown:
        if cell.get("is_clue"):
            continue
        row = cell.get("row")
        col = cell.get("col")
        if row is None or col is None:
            continue
        position = (row, col)
        non_clue_cells[position] = {
            "expected": cell.get("expected"),
            "predicted": cell.get("predicted"),
            "is_correct": cell.get("is_correct"),
        }

    if not non_clue_cells:
        return None

    return {
        "puzzle_id": inner_payload.get("puzzle_id"),
        "cells": non_clue_cells,
    }


def format_prediction(value: Optional[int]) -> str:
    return "-" if value is None else str(value)


def _majority_vote(counter: Counter) -> Optional[Any]:
    if not counter:
        return None
    top_value = None
    top_count = 0
    tie = False
    for value, count in counter.items():
        if count > top_count:
            top_value = value
            top_count = count
            tie = False
        elif count == top_count:
            tie = True
    if top_count == 0 or tie:
        return None
    return top_value


def summarize_votes(vote_root: Path) -> bool:
    puzzle_dirs = sorted(
        path for path in vote_root.iterdir() if path.is_dir() and path.name.startswith("sudoku_")
    )

    if not puzzle_dirs:
        print("No sudoku vote outputs found.")
        return False

    total_vote_correct = 0
    total_vote_positions = 0
    total_attempt_correct = 0
    total_attempt_positions = 0

    group_vote_correct = defaultdict(int)
    group_vote_positions = defaultdict(int)
    group_attempt_correct = defaultdict(int)
    group_attempt_positions = defaultdict(int)
    group_puzzle_counts = defaultdict(int)

    for puzzle_dir in puzzle_dirs:
        attempts = sorted(
            path for path in puzzle_dir.iterdir() if path.is_dir() and path.name.startswith("attempt_")
        )

        attempt_payloads: Dict[str, Dict[str, Any]] = {}
        expected_lookup: Dict[Position, int] = {}

        for attempt_dir in attempts:
            payload = load_attempt(attempt_dir)
            if payload is None:
                continue
            attempt_payloads[attempt_dir.name] = payload
            for position, cell in payload["cells"].items():
                if position not in expected_lookup and cell.get("expected") is not None:
                    expected_lookup[position] = cell["expected"]

        if not attempt_payloads:
            continue

        puzzle_id = next(iter(attempt_payloads.values())).get("puzzle_id") or puzzle_dir.name
        positions = sorted(expected_lookup.keys())

        vote_tallies: Dict[Position, Counter] = defaultdict(Counter)
        per_attempt_correct: Dict[str, float] = {}

        for attempt_name, payload in attempt_payloads.items():
            cells = payload["cells"]
            correct = 0
            total = len(positions)
            for position in positions:
                cell = cells.get(position)
                predicted = cell.get("predicted") if cell else None
                if predicted is not None:
                    vote_tallies[position][predicted] += 1
                expected = expected_lookup.get(position)
                if cell and predicted == expected:
                    correct += 1
            rate = correct / total if total else 0.0
            per_attempt_correct[attempt_name] = rate
            if total:
                group_attempt_correct[total] += correct
                group_attempt_positions[total] += total
                total_attempt_correct += correct
                total_attempt_positions += total

        vote_results: Dict[Position, Dict[str, Any]] = {}
        vote_correct = 0
        total_positions = len(positions)

        for position in positions:
            tally = vote_tallies.get(position, Counter())
            vote_choice = _majority_vote(tally)
            expected_value = expected_lookup.get(position)
            if vote_choice == expected_value:
                vote_correct += 1
            vote_results[position] = {
                "choice": vote_choice,
                "expected": expected_value,
                "votes": dict(tally),
            }

        if total_positions:
            total_vote_correct += vote_correct
            total_vote_positions += total_positions
            group_vote_correct[total_positions] += vote_correct
            group_vote_positions[total_positions] += total_positions
            group_puzzle_counts[total_positions] += 1

        vote_rate = vote_correct / total_positions if total_positions else 0.0

        print(f"Puzzle: {puzzle_id}")
        print(f"  Attempts evaluated: {len(attempt_payloads)}")
        print(f"  Non-clue cells: {total_positions}")
        print("  Cells:")
        for position in positions:
            row, col = position
            expected_value = expected_lookup.get(position)
            vote_info = vote_results[position]
            vote_choice = format_prediction(vote_info["choice"])
            votes_display = ", ".join(
                f"{value}:{count}" for value, count in sorted(vote_info["votes"].items())
            ) or "no votes"
            print(
                f"    (r{row}, c{col}) expected {expected_value} | vote {vote_choice} | votes [{votes_display}]"
            )

        print("  Attempts:")
        for attempt_name in sorted(per_attempt_correct.keys()):
            payload = attempt_payloads[attempt_name]
            cells = payload["cells"]
            entries = []
            for position in positions:
                cell = cells.get(position)
                predicted = format_prediction(cell.get("predicted") if cell else None)
                entries.append(f"(r{position[0]},c{position[1]})={predicted}")
            rate = per_attempt_correct[attempt_name]
            print(f"    {attempt_name}: {'; '.join(entries)} | correct rate {rate:.0%}")

        print(f"  Vote correct rate: {vote_rate:.0%}")
        print("  Individual correct rates:")
        for attempt_name in sorted(per_attempt_correct.keys()):
            print(f"    {attempt_name}: {per_attempt_correct[attempt_name]:.0%}")
        print()

    if group_puzzle_counts:
        print("Grouped results by non-clue cell count:")
        for cell_count in sorted(group_puzzle_counts.keys()):
            puzzles = group_puzzle_counts[cell_count]
            vote_total = group_vote_positions.get(cell_count, 0)
            attempt_total = group_attempt_positions.get(cell_count, 0)
            vote_rate = (
                group_vote_correct[cell_count] / vote_total if vote_total else 0.0
            )
            attempt_rate = (
                group_attempt_correct[cell_count] / attempt_total if attempt_total else 0.0
            )
            print(
                f"  Cells {cell_count}: puzzles {puzzles}, vote correct rate {vote_rate:.0%}, "
                f"attempt correct rate {attempt_rate:.0%}"
            )
        print()

    if total_attempt_positions:
        overall_attempt_rate = total_attempt_correct / total_attempt_positions
    else:
        overall_attempt_rate = 0.0

    if total_vote_positions:
        overall_vote_rate = total_vote_correct / total_vote_positions
    else:
        overall_vote_rate = 0.0

    print("Overall attempt correct rate: {:.0%}".format(overall_attempt_rate))
    print("Overall vote correct rate: {:.0%}".format(overall_vote_rate))

    return True


__all__ = [
    "load_attempt",
    "format_prediction",
    "summarize_votes",
    "SudokuVoteSummarizer",
]


class SudokuVoteSummarizer(AbstractVoteSummarizer):
    """Summarizer implementation for Sudoku vote outputs."""

    def summarize(self, vote_root: Path, *, prefix_newline: bool = False) -> bool:
        # Sudoku summaries always print; ignore prefix_newline for compatibility
        return summarize_votes(vote_root)
