"""Voting utilities for mirror puzzles."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from puzzle.base import EvaluationPayloadReader, AbstractVoteSummarizer

Position = Tuple[int, int]
Color = Tuple[float, float, float]

WHITE_COLOR: Color = (255.0, 255.0, 255.0)
WHITE_LABEL = "white"
MONO_LABEL = "monochrome"


_reader = EvaluationPayloadReader()


def load_attempt(attempt_dir: Path) -> Optional[Dict[str, Any]]:
    """Parse a mirror evaluation attempt returning per-cell expected/actual colors."""

    inner_payload = _reader.read_inner_payload(attempt_dir)
    if inner_payload is None:
        return None

    cell_breakdown = inner_payload.get("cell_breakdown", [])
    cells: Dict[Position, Dict[str, Optional[Color]]] = {}
    for cell in cell_breakdown:
        row = cell.get("row")
        col = cell.get("col")
        if row is None or col is None:
            continue
        expected_list = cell.get("expected_color")
        actual_list = cell.get("actual_color")
        expected_color: Optional[Color]
        actual_color: Optional[Color]
        if isinstance(expected_list, (list, tuple)) and len(expected_list) == 3:
            expected_color = tuple(float(value) for value in expected_list)
        else:
            expected_color = None
        if isinstance(actual_list, (list, tuple)) and len(actual_list) == 3:
            actual_color = tuple(float(value) for value in actual_list)
        else:
            actual_color = None
        cells[(row, col)] = {
            "expected_color": expected_color,
            "actual_color": actual_color,
        }

    if not cells:
        return None

    return {
        "puzzle_id": inner_payload.get("puzzle_id"),
        "cells": cells,
    }


def color_distance(color_a: Color, color_b: Color) -> float:
    return math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(color_a, color_b)))


def classify_monochrome_prediction(
    color: Optional[Color],
    monochrome_color: Color,
) -> Optional[str]:
    if color is None:
        return None
    distance_to_white = color_distance(color, WHITE_COLOR)
    distance_to_mono = color_distance(color, monochrome_color)
    return MONO_LABEL if distance_to_mono < distance_to_white else WHITE_LABEL


def load_metadata(data_root: Path) -> Dict[str, Dict[str, Any]]:
    metadata_path = data_root / "mirror" / "data.json"
    if not metadata_path.exists():
        return {}
    try:
        with metadata_path.open("r", encoding="utf-8") as handle:
            records = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}

    metadata: Dict[str, Dict[str, Any]] = {}
    for record in records:
        puzzle_id = record.get("id")
        if puzzle_id:
            metadata[str(puzzle_id)] = record
    return metadata


def extract_monochrome_color(record: Dict[str, Any]) -> Optional[Color]:
    colored_cells = record.get("colored_cells", [])
    for cell in colored_cells:
        color = cell.get("color")
        if isinstance(color, (list, tuple)) and len(color) == 3:
            return tuple(float(value) for value in color)
    return None


def format_color(color: Color) -> str:
    return "({:.0f}, {:.0f}, {:.0f})".format(*color)


def format_prediction(label: Optional[str]) -> str:
    if label == MONO_LABEL:
        return "color"
    if label == WHITE_LABEL:
        return "white"
    return "-"


def _majority_vote(counter: Counter) -> Optional[str]:
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


def summarize_monochrome_votes(
    vote_root: Path,
    *,
    prefix_newline: bool = False,
) -> bool:
    puzzle_dirs = sorted(
        path for path in vote_root.iterdir() if path.is_dir() and path.name.startswith("mirror_")
    )
    if not puzzle_dirs:
        return False

    metadata = load_metadata(vote_root.parent)

    output_lines = []
    diagnostics = []
    processed_puzzles = 0

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
        for attempt_dir in attempts:
            payload = load_attempt(attempt_dir)
            if payload is None:
                continue
            attempt_payloads[attempt_dir.name] = payload

        if not attempt_payloads:
            continue

        puzzle_id = next(iter(attempt_payloads.values())).get("puzzle_id") or puzzle_dir.name
        record = metadata.get(str(puzzle_id))
        if record is None:
            diagnostics.append(f"Skipping {puzzle_id}: puzzle metadata not found.")
            continue
        if not record.get("monochrome"):
            continue

        monochrome_color = extract_monochrome_color(record)
        if monochrome_color is None:
            diagnostics.append(f"Skipping {puzzle_id}: monochrome color unavailable.")
            continue

        expected_lookup: Dict[Position, str] = {}
        for payload in attempt_payloads.values():
            for position, cell in payload["cells"].items():
                if position in expected_lookup:
                    continue
                expected_label = classify_monochrome_prediction(
                    cell.get("expected_color"), monochrome_color
                )
                if expected_label is not None:
                    expected_lookup[position] = expected_label

        positions = sorted(expected_lookup.keys())
        if not positions:
            diagnostics.append(f"Skipping {puzzle_id}: no evaluable mirror cells found.")
            continue

        vote_tallies: Dict[Position, Counter] = defaultdict(Counter)
        per_attempt_correct: Dict[str, float] = {}

        for attempt_name, payload in attempt_payloads.items():
            cells = payload["cells"]
            correct = 0
            total = len(positions)
            for position in positions:
                cell = cells.get(position)
                predicted_label = classify_monochrome_prediction(
                    cell.get("actual_color") if cell else None,
                    monochrome_color,
                )
                if predicted_label is not None:
                    vote_tallies[position][predicted_label] += 1
                expected_label = expected_lookup.get(position)
                if (
                    cell
                    and expected_label is not None
                    and predicted_label == expected_label
                ):
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
            expected_label = expected_lookup.get(position)
            if vote_choice is not None and vote_choice == expected_label:
                vote_correct += 1
            vote_results[position] = {
                "choice": vote_choice,
                "expected": expected_label,
                "votes": dict(tally),
            }

        if total_positions:
            total_vote_correct += vote_correct
            total_vote_positions += total_positions
            group_vote_correct[total_positions] += vote_correct
            group_vote_positions[total_positions] += total_positions
            group_puzzle_counts[total_positions] += 1

        vote_rate = vote_correct / total_positions if total_positions else 0.0

        output_lines.append(
            f"Puzzle: {puzzle_id} (monochrome color {format_color(monochrome_color)})"
        )
        output_lines.append(f"  Attempts evaluated: {len(attempt_payloads)}")
        output_lines.append(f"  Mirror cells: {total_positions}")
        # output_lines.append("  Cells:")
        # for position in positions:
        #     row, col = position
        #     vote_info = vote_results[position]
        #     vote_choice = format_prediction(vote_info["choice"])
        #     expected_label = format_prediction(vote_info["expected"])
        #     votes_display = ", ".join(
        #         f"{format_prediction(value)}:{count}" for value, count in sorted(vote_info["votes"].items())
        #     ) or "no votes"
        #     output_lines.append(
        #         f"    (r{row}, c{col}) expected {expected_label} | vote {vote_choice} | votes [{votes_display}]"
        #     )

        # output_lines.append("  Attempts:")
        # for attempt_name in sorted(per_attempt_correct.keys()):
        #     payload = attempt_payloads[attempt_name]
        #     cells = payload["cells"]
        #     entries = []
        #     for position in positions:
        #         cell = cells.get(position)
        #         predicted_label = classify_monochrome_prediction(
        #             cell.get("actual_color") if cell else None,
        #             monochrome_color,
        #         )
        #         entries.append(
        #             f"(r{position[0]},c{position[1]})={format_prediction(predicted_label)}"
        #         )
        #     rate = per_attempt_correct[attempt_name]
        #     output_lines.append(
        #         f"    {attempt_name}: {'; '.join(entries)} | correct rate {rate:.0%}"
        #     )

        output_lines.append(f"  Vote correct rate: {vote_rate:.0%}")
        output_lines.append("  Individual correct rates:")
        for attempt_name in sorted(per_attempt_correct.keys()):
            output_lines.append(
                f"    {attempt_name}: {per_attempt_correct[attempt_name]:.0%}"
            )
        output_lines.append("")

        processed_puzzles += 1

    if processed_puzzles:
        if group_puzzle_counts:
            output_lines.append("Grouped results by mirror cell count:")
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
                output_lines.append(
                    f"  Cells {cell_count}: puzzles {puzzles}, vote correct rate {vote_rate:.0%}, "
                    f"attempt correct rate {attempt_rate:.0%}"
                )
            output_lines.append("")

        overall_attempt_rate = (
            total_attempt_correct / total_attempt_positions if total_attempt_positions else 0.0
        )
        overall_vote_rate = (
            total_vote_correct / total_vote_positions if total_vote_positions else 0.0
        )
        output_lines.append("Overall attempt correct rate: {:.0%}".format(overall_attempt_rate))
        output_lines.append("Overall vote correct rate: {:.0%}".format(overall_vote_rate))

    if output_lines:
        if prefix_newline:
            print()
        for line in output_lines:
            print(line)
        return True

    if diagnostics:
        if prefix_newline:
            print()
        for message in diagnostics:
            print(message)
        return False

    if puzzle_dirs:
        if prefix_newline:
            print()
        print("No monochrome mirror vote outputs found.")
    return False


__all__ = [
    "WHITE_COLOR",
    "WHITE_LABEL",
    "MONO_LABEL",
    "load_attempt",
    "classify_monochrome_prediction",
    "load_metadata",
    "extract_monochrome_color",
    "format_color",
    "format_prediction",
    "summarize_monochrome_votes",
    "MirrorVoteSummarizer",
]


class MirrorVoteSummarizer(AbstractVoteSummarizer):
    """Summarizer implementation for mirror puzzle vote outputs."""

    def summarize(self, vote_root: Path, *, prefix_newline: bool = False) -> bool:
        return summarize_monochrome_votes(vote_root, prefix_newline=prefix_newline)

