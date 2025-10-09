"""Voting utilities for rectangles-order puzzles."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from puzzle.base import EvaluationPayloadReader, AbstractVoteSummarizer


Color = Tuple[int, int, int]


_reader = EvaluationPayloadReader()


def _color_key(c: Optional[List[int]]) -> Optional[str]:
    if not c or len(c) != 3:
        return None
    r, g, b = (int(c[0]), int(c[1]), int(c[2]))
    return f"#{r:02x}{g:02x}{b:02x}"


def load_attempt(attempt_dir: Path) -> Optional[Dict[str, Any]]:
    """Parse an evaluation attempt and extract expected/predicted color orders."""

    inner = _reader.read_inner_payload(attempt_dir)
    if inner is None:
        return None
    expected = inner.get("expected_order") or []
    predicted = inner.get("predicted_order") or []
    expected_keys: List[str] = []
    for c in expected:
        key = _color_key(c)
        if key is not None:
            expected_keys.append(key)
    predicted_keys: List[Optional[str]] = []
    for c in predicted:
        key = _color_key(c)
        predicted_keys.append(key)
    if not expected_keys:
        return None
    return {
        "puzzle_id": inner.get("puzzle_id"),
        "expected": expected_keys,
        "predicted": predicted_keys,
    }


def summarize_color_order_votes(vote_root: Path) -> bool:
    puzzle_dirs = sorted(
        p for p in vote_root.iterdir() if p.is_dir() and p.name.startswith("rects_")
    )
    if not puzzle_dirs:
        print("No rects vote outputs found.")
        return False

    total_vote_correct = 0
    total_vote_positions = 0
    total_attempt_correct = 0
    total_attempt_positions = 0

    for puzzle_dir in puzzle_dirs:
        attempts = sorted(
            p for p in puzzle_dir.iterdir() if p.is_dir() and p.name.startswith("attempt_")
        )
        payloads: Dict[str, Dict[str, Any]] = {}
        for attempt_dir in attempts:
            payload = load_attempt(attempt_dir)
            if payload is None:
                continue
            payloads[attempt_dir.name] = payload
        if not payloads:
            continue

        expected = next(iter(payloads.values())).get("expected", [])
        positions = list(range(len(expected)))
        tallies: Dict[int, Counter] = defaultdict(Counter)
        per_attempt_correct: Dict[str, float] = {}

        for name, payload in payloads.items():
            predicted = payload.get("predicted", [])
            correct = 0
            total = len(positions)
            for i in positions:
                choice = predicted[i] if i < len(predicted) else None
                if choice is not None:
                    tallies[i][choice] += 1
                if choice == expected[i]:
                    correct += 1
            rate = correct / total if total else 0.0
            per_attempt_correct[name] = rate
            total_attempt_correct += correct
            total_attempt_positions += total

        vote_choice_seq: List[Optional[str]] = []
        vote_correct = 0
        for i in positions:
            tally = tallies.get(i, Counter())
            choice = max(tally.items(), key=lambda kv: kv[1])[0] if tally else None
            vote_choice_seq.append(choice)
            if choice == expected[i]:
                vote_correct += 1
        total_vote_correct += vote_correct
        total_vote_positions += len(positions)

        print(f"Puzzle: {puzzle_dir.name}")
        print(f"  Positions: {len(positions)}")
        print(f"  Vote correct rate: {vote_correct/len(positions):.0%}")
        print("  Individual correct rates:")
        for name in sorted(per_attempt_correct.keys()):
            print(f"    {name}: {per_attempt_correct[name]:.0%}")
        print()

    if total_attempt_positions:
        print("Overall attempt correct rate: {:.0%}".format(total_attempt_correct / total_attempt_positions))
    else:
        print("Overall attempt correct rate: 0%")
    if total_vote_positions:
        print("Overall vote correct rate: {:.0%}".format(total_vote_correct / total_vote_positions))
    else:
        print("Overall vote correct rate: 0%")
    return True


class RectsVoteSummarizer(AbstractVoteSummarizer):
    def summarize(self, vote_root: Path, *, prefix_newline: bool = False) -> bool:
        # This summarizer always prints; ignore prefix_newline for compatibility
        return summarize_color_order_votes(vote_root)


__all__ = [
    "load_attempt",
    "summarize_color_order_votes",
    "RectsVoteSummarizer",
]

