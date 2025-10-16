"""Voting utilities for rectangles-order puzzles."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from puzzle.base import EvaluationPayloadReader, AbstractVoteSummarizer

VOTE_SPOKEN=False


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
    spoken_rgb = inner.get("spoken_color_rgb") or []
    expected_keys: List[str] = []
    for c in expected:
        key = _color_key(c)
        if key is not None:
            expected_keys.append(key)
    predicted_keys: List[Optional[str]] = []
    for c in predicted:
        key = _color_key(c)
        predicted_keys.append(key)
    spoken_keys: List[Optional[str]] = []
    for c in spoken_rgb:
        key = _color_key(c)
        spoken_keys.append(key)
    if not expected_keys:
        return None
    effective_predicted = predicted_keys
    if not any(effective_predicted) and any(spoken_keys):
        effective_predicted = spoken_keys
    return {
        "puzzle_id": inner.get("puzzle_id"),
        "expected": expected_keys,
        "predicted": effective_predicted,
        "spoken": spoken_keys,
    }


def summarize_color_order_votes(vote_root: Path) -> bool:
    puzzle_dirs = sorted(
        p for p in vote_root.iterdir() if p.is_dir() and p.name.startswith("rects_")
    )
    if not puzzle_dirs:
        print("No rects vote outputs found.")
        return False

    # Strict metrics: a puzzle/attempt counts as correct only if ALL positions match.
    total_vote_strict_correct = 0
    total_vote_puzzles = 0
    total_attempt_strict_correct = 0
    total_attempts = 0

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
            predicted = payload.get("spoken" if VOTE_SPOKEN else "predicted", [])
            # Tally predictions for downstream voted choice (ignore None).
            for i in positions:
                choice = predicted[i] if i < len(predicted) else None
                if choice is not None:
                    tallies[i][choice] += 1
            # Strict correctness: entire sequence matches exactly (length and content).
            is_strict_correct = (
                len(predicted) == len(expected)
                and all(
                    (predicted[i] if i < len(predicted) else None) == expected[i]
                    for i in positions
                )
            )
            per_attempt_correct[name] = 1.0 if is_strict_correct else 0.0
            total_attempt_strict_correct += 1 if is_strict_correct else 0
            total_attempts += 1

        vote_choice_seq: List[Optional[str]] = []
        for i in positions:
            tally = tallies.get(i, Counter())
            choice = max(tally.items(), key=lambda kv: kv[1])[0] if tally else None
            vote_choice_seq.append(choice)
        vote_all_correct = (
            len(vote_choice_seq) == len(expected)
            and all(vote_choice_seq[i] == expected[i] for i in positions)
        )
        total_vote_strict_correct += 1 if vote_all_correct else 0
        total_vote_puzzles += 1

        print(f"Puzzle: {puzzle_dir.name}")
        print(f"  Positions: {len(positions)}")
        print(f"  Vote correct rate: {'100%' if vote_all_correct else '0%'}")
        print("  Individual correct rates:")
        for name in sorted(per_attempt_correct.keys()):
            print(f"    {name}: {per_attempt_correct[name]:.0%}")
        print()
    print("Total attempts:", total_attempts)
    if total_attempts:
        print(
            "Overall attempt correct rate: {:.0%}".format(
                total_attempt_strict_correct / total_attempts
            )
        )
    else:
        print("Overall attempt correct rate: 0%")
    if total_vote_puzzles:
        print(
            "Overall vote correct rate: {:.0%}".format(
                total_vote_strict_correct / total_vote_puzzles
            )
        )
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
