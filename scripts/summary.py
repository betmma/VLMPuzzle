"""Aggregate evaluation votes for Sudoku and mirror puzzles."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from puzzle.sudoku.vote import summarize_votes as summarize_sudoku_votes
from puzzle.mirror.vote import summarize_monochrome_votes
from puzzle.rects.vote import summarize_color_order_votes
from scripts import multiple_choice_summary as mc_summary

DEFAULT_VOTE_ROOT = REPO_ROOT / "data" / "voteOutput"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data" / "output"


def main() -> None:
    output_root = DEFAULT_OUTPUT_ROOT
    vote_root = DEFAULT_VOTE_ROOT
    mc_summary.summarize_all(vote_root, top_misses=5)
    # processed_sudoku = summarize_sudoku_votes(vote_root)
    # summarize_monochrome_votes(vote_root, prefix_newline=processed_sudoku)
    # summarize_color_order_votes(vote_root)


if __name__ == "__main__":
    main()

