#!/usr/bin/env python3
"""Generate ARC-AGI puzzles for every task and sort metadata by difficulty."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from puzzle.arcagi import ArcPuzzleGenerator


def _grid_cell_count(grid: Iterable[Iterable[int]]) -> int:
    rows = list(grid)
    if not rows:
        return 0
    return len(rows) * len(rows[0])


def _average_cells(task_payload: dict) -> float:
    counts: List[int] = []
    for pair in task_payload.get("train", []):
        counts.append(_grid_cell_count(pair.get("input", [])))
        counts.append(_grid_cell_count(pair.get("output", [])))
    test_pairs = task_payload.get("test", [])
    if test_pairs:
        first_test = test_pairs[0]
        counts.append(_grid_cell_count(first_test.get("input", [])))
        counts.append(_grid_cell_count(first_test.get("output", [])))
    if not counts:
        return 0.0
    return sum(counts) / len(counts)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/training"),
        help="Directory containing ARC task JSON files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/arcagi"),
        help="Directory to write puzzle assets",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="Optional path for the difficulty-sorted metadata JSON",
    )
    parser.add_argument(
        "--cell-size",
        type=int,
        default=32,
        help="Pixel size for an individual grid cell",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Study the solved examples and produce the correct output for the test input.",
        help="Prompt text stored with each puzzle record",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed (only used when sampling puzzles without explicit ids)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    dataset_dir = args.dataset.resolve()
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    generator = ArcPuzzleGenerator(
        dataset_dir=dataset_dir,
        output_dir=args.output_dir,
        cell_size=args.cell_size,
        prompt=args.prompt,
        seed=args.seed,
    )

    metadata_path = args.metadata or (generator.output_dir / "puzzles.json")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    task_paths = sorted(dataset_dir.rglob("*.json"))
    if not task_paths:
        raise ValueError(f"No ARC task files found in {dataset_dir}")

    records: List[dict] = []
    for index, task_path in enumerate(task_paths, start=1):
        task_payload = json.loads(task_path.read_text(encoding="utf-8"))
        difficulty = _average_cells(task_payload)
        puzzle_id = task_path.stem
        record = generator.create_puzzle(task_path=task_path, puzzle_id=puzzle_id)
        record_dict = record.to_dict()
        record_dict["difficulty"] = difficulty
        records.append(record_dict)
        print(f"[{index}/{len(task_paths)}] generated {puzzle_id} (difficulty={difficulty:.2f})")

    records.sort(key=lambda item: (item["difficulty"], item["id"]))

    metadata_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"Wrote {len(records)} puzzles to {metadata_path}")


if __name__ == "__main__":
    main()
