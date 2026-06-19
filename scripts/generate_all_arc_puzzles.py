#!/usr/bin/env python3
"""Generate ARC-AGI puzzles for every task and sort metadata by difficulty."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path
from typing import Iterable, List, Optional

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


def _parse_ratio(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        if ":" in text:
            left, right = text.split(":", 1)
            a = float(left)
            b = float(right)
            if b == 0:
                raise ValueError
            return a / b
        # allow plain float like 1.7778
        ratio = float(text)
        if ratio <= 0:
            raise ValueError
        return ratio
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid aspect ratio '{value}'. Use W:H (e.g. 16:9) or a positive float."
        )


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
        default="Each row contains input and output grids. Learn the pattern and generate the output grid for the last input while keeping existing patterns without modification. Static camera perspective, no zoom or pan. In portrait.",
        help="Prompt text stored with each puzzle record",
    )
    parser.add_argument(
        "--aspect-ratio",
        type=_parse_ratio,
        default=None,
        help="Optional aspect ratio for output images, e.g. 16:9 or 1.7778. Adds white padding on right/bottom to fit.",
    )
    parser.add_argument(
        "--canvas-width",
        type=int,
        default=None,
        help="Optional width to force resize the image to.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed (only used when sampling puzzles without explicit ids)",
    )
    parser.add_argument(
        "--split",
        type=int,
        default=0,
        help="Split training examples to create more puzzles",
    )
    parser.add_argument(
        "--video",
        action="store_true",
        help="Generate video of the solution being drawn",
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
        aspect=args.aspect_ratio,
        canvas_width=args.canvas_width,
    )

    metadata_path = args.metadata or (generator.output_dir / "data.json")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    task_paths = sorted(dataset_dir.rglob("*.json"))
    if not task_paths:
        raise ValueError(f"No ARC task files found in {dataset_dir}")

    records: List[dict] = []
    for index, task_path in enumerate(task_paths, start=1):
        task_payload = json.loads(task_path.read_text(encoding="utf-8"))
        difficulty = _average_cells(task_payload)
        puzzle_id = task_path.stem

        tasks_to_generate = [(puzzle_id, None, None)]

        if args.split > 0:
            train_pairs = task_payload.get("train", [])
            indices = range(len(train_pairs))
            for s in range(1, args.split + 1):
                # result train list must include at least 2 examples
                if len(train_pairs) - s < 2:
                    break
                for removed_indices in itertools.combinations(indices, s):
                    context_pairs = [
                        train_pairs[i] for i in indices if i not in removed_indices
                    ]
                    for target_idx in removed_indices:
                        target_pair = train_pairs[target_idx]
                        removed_str = "-".join(map(str, removed_indices))
                        sub_id = f"{puzzle_id}_s{s}_r{removed_str}_t{target_idx}"
                        tasks_to_generate.append((sub_id, context_pairs, target_pair))

        for pid, p_train, p_test in tasks_to_generate:
            record = generator.create_puzzle(
                task_path=task_path,
                puzzle_id=pid,
                train_pairs=p_train,
                test_pair=p_test,
                make_video=args.video,
            )
            record_dict = record.to_dict()
            record_dict["difficulty"] = difficulty
            records.append(record_dict)

        print(
            f"[{index}/{len(task_paths)}] generated {puzzle_id} (difficulty={difficulty:.2f}) + {len(tasks_to_generate)-1} splits"
        )

    # records.sort(key=lambda item: (item["difficulty"], item["id"]))

    metadata_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"Wrote {len(records)} puzzles to {metadata_path}")


if __name__ == "__main__":
    main()
