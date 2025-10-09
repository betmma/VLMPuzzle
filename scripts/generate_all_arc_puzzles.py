#!/usr/bin/env python3
"""Generate ARC-AGI puzzles for every task and sort metadata by difficulty."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Iterable, List, Optional
from PIL import Image

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


def _pad_image_to_aspect(image_path: Path, target_aspect: float, *, color=(255, 255, 255)) -> bool:
    """Pad image on the right or bottom with a solid color to reach target aspect.

    Returns True if padding was applied, False if already at target or image missing.
    """
    if not image_path.exists():
        return False
    with Image.open(image_path) as im:
        w, h = im.size
        if w == 0 or h == 0:
            return False
        current = w / h
        # treat near-equality as equal to avoid tiny pads
        if math.isclose(current, target_aspect, rel_tol=0.0, abs_tol=1e-4):
            return False
        if current < target_aspect:
            # Need to increase width; pad on the right
            new_w = int(math.ceil(target_aspect * h))
            new_h = h
            pad_right = new_w - w
            if pad_right <= 0:
                return False
            canvas = Image.new("RGB", (new_w, new_h), color)
            canvas.paste(im, (0, 0))
        else:
            # Need to increase height; pad on the bottom
            new_w = w
            new_h = int(math.ceil(w / target_aspect))
            pad_bottom = new_h - h
            if pad_bottom <= 0:
                return False
            canvas = Image.new("RGB", (new_w, new_h), color)
            canvas.paste(im, (0, 0))
        canvas.save(image_path)
        return True


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
        # Optionally pad images to desired aspect ratio by extending canvas
        if args.aspect_ratio:
            puzzle_path = (generator.output_dir / record_dict["puzzle_image_path"]).resolve()
            solution_path = (generator.output_dir / record_dict["solution_image_path"]).resolve()
            padded_puzzle = _pad_image_to_aspect(puzzle_path, args.aspect_ratio)
            padded_solution = _pad_image_to_aspect(solution_path, args.aspect_ratio)
            if padded_puzzle or padded_solution:
                print(
                    f"    padded to aspect {args.aspect_ratio:.6g}: "
                    f"{padded_puzzle and 'puzzle ' or ''}{padded_solution and 'solution' or ''}"
                )
        records.append(record_dict)
        print(f"[{index}/{len(task_paths)}] generated {puzzle_id} (difficulty={difficulty:.2f})")

    records.sort(key=lambda item: (item["difficulty"], item["id"]))

    metadata_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"Wrote {len(records)} puzzles to {metadata_path}")


if __name__ == "__main__":
    main()
