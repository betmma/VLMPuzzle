﻿"""ARC-AGI puzzle generator that renders example/test grids into composite images."""

from __future__ import annotations

import argparse
import json
import math
import random
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw

from ..base import AbstractPuzzleGenerator, PathLike

# ARC canonical palette (0-9) mapped to RGB triples.
ARC_PALETTE: Dict[int, Tuple[int, int, int]] = {
    0: (0, 0, 0),
    1: (0, 0, 255),
    2: (255, 0, 0),
    3: (0, 255, 0),
    4: (255, 255, 0),
    5: (128, 128, 128),
    6: (255, 0, 255),
    7: (255, 128, 0),
    8: (0, 255, 255),
    9: (128, 64, 0),
}

BACKGROUND_COLOR = (245, 245, 245)
GRID_BACKGROUND = (255, 255, 255)
GRID_LINE_COLOR = (15, 15, 15)
ARROW_COLOR = (32, 32, 32)


@dataclass
class GridPlacement:
    """Metadata describing where a grid is rendered in the composite image."""

    kind: str
    bbox: Tuple[int, int, int, int]
    rows: int
    cols: int

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "bbox": list(self.bbox),
            "rows": self.rows,
            "cols": self.cols,
        }


@dataclass
class RowLayout:
    """Layout definition for a single example/test row."""

    kind: str
    input_grid: List[List[int]]
    output_grid: List[List[int]]
    input_pos: Tuple[int, int]
    output_pos: Tuple[int, int]
    input_size: Tuple[int, int]
    output_size: Tuple[int, int]
    arrow_start: Tuple[int, int]
    arrow_end: Tuple[int, int]
    arrow_head: int


@dataclass
class ArcPuzzleLayout:
    size: Tuple[int, int]
    rows: List[RowLayout]


@dataclass
class ArcPuzzleRecord:
    id: str
    task_id: str
    task_path: str
    prompt: str
    puzzle_image_path: str
    solution_image_path: str
    cell_size: int
    placements: List[GridPlacement]
    test_rows: int
    test_cols: int
    test_output: List[List[int]]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "task_id": self.task_id,
            "task_path": self.task_path,
            "prompt": self.prompt,
            "puzzle_image_path": self.puzzle_image_path,
            "solution_image_path": self.solution_image_path,
            "cell_size": self.cell_size,
            "placements": [placement.to_dict() for placement in self.placements],
            "test_rows": self.test_rows,
            "test_cols": self.test_cols,
            "test_output": self.test_output,
        }


class ArcPuzzleGenerator(AbstractPuzzleGenerator[ArcPuzzleRecord]):
    """Generate ARC-AGI composite puzzles from task JSON files."""

    def __init__(
        self,
        *,
        dataset_dir: PathLike = "data/training",
        output_dir: PathLike = "data/arcagi",
        cell_size: int = 32,
        prompt: str = "Study the solved examples and produce the correct output for the test input.",
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(output_dir)
        self.dataset_dir = Path(dataset_dir)
        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.dataset_dir}")
        self.cell_size = cell_size
        self.prompt = prompt
        self._rng = random.Random(seed)

        self.puzzle_dir = self.output_dir / "puzzles"
        self.solution_dir = self.output_dir / "solutions"
        for directory in (self.puzzle_dir, self.solution_dir):
            directory.mkdir(parents=True, exist_ok=True)

        self._task_paths = self._discover_tasks()
        if not self._task_paths:
            raise ValueError(f"No ARC task JSON files found under {self.dataset_dir}")

    def _discover_tasks(self) -> List[Path]:
        return sorted(self.dataset_dir.rglob("*.json"))

    def create_puzzle(
        self,
        *,
        task_path: Optional[PathLike] = None,
        puzzle_id: Optional[str] = None,
    ) -> ArcPuzzleRecord:
        task_file = self._resolve_task_path(task_path)
        task_data = json.loads(task_file.read_text(encoding="utf-8"))
        train_pairs = task_data.get("train") or []
        test_pairs = task_data.get("test") or []
        if not test_pairs:
            raise ValueError(f"Task {task_file} does not include any test pairs")

        test_pair = test_pairs[0]
        layout, placements = self._build_layout(train_pairs, test_pair)

        record_id = puzzle_id or f"{task_file.stem}-{uuid.uuid4().hex[:8]}"
        puzzle_image = self._render_layout(layout, include_test_solution=False)
        solution_image = self._render_layout(layout, include_test_solution=True)

        puzzle_path = self.puzzle_dir / f"{record_id}_puzzle.png"
        solution_path = self.solution_dir / f"{record_id}_solution.png"
        puzzle_image.save(puzzle_path)
        solution_image.save(solution_path)

        task_rel_path = task_file.relative_to(self.dataset_dir).as_posix()
        test_output_grid: List[List[int]] = test_pair["output"]
        test_rows = len(test_output_grid)
        test_cols = len(test_output_grid[0]) if test_rows else 0

        return ArcPuzzleRecord(
            id=record_id,
            task_id=task_file.stem,
            task_path=task_rel_path,
            prompt=self.prompt,
            puzzle_image_path=self.relativize_path(puzzle_path),
            solution_image_path=self.relativize_path(solution_path),
            cell_size=self.cell_size,
            placements=placements,
            test_rows=test_rows,
            test_cols=test_cols,
            test_output=test_output_grid,
        )

    def _resolve_task_path(self, task_path: Optional[PathLike]) -> Path:
        if task_path is None:
            return self._rng.choice(self._task_paths)
        candidate = Path(task_path)
        if not candidate.is_absolute():
            candidate = self.dataset_dir / candidate
        if not candidate.exists():
            raise FileNotFoundError(f"Task file not found: {candidate}")
        return candidate

    def _build_layout(
        self,
        train_pairs: Sequence[Dict[str, List[List[int]]]],
        test_pair: Dict[str, List[List[int]]],
    ) -> Tuple[ArcPuzzleLayout, List[GridPlacement]]:
        pairs: List[Tuple[str, List[List[int]], List[List[int]]]] = []
        for idx, pair in enumerate(train_pairs):
            pairs.append((f"train_{idx}", pair["input"], pair["output"]))
        pairs.append(("test", test_pair["input"], test_pair["output"]))

        margin = max(24, self.cell_size // 2)
        gap = max(72, self.cell_size * 2)
        row_spacing = max(36, self.cell_size)

        row_layouts: List[RowLayout] = []
        placements: List[GridPlacement] = []
        row_widths: List[int] = []
        row_heights: List[int] = []

        for kind, input_grid, output_grid in pairs:
            input_rows, input_cols = self._grid_shape(input_grid)
            output_rows, output_cols = self._grid_shape(output_grid)
            input_size = (input_cols * self.cell_size, input_rows * self.cell_size)
            output_size = (output_cols * self.cell_size, output_rows * self.cell_size)
            row_height = max(input_size[1], output_size[1])
            row_width = input_size[0] + gap + output_size[0]
            row_widths.append(row_width)
            row_heights.append(row_height)
            row_layouts.append(
                RowLayout(
                    kind=kind,
                    input_grid=input_grid,
                    output_grid=output_grid,
                    input_pos=(0, 0),  # filled later
                    output_pos=(0, 0),
                    input_size=input_size,
                    output_size=output_size,
                    arrow_start=(0, 0),
                    arrow_end=(0, 0),
                    arrow_head=max(8, self.cell_size // 2),
                )
            )

        if not row_layouts:
            raise ValueError("At least one train pair is required to build a layout")

        max_row_width = max(row_widths)
        canvas_width = max_row_width + 2 * margin
        y_cursor = margin

        for idx, layout_row in enumerate(row_layouts):
            row_height = row_heights[idx]
            row_width = row_widths[idx]
            x_offset = margin + (max_row_width - row_width) // 2

            in_width, in_height = layout_row.input_size
            out_width, out_height = layout_row.output_size
            input_top = y_cursor + (row_height - in_height) // 2
            output_top = y_cursor + (row_height - out_height) // 2

            input_pos = (x_offset, input_top)
            output_pos = (x_offset + in_width + gap, output_top)

            center_y = y_cursor + row_height // 2
            arrow_start = (input_pos[0] + in_width + gap // 4, center_y)
            arrow_end = (output_pos[0] - gap // 4, center_y)
            arrow_head = max(8, min(layout_row.arrow_head, (arrow_end[0] - arrow_start[0]) // 2))

            layout_row.input_pos = input_pos
            layout_row.output_pos = output_pos
            layout_row.arrow_start = arrow_start
            layout_row.arrow_end = arrow_end
            layout_row.arrow_head = arrow_head

            placements.extend(
                [
                    GridPlacement(
                        kind=f"{layout_row.kind}_input",
                        bbox=(input_pos[0], input_pos[1], input_pos[0] + in_width, input_pos[1] + in_height),
                        rows=self._grid_shape(layout_row.input_grid)[0],
                        cols=self._grid_shape(layout_row.input_grid)[1],
                    ),
                    GridPlacement(
                        kind=f"{layout_row.kind}_output",
                        bbox=(output_pos[0], output_pos[1], output_pos[0] + out_width, output_pos[1] + out_height),
                        rows=self._grid_shape(layout_row.output_grid)[0],
                        cols=self._grid_shape(layout_row.output_grid)[1],
                    ),
                ]
            )

            y_cursor += row_height + row_spacing

        canvas_height = y_cursor - row_spacing + margin
        layout = ArcPuzzleLayout(size=(canvas_width, canvas_height), rows=row_layouts)
        return layout, placements

    def _render_layout(self, layout: ArcPuzzleLayout, *, include_test_solution: bool) -> Image.Image:
        image = Image.new("RGB", layout.size, BACKGROUND_COLOR)
        draw = ImageDraw.Draw(image)

        for row in layout.rows:
            input_image = self._render_grid(row.input_grid)
            image.paste(input_image, row.input_pos)

            is_test_row = row.kind == "test"
            if is_test_row and not include_test_solution:
                output_image = self._render_grid(None, rows=row.output_size[1] // self.cell_size, cols=row.output_size[0] // self.cell_size)
            else:
                output_image = self._render_grid(row.output_grid)
            image.paste(output_image, row.output_pos)

            self._draw_arrow(draw, row.arrow_start, row.arrow_end, head=row.arrow_head)

        return image

    def _render_grid(
        self,
        grid: Optional[Sequence[Sequence[int]]],
        *,
        rows: Optional[int] = None,
        cols: Optional[int] = None,
    ) -> Image.Image:
        if grid is not None:
            inferred_rows, inferred_cols = self._grid_shape(grid)
        else:
            if rows is None or cols is None:
                raise ValueError("rows and cols must be provided when grid is None")
            inferred_rows, inferred_cols = rows, cols

        width = inferred_cols * self.cell_size
        height = inferred_rows * self.cell_size
        image = Image.new("RGB", (width, height), GRID_BACKGROUND)
        draw = ImageDraw.Draw(image)

        if grid is not None:
            for r in range(inferred_rows):
                row_values = grid[r]
                for c in range(inferred_cols):
                    value = int(row_values[c])
                    color = ARC_PALETTE.get(value, ARC_PALETTE[0])
                    x0 = c * self.cell_size
                    y0 = r * self.cell_size
                    x1 = x0 + self.cell_size
                    y1 = y0 + self.cell_size
                    draw.rectangle((x0 + 1, y0 + 1, x1 - 1, y1 - 1), fill=color)

        for r in range(inferred_rows + 1):
            y = r * self.cell_size
            draw.line((0, y, width, y), fill=GRID_LINE_COLOR, width=1)
        for c in range(inferred_cols + 1):
            x = c * self.cell_size
            draw.line((x, 0, x, height), fill=GRID_LINE_COLOR, width=1)

        return image

    def _draw_arrow(self, draw: ImageDraw.ImageDraw, start: Tuple[int, int], end: Tuple[int, int], *, head: int) -> None:
        draw.line((start[0], start[1], end[0], end[1]), fill=ARROW_COLOR, width=3)
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = math.hypot(dx, dy)
        if length == 0:
            return
        ux = dx / length
        uy = dy / length
        px = -uy
        py = ux
        head_length = head
        head_width = head * 0.6
        tip = end
        left = (
            int(tip[0] - ux * head_length + px * head_width),
            int(tip[1] - uy * head_length + py * head_width),
        )
        right = (
            int(tip[0] - ux * head_length - px * head_width),
            int(tip[1] - uy * head_length - py * head_width),
        )
        draw.polygon([tip, left, right], fill=ARROW_COLOR)

    @staticmethod
    def _grid_shape(grid: Sequence[Sequence[int]]) -> Tuple[int, int]:
        rows = len(grid)
        cols = len(grid[0]) if rows else 0
        return rows, cols

    def create_random_puzzle(self) -> ArcPuzzleRecord:
        return self.create_puzzle()


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ARC-AGI composite puzzles")
    parser.add_argument("count", type=int, help="Number of puzzles to generate")
    parser.add_argument("--dataset", type=Path, default=Path("data/training"), help="Directory containing ARC JSON tasks")
    parser.add_argument("--output-dir", type=Path, default=Path("data/arcagi"), help="Directory to save puzzle assets")
    parser.add_argument("--cell-size", type=int, default=32, help="Pixel size of each grid cell")
    parser.add_argument("--prompt", type=str, default="Study the solved examples and produce the correct output for the test input.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--metadata", type=Path, default=None, help="Optional path to store metadata JSON")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)
    generator = ArcPuzzleGenerator(
        dataset_dir=args.dataset,
        output_dir=args.output_dir,
        cell_size=args.cell_size,
        prompt=args.prompt,
        seed=args.seed,
    )
    metadata_path = args.metadata or (generator.output_dir / "puzzles.json")
    records = generator.generate_dataset(args.count, metadata_path=metadata_path, append=True)
    print(f"Generated {len(records)} ARC puzzles -> {metadata_path}")


if __name__ == "__main__":
    main()
