"""Mirror puzzle generator for left-right symmetry tasks."""

from __future__ import annotations

import argparse
import random
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw

from ..base import AbstractPuzzleGenerator, PathLike


@dataclass
class CellColor:
    row: int
    col: int
    color: Tuple[int, int, int]

    def to_dict(self) -> dict:
        return {
            "row": self.row,
            "col": self.col,
            "color": list(self.color),
        }


@dataclass
class MirrorPuzzleRecord:
    id: str
    prompt: str
    grid_size: Tuple[int, int]
    cell_size: int
    colored_cells: List[CellColor]
    puzzle_image_path: str
    solution_image_path: str
    monochrome: bool

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "prompt": self.prompt,
            "grid_size": list(self.grid_size),
            "cell_size": self.cell_size,
            "colored_cells": [cell.to_dict() for cell in self.colored_cells],
            "puzzle_image_path": self.puzzle_image_path,
            "solution_image_path": self.solution_image_path,
            "monochrome": self.monochrome,
        }


class MirrorGenerator(AbstractPuzzleGenerator[MirrorPuzzleRecord]):
    """Generate left-half colored grids with mirrored solutions."""

    def __init__(
        self,
        output_dir: PathLike = "data/mirror",
        *,
        rows: int = 6,
        cols: int = 8,
        cell_size: int = 48,
        prompt: str = "Mirror the colors from the left half onto the right half of the grid.",
        left_fill_ratio: float = 0.6,
        monochrome: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(output_dir)
        if cols % 2 != 0:
            raise ValueError("Column count must be even for mirroring")
        self.rows = rows
        self.cols = cols
        self.cell_size = cell_size
        self.prompt = prompt
        self.left_fill_ratio = max(0.1, min(left_fill_ratio, 1.0))
        self.monochrome = monochrome
        self._rng = random.Random(seed)

        self.puzzle_dir = self.output_dir / "puzzles"
        self.solution_dir = self.output_dir / "solutions"
        for path in (self.puzzle_dir, self.solution_dir):
            path.mkdir(parents=True, exist_ok=True)

    def create_puzzle(self, *, puzzle_id: Optional[str] = None) -> MirrorPuzzleRecord:
        puzzle_uuid = puzzle_id or str(uuid.uuid4())
        colored_cells = self._create_colored_cells()
        puzzle_image = self._render(colored_cells, mirror=False)
        solution_image = self._render(colored_cells, mirror=True)

        puzzle_path = self.puzzle_dir / f"{puzzle_uuid}_puzzle.png"
        solution_path = self.solution_dir / f"{puzzle_uuid}_solution.png"
        puzzle_image.save(puzzle_path)
        solution_image.save(solution_path)

        return MirrorPuzzleRecord(
            id=puzzle_uuid,
            prompt=self.prompt,
            grid_size=(self.rows, self.cols),
            cell_size=self.cell_size,
            colored_cells=colored_cells,
            puzzle_image_path=self.relativize_path(puzzle_path),
            solution_image_path=self.relativize_path(solution_path),
            monochrome=self.monochrome,
        )

    def create_random_puzzle(self) -> MirrorPuzzleRecord:
        return self.create_puzzle()

    def _create_colored_cells(self) -> List[CellColor]:
        half_cols = self.cols // 2
        total_left_cells = self.rows * half_cols
        target_filled = max(1, int(total_left_cells * self.left_fill_ratio))
        cells = [(r, c) for r in range(self.rows) for c in range(half_cols)]
        self._rng.shuffle(cells)
        colored: List[CellColor] = []
        base_color = tuple(self._rng.randint(32, 224) for _ in range(3)) if self.monochrome else None
        for r, c in cells[:target_filled]:
            if self.monochrome:
                color = base_color
            else:
                color = tuple(self._rng.randint(32, 224) for _ in range(3))
            colored.append(CellColor(row=r, col=c, color=color))
        return colored

    def _render(self, colored_cells: Sequence[CellColor], *, mirror: bool) -> Image.Image:
        width = self.cols * self.cell_size
        height = self.rows * self.cell_size
        image = Image.new("RGB", (width, height), (255, 255, 255))
        draw = ImageDraw.Draw(image)

        # Draw grid
        for r in range(self.rows + 1):
            y = r * self.cell_size
            width_px = 3 if r in (0, self.rows) else 1
            draw.line((0, y, width, y), fill=(0, 0, 0), width=width_px)
        for c in range(self.cols + 1):
            x = c * self.cell_size
            width_px = 4 if c == self.cols // 2 else (3 if c in (0, self.cols) else 1)
            draw.line((x, 0, x, height), fill=(0, 0, 0), width=width_px)

        half_cols = self.cols // 2
        color_map = {(cell.row, cell.col): cell.color for cell in colored_cells}
        for r in range(self.rows):
            for c in range(self.cols):
                if c < half_cols:
                    color = color_map.get((r, c))
                elif mirror:
                    mirror_col = half_cols - 1 - (c - half_cols)
                    color = color_map.get((r, mirror_col))
                else:
                    color = None
                if color is None:
                    continue
                x0 = c * self.cell_size + 1
                y0 = r * self.cell_size + 1
                x1 = (c + 1) * self.cell_size - 1
                y1 = (r + 1) * self.cell_size - 1
                draw.rectangle((x0, y0, x1, y1), fill=color)
        return image


__all__ = ["MirrorGenerator", "MirrorPuzzleRecord", "CellColor"]


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate mirror puzzles for VLM training")
    parser.add_argument("count", type=int, help="Number of puzzles to generate")
    parser.add_argument("--output-dir", type=Path, default=Path("data/mirror"), help="Where to save assets")
    parser.add_argument("--rows", type=int, default=6)
    parser.add_argument("--cols", type=int, default=8, help="Must be even")
    parser.add_argument("--cell-size", type=int, default=48)
    parser.add_argument("--fill", type=float, default=0.6, help="Fraction of left-half cells to color")
    parser.add_argument("--monochrome", action="store_true", help="Use a single color for all filled cells")
    parser.add_argument("--prompt", type=str, default="Mirror the colors from the left half onto the right half of the grid.")
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)
    generator = MirrorGenerator(
        output_dir=args.output_dir,
        rows=args.rows,
        cols=args.cols,
        cell_size=args.cell_size,
        left_fill_ratio=args.fill,
        monochrome=args.monochrome,
        prompt=args.prompt,
        seed=args.seed,
    )
    metadata_path = generator.output_dir / "puzzles.json"
    generator.generate_dataset(args.count, metadata_path=metadata_path)


if __name__ == "__main__":
    main()
