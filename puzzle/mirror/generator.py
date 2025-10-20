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
    cell_width: int
    cell_height: int
    cell_inner_size: int
    cell_padding: Tuple[int, int, int, int]
    cell_inner_bounds: Tuple[int, int, int, int]
    cell_aspect_ratio: float
    canvas_size: Tuple[int, int]
    colored_cells: List[CellColor]
    image: str
    solution_image_path: str
    monochrome: bool

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "prompt": self.prompt,
            "grid_size": list(self.grid_size),
            "cell_size": self.cell_size,
            "cell_width": self.cell_width,
            "cell_height": self.cell_height,
            "cell_inner_size": self.cell_inner_size,
            "cell_padding": list(self.cell_padding),
            "cell_inner_bounds": list(self.cell_inner_bounds),
            "cell_aspect_ratio": self.cell_aspect_ratio,
            "canvas_size": list(self.canvas_size),
            "colored_cells": [cell.to_dict() for cell in self.colored_cells],
            "image": self.image,
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
        cell_aspect_ratio: Optional[float] = None,
        prompt: str = "Instantly reflect this pattern along the central, vertical axis while keeping the existing colored pattern without modification. Static camera perspective, no zoom or pan.",
        left_fill_ratio: float = 0.6,
        monochrome: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(output_dir)
        if cols % 2 != 0:
            raise ValueError("Column count must be even for mirroring")
        self.rows = rows
        self.cols = cols
        self.cell_size = int(cell_size)
        (
            self.pad_left,
            self.pad_top,
            self.pad_right,
            self.pad_bottom,
            self.cell_aspect_ratio,
            self.canvas_size,
        ) = self._compute_padding(self.cell_size, cell_aspect_ratio, self.rows, self.cols)
        self.cell_width = self.cell_size
        self.cell_height = self.cell_size
        self.cell_inner_size = max(1, self.cell_size - 2)
        inner_right = 1 + self.cell_inner_size
        inner_bottom = 1 + self.cell_inner_size
        self.cell_inner_bounds = (1, 1, inner_right, inner_bottom)
        self.prompt = prompt
        self.left_fill_ratio = max(0.1, min(left_fill_ratio, 1.0))
        self.monochrome = monochrome
        self._rng = random.Random(seed)

        self.puzzle_dir = self.output_dir / "puzzles"
        self.solution_dir = self.output_dir / "solutions"
        for path in (self.puzzle_dir, self.solution_dir):
            path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _compute_padding(
        cell_size: int,
        aspect_ratio: Optional[float],
        rows: int,
        cols: int,
    ) -> Tuple[int, int, int, int, float, Tuple[int, int]]:
        base_width = cols * cell_size
        base_height = rows * cell_size
        if base_height == 0 or base_width == 0:
            raise ValueError("Grid dimensions must be positive")

        base_ratio = base_width / base_height
        if aspect_ratio is None:
            ratio = base_ratio
            pad_left = pad_top = pad_right = pad_bottom = 0
            final_width = base_width
            final_height = base_height
        else:
            ratio = float(aspect_ratio)
            if ratio <= 0:
                raise ValueError("cell_aspect_ratio must be positive")
            if ratio >= base_ratio:
                final_height = base_height
                final_width = max(base_width, int(round(final_height * ratio)))
                extra = final_width - base_width
                pad_left = extra // 2
                pad_right = extra - pad_left
                pad_top = pad_bottom = 0
            else:
                final_width = base_width
                final_height = max(base_height, int(round(final_width / ratio)))
                extra = final_height - base_height
                pad_top = extra // 2
                pad_bottom = extra - pad_top
                pad_left = pad_right = 0
        canvas_size = (base_width + pad_left + pad_right, base_height + pad_top + pad_bottom)
        return pad_left, pad_top, pad_right, pad_bottom, ratio, canvas_size

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
            cell_width=self.cell_width,
            cell_height=self.cell_height,
            cell_inner_size=self.cell_inner_size,
            cell_padding=(self.pad_left, self.pad_top, self.pad_right, self.pad_bottom),
            cell_inner_bounds=self.cell_inner_bounds,
            cell_aspect_ratio=self.cell_aspect_ratio,
            canvas_size=self.canvas_size,
            colored_cells=colored_cells,
            image=self.relativize_path(puzzle_path),
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
        image = Image.new("RGB", self.canvas_size, (0, 0, 0))
        draw = ImageDraw.Draw(image)

        base_width = self.cols * self.cell_size
        base_height = self.rows * self.cell_size
        grid_left = self.pad_left
        grid_top = self.pad_top
        grid_right = grid_left + base_width
        grid_bottom = grid_top + base_height

        inner_left, inner_top, inner_right, inner_bottom = self.cell_inner_bounds
        color_map = {(cell.row, cell.col): cell.color for cell in colored_cells}
        half_cols = self.cols // 2

        # Paint cell backgrounds
        for r in range(self.rows):
            for c in range(self.cols):
                cell_left = grid_left + c * self.cell_size
                cell_top = grid_top + r * self.cell_size
                cell_right = cell_left + self.cell_size - 1
                cell_bottom = cell_top + self.cell_size - 1
                draw.rectangle((cell_left, cell_top, cell_right, cell_bottom), fill=(255, 255, 255))

                if c < half_cols:
                    color = color_map.get((r, c))
                elif mirror:
                    mirror_col = half_cols - 1 - (c - half_cols)
                    color = color_map.get((r, mirror_col))
                else:
                    color = None
                if color is None:
                    continue
                x0 = cell_left + inner_left
                y0 = cell_top + inner_top
                x1 = cell_left + inner_right - 1
                y1 = cell_top + inner_bottom - 1
                draw.rectangle((x0, y0, x1, y1), fill=color)

        # Draw grid lines on top of filled cells
        for r in range(self.rows + 1):
            y = grid_top + r * self.cell_size
            width_px = 3 if r in (0, self.rows) else 1
            draw.line((grid_left, y, grid_right, y), fill=(0, 0, 0), width=width_px)
        for c in range(self.cols + 1):
            x = grid_left + c * self.cell_size
            width_px = 4 if c == self.cols // 2 else (3 if c in (0, self.cols) else 1)
            draw.line((x, grid_top, x, grid_bottom), fill=(0, 0, 0), width=width_px)

        return image


__all__ = ["MirrorGenerator", "MirrorPuzzleRecord", "CellColor"]


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate mirror puzzles for VLM training")
    parser.add_argument("count", type=int, help="Number of puzzles to generate")
    parser.add_argument("--output-dir", type=Path, default=Path("data/mirror"), help="Where to save assets")
    parser.add_argument("--rows", type=int, default=6)
    parser.add_argument("--cols", type=int, default=8, help="Must be even")
    parser.add_argument("--cell-size", type=int, default=48, help="Side length (in pixels) of the square paintable region in each cell")
    parser.add_argument(
        "--aspect-ratio",
        type=float,
        default=None,
        help="Width/height ratio for the overall puzzle. Padding is only added at the outer edges while each cell remains square.",
    )
    parser.add_argument("--fill", type=float, default=0.6, help="Fraction of left-half cells to color")
    parser.add_argument("--monochrome", action="store_true", help="Use a single color for all filled cells")
    parser.add_argument("--prompt", type=str, default="Instantly reflect this pattern along the central, vertical axis while keeping the existing colored pattern without modification. Static camera perspective, no zoom or pan.")
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)
    generator = MirrorGenerator(
        output_dir=args.output_dir,
        rows=args.rows,
        cols=args.cols,
        cell_size=args.cell_size,
        cell_aspect_ratio=args.aspect_ratio,
        left_fill_ratio=args.fill,
        monochrome=args.monochrome,
        prompt=args.prompt,
        seed=args.seed,
    )
    metadata_path = generator.output_dir / "data.json"
    generator.generate_dataset(args.count, metadata_path=metadata_path)


if __name__ == "__main__":
    main()

