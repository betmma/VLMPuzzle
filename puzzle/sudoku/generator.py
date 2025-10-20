"""Sudoku puzzle generator implementation (4x4 grid)."""

from __future__ import annotations

import argparse
import random
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont

from ..base import AbstractPuzzleGenerator, PathLike

GRID_SIZE = 4
SUBGRID_SIZE = 2
TOTAL_CELLS = GRID_SIZE * GRID_SIZE


@dataclass
class SudokuPuzzleRecord:
    """Persisted Sudoku puzzle metadata."""

    id: str
    prompt: str
    puzzle_grid: List[List[int]]
    solution_grid: List[List[int]]
    clue_count: int
    image: str
    solution_image_path: str
    cell_bboxes: List[List[Tuple[int, int, int, int]]]
    canvas_size: int
    canvas_dimensions: Tuple[int, int]
    padding: Tuple[int, int, int, int]
    font: Dict[str, Optional[int]]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "prompt": self.prompt,
            "puzzle_grid": self.puzzle_grid,
            "solution_grid": self.solution_grid,
            "clue_count": self.clue_count,
            "image": self.image,
            "solution_image_path": self.solution_image_path,
            "cell_bboxes": [
                [list(map(int, bbox)) for bbox in row] for row in self.cell_bboxes
            ],
            "canvas_size": self.canvas_size,
            "canvas_dimensions": list(self.canvas_dimensions),
            "padding": list(self.padding),
            "font": self.font,
        }


class SudokuGenerator(AbstractPuzzleGenerator[SudokuPuzzleRecord]):
    """Generate 4x4 Sudoku puzzles with controllable clue counts."""

    def __init__(
        self,
        output_dir: PathLike = "data/sudoku",
        *,
        clue_target: int = 12,
        ensure_unique: bool = True,
        prompt: str = "Create a static, smooth, animation that solves the given 4x4 sudoku. Enter the missing numbers one by one. Do not change anything else in the picture. Only fill the numbers in the empty cells so the sudoku is solved properly. A cursor moves and fills the correct number in the empty boxes.",
        canvas_size: int = 360,
        canvas_aspect_ratio: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(output_dir)
        min_clues = max(4, SUBGRID_SIZE * SUBGRID_SIZE)
        max_clues = TOTAL_CELLS - 1  # leave at least one blank
        self.clue_target = max(min_clues, min(clue_target, max_clues))
        self.ensure_unique = ensure_unique
        self.prompt = prompt
        self.canvas_size = canvas_size
        (
            self.pad_left,
            self.pad_top,
            self.pad_right,
            self.pad_bottom,
            self.canvas_aspect_ratio,
            self.canvas_dimensions,
        ) = self._compute_outer_padding(canvas_size, canvas_aspect_ratio)
        self._rng = random.Random(seed)

        self.puzzle_dir = self.output_dir / "puzzles"
        self.solution_dir = self.output_dir / "solutions"
        for path in (self.puzzle_dir, self.solution_dir):
            path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _compute_outer_padding(
        canvas_size: int,
        aspect_ratio: Optional[float],
    ) -> Tuple[int, int, int, int, float, Tuple[int, int]]:
        base_width = base_height = canvas_size
        if base_width <= 0 or base_height <= 0:
            raise ValueError("canvas_size must be positive")
        base_ratio = base_width / base_height
        if aspect_ratio is None:
            ratio = base_ratio
            pad_left = pad_top = pad_right = pad_bottom = 0
        else:
            ratio = float(aspect_ratio)
            if ratio <= 0:
                raise ValueError("canvas_aspect_ratio must be positive")
            if ratio >= base_ratio:
                extra_width = max(0, int(round(base_height * ratio)) - base_width)
                pad_left = extra_width // 2
                pad_right = extra_width - pad_left
                pad_top = pad_bottom = 0
            else:
                extra_height = max(0, int(round(base_width / ratio)) - base_height)
                pad_top = extra_height // 2
                pad_bottom = extra_height - pad_top
                pad_left = pad_right = 0
        canvas_width = base_width + pad_left + pad_right
        canvas_height = base_height + pad_top + pad_bottom
        return pad_left, pad_top, pad_right, pad_bottom, ratio, (canvas_width, canvas_height)

    def create_puzzle(self, *, puzzle_id: Optional[str] = None) -> SudokuPuzzleRecord:
        puzzle_uuid = puzzle_id or str(uuid.uuid4())
        solution = self._generate_solution()
        puzzle = self._carve_puzzle([row[:] for row in solution])

        cell_size = self.canvas_size // GRID_SIZE
        cell_bboxes = self._compute_cell_bboxes(cell_size)
        font, font_info = self._resolve_font(cell_size)

        puzzle_image = self._render_board(
            puzzle,
            cell_size=cell_size,
            font=font,
            puzzle_grid=puzzle,
            highlight_solution=False,
        )
        solution_image = self._render_board(
            solution,
            cell_size=cell_size,
            font=font,
            puzzle_grid=puzzle,
            highlight_solution=True,
        )

        puzzle_path = self.puzzle_dir / f"{puzzle_uuid}_puzzle.png"
        solution_path = self.solution_dir / f"{puzzle_uuid}_solution.png"
        puzzle_image.save(puzzle_path)
        solution_image.save(solution_path)

        record = SudokuPuzzleRecord(
            id=puzzle_uuid,
            prompt=self.prompt,
            puzzle_grid=puzzle,
            solution_grid=solution,
            clue_count=sum(cell != 0 for row in puzzle for cell in row),
            image=self.relativize_path(puzzle_path),
            solution_image_path=self.relativize_path(solution_path),
            cell_bboxes=cell_bboxes,
            canvas_size=self.canvas_size,
            canvas_dimensions=self.canvas_dimensions,
            padding=(self.pad_left, self.pad_top, self.pad_right, self.pad_bottom),
            font=font_info,
        )
        return record

    def create_random_puzzle(self) -> SudokuPuzzleRecord:
        return self.create_puzzle()

    # --- Sudoku generation internals -------------------------------------------------

    def _generate_solution(self) -> List[List[int]]:
        grid = [row[:] for row in _BASE_SOLUTION]
        self._shuffle_digits(grid)
        self._shuffle_rows(grid)
        self._shuffle_columns(grid)
        return grid

    def _shuffle_digits(self, grid: List[List[int]]) -> None:
        mapping = list(range(1, GRID_SIZE + 1))
        self._rng.shuffle(mapping)
        digit_map = {i + 1: mapping[i] for i in range(GRID_SIZE)}
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                grid[r][c] = digit_map[grid[r][c]]

    def _shuffle_rows(self, grid: List[List[int]]) -> None:
        for band_start in range(0, GRID_SIZE, SUBGRID_SIZE):
            band_rows = list(range(band_start, band_start + SUBGRID_SIZE))
            self._rng.shuffle(band_rows)
            grid[band_start : band_start + SUBGRID_SIZE] = [grid[i] for i in band_rows]
        bands = [grid[i : i + SUBGRID_SIZE] for i in range(0, GRID_SIZE, SUBGRID_SIZE)]
        self._rng.shuffle(bands)
        grid[:] = [row for band in bands for row in band]

    def _shuffle_columns(self, grid: List[List[int]]) -> None:
        columns = list(zip(*grid))
        mutable_cols = [list(col) for col in columns]
        for stack_start in range(0, GRID_SIZE, SUBGRID_SIZE):
            stack_cols = list(range(stack_start, stack_start + SUBGRID_SIZE))
            self._rng.shuffle(stack_cols)
            mutable_cols[stack_start : stack_start + SUBGRID_SIZE] = [mutable_cols[i] for i in stack_cols]
        stacks = [mutable_cols[i : i + SUBGRID_SIZE] for i in range(0, GRID_SIZE, SUBGRID_SIZE)]
        self._rng.shuffle(stacks)
        shuffled_cols = [col for stack in stacks for col in stack]
        for r in range(GRID_SIZE):
            grid[r] = [shuffled_cols[c][r] for c in range(GRID_SIZE)]

    def _carve_puzzle(self, grid: List[List[int]]) -> List[List[int]]:
        positions = list(range(TOTAL_CELLS))
        self._rng.shuffle(positions)
        removals = TOTAL_CELLS - self.clue_target
        for pos in positions:
            if removals <= 0:
                break
            r, c = divmod(pos, GRID_SIZE)
            if grid[r][c] == 0:
                continue
            backup = grid[r][c]
            grid[r][c] = 0
            if self.ensure_unique and not self._has_unique_solution(grid):
                grid[r][c] = backup
            else:
                removals -= 1
        return grid

    def _has_unique_solution(self, grid: List[List[int]]) -> bool:
        grid_copy = [row[:] for row in grid]
        count = self._count_solutions(grid_copy, limit=2)
        return count == 1

    def _count_solutions(self, grid: List[List[int]], *, limit: int) -> int:
        def solve(idx: int, count: int) -> int:
            if count >= limit:
                return count
            if idx == TOTAL_CELLS:
                return count + 1
            r, c = divmod(idx, GRID_SIZE)
            if grid[r][c] != 0:
                return solve(idx + 1, count)
            for value in _DIGITS:
                if self._is_safe(grid, r, c, value):
                    grid[r][c] = value
                    count = solve(idx + 1, count)
                    grid[r][c] = 0
                    if count >= limit:
                        break
            return count

        return solve(0, 0)

    @staticmethod
    def _is_safe(grid: List[List[int]], row: int, col: int, value: int) -> bool:
        if any(grid[row][c] == value for c in range(GRID_SIZE)):
            return False
        if any(grid[r][col] == value for r in range(GRID_SIZE)):
            return False
        start_row = row - row % SUBGRID_SIZE
        start_col = col - col % SUBGRID_SIZE
        for r in range(start_row, start_row + SUBGRID_SIZE):
            for c in range(start_col, start_col + SUBGRID_SIZE):
                if grid[r][c] == value:
                    return False
        return True

    # --- Rendering -------------------------------------------------------------------

    def _render_board(
        self,
        grid: Sequence[Sequence[int]],
        *,
        cell_size: int,
        font: ImageFont.FreeTypeFont,
        puzzle_grid: Optional[Sequence[Sequence[int]]] = None,
        highlight_solution: bool = False,
    ) -> Image.Image:
        canvas = Image.new("RGB", self.canvas_dimensions, color="black")
        draw = ImageDraw.Draw(canvas)

        board_left = self.pad_left
        board_top = self.pad_top
        board_right = board_left + self.canvas_size
        board_bottom = board_top + self.canvas_size
        draw.rectangle((board_left, board_top, board_right - 1, board_bottom - 1), fill="white")

        # Grid lines
        for i in range(GRID_SIZE + 1):
            line_width = 4 if i % SUBGRID_SIZE == 0 else 1
            offset = i * cell_size
            y = board_top + offset
            x = board_left + offset
            draw.line((board_left, y, board_right, y), fill="black", width=line_width)
            draw.line((x, board_top, x, board_bottom), fill="black", width=line_width)

        # Digits
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                value = grid[r][c]
                if value == 0:
                    continue
                text = str(value)
                x0 = board_left + c * cell_size
                y0 = board_top + r * cell_size
                bbox = draw.textbbox((0, 0), text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                x_text = x0 + (cell_size - text_width) / 2 - bbox[0]
                y_text = y0 + (cell_size - text_height) / 2 - bbox[1]
                if highlight_solution:
                    is_clue = puzzle_grid is not None and puzzle_grid[r][c] != 0
                    fill = "black" if is_clue else "blue"
                else:
                    fill = "black"
                draw.text((x_text, y_text), text, fill=fill, font=font)
        return canvas

    def _resolve_font(self, cell_size: int) -> Tuple[ImageFont.FreeTypeFont, Dict[str, Optional[int]]]:
        target_size = max(12, int(cell_size * 0.7))
        for font_name in ["arial.ttf", "LiberationSans-Regular.ttf", "DejaVuSans.ttf"]:
            try:
                font = ImageFont.truetype(font_name, target_size)
                return font, {"type": "truetype", "name": font_name, "size": target_size}
            except OSError:
                continue
        font = ImageFont.load_default()
        size_attr = getattr(font, "size", target_size)
        return font, {"type": "default", "name": None, "size": int(size_attr)}

    def _compute_cell_bboxes(self, cell_size: int) -> List[List[Tuple[int, int, int, int]]]:
        bboxes: List[List[Tuple[int, int, int, int]]] = []
        for r in range(GRID_SIZE):
            row_boxes: List[Tuple[int, int, int, int]] = []
            for c in range(GRID_SIZE):
                left = self.pad_left + c * cell_size
                top = self.pad_top + r * cell_size
                row_boxes.append((left, top, left + cell_size, top + cell_size))
            bboxes.append(row_boxes)
        return bboxes


_BASE_SOLUTION: List[List[int]] = [
    [1, 2, 3, 4],
    [3, 4, 1, 2],
    [2, 1, 4, 3],
    [4, 3, 2, 1],
]

_DIGITS = list(range(1, GRID_SIZE + 1))


__all__ = ["SudokuGenerator", "SudokuPuzzleRecord"]


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Sudoku puzzles for video LM training")
    parser.add_argument("count", type=int, help="Number of puzzles to generate")
    parser.add_argument("--output-dir", type=Path, default=Path("data/sudoku"), help="Where to save artifacts")
    parser.add_argument("--clue-target", type=int, default=12, help="Approximate number of given cells")
    parser.add_argument("--no-unique", action="store_true", help="Skip uniqueness enforcement for faster generation")
    parser.add_argument("--canvas-size", type=int, default=360, help="Render size in pixels for the Sudoku board")
    parser.add_argument("--aspect-ratio", type=float, default=None, help="Optional width/height ratio for the full image (adds black padding on outer borders only)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)
    generator = SudokuGenerator(
        output_dir=args.output_dir,
        clue_target=args.clue_target,
        ensure_unique=not args.no_unique,
        canvas_size=args.canvas_size,
        canvas_aspect_ratio=args.aspect_ratio,
        seed=args.seed,
    )
    metadata_path = generator.output_dir / "data.json"
    generator.generate_dataset(args.count, metadata_path=metadata_path)


if __name__ == "__main__":
    main()
