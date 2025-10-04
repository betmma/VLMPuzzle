"""Maze puzzle generator for grid-based path drawing tasks."""

from __future__ import annotations

import argparse
import random
import uuid
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw

from ..base import AbstractPuzzleGenerator, PathLike

WALL = 1
PATH = 0

WALL_COLOR = (0, 0, 0)
PATH_COLOR = (255, 255, 255)
START_COLOR = (220, 30, 30)
GOAL_COLOR = (40, 180, 80)
LINE_COLOR = (220, 0, 0)
BACKGROUND_COLOR = (0, 0, 0)


@dataclass
class MazePuzzleRecord:
    id: str
    prompt: str
    grid_size: Tuple[int, int]
    cell_size: int
    maze_grid: List[List[int]]
    start: Tuple[int, int]
    goal: Tuple[int, int]
    cell_bboxes: List[List[Tuple[int, int, int, int]]]
    padding: Tuple[int, int, int, int]
    canvas_dimensions: Tuple[int, int]
    puzzle_image_path: str
    solution_image_path: str

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "prompt": self.prompt,
            "grid_size": list(self.grid_size),
            "cell_size": self.cell_size,
            "maze_grid": self.maze_grid,
            "start": list(self.start),
            "goal": list(self.goal),
            "cell_bboxes": [
                [list(map(int, bbox)) for bbox in row] for row in self.cell_bboxes
            ],
            "padding": list(self.padding),
            "canvas_dimensions": list(self.canvas_dimensions),
            "puzzle_image_path": self.puzzle_image_path,
            "solution_image_path": self.solution_image_path,
        }


class MazeGenerator(AbstractPuzzleGenerator[MazePuzzleRecord]):
    """Generate maze puzzles that require drawing a path from start to goal."""

    def __init__(
        self,
        output_dir: PathLike = "data/maze",
        *,
        rows: int = 15,
        cols: int = 15,
        cell_size: int = 32,
        prompt: str = "Draw a red line from the red start cell to the green goal cell, avoiding the black walls. Static camera, no zoom, no pan, no dolly.",
        aspect_ratio: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(output_dir)
        if rows < 5 or cols < 5:
            raise ValueError("rows and cols must be at least 5")
        self.rows = rows if rows % 2 == 1 else rows + 1
        self.cols = cols if cols % 2 == 1 else cols + 1
        self.cell_size = cell_size
        self.prompt = prompt
        self.aspect_ratio = aspect_ratio
        self._rng = random.Random(seed)

        self.puzzle_dir = self.output_dir / "puzzles"
        self.solution_dir = self.output_dir / "solutions"
        for directory in (self.puzzle_dir, self.solution_dir):
            directory.mkdir(parents=True, exist_ok=True)

    def create_puzzle(self, *, puzzle_id: Optional[str] = None) -> MazePuzzleRecord:
        puzzle_uuid = puzzle_id or str(uuid.uuid4())
        maze_grid = self._generate_maze()
        start = (1, 1)
        goal = (self.rows - 2, self.cols - 2)
        path = self._bfs_path(maze_grid, start, goal)
        if not path:
            raise RuntimeError("Failed to generate maze path")

        pad_left, pad_top, pad_right, pad_bottom, canvas_dims = self._compute_padding()
        cell_bboxes = self._compute_cell_bboxes(pad_left, pad_top)

        puzzle_image = self._render_maze(
            maze_grid,
            start=start,
            goal=goal,
            path=None,
            padding=(pad_left, pad_top),
            canvas_dims=canvas_dims,
        )
        solution_image = self._render_maze(
            maze_grid,
            start=start,
            goal=goal,
            path=path,
            padding=(pad_left, pad_top),
            canvas_dims=canvas_dims,
        )

        puzzle_path = self.puzzle_dir / f"{puzzle_uuid}_puzzle.png"
        solution_path = self.solution_dir / f"{puzzle_uuid}_solution.png"
        puzzle_image.save(puzzle_path)
        solution_image.save(solution_path)

        return MazePuzzleRecord(
            id=puzzle_uuid,
            prompt=self.prompt,
            grid_size=(self.rows, self.cols),
            cell_size=self.cell_size,
            maze_grid=maze_grid,
            start=start,
            goal=goal,
            cell_bboxes=cell_bboxes,
            padding=(pad_left, pad_top, pad_right, pad_bottom),
            canvas_dimensions=canvas_dims,
            puzzle_image_path=self.relativize_path(puzzle_path),
            solution_image_path=self.relativize_path(solution_path),
        )

    def create_random_puzzle(self) -> MazePuzzleRecord:
        return self.create_puzzle()

    # ------------------------------------------------------------------

    def _generate_maze(self) -> List[List[int]]:
        grid = [[WALL for _ in range(self.cols)] for _ in range(self.rows)]

        def carve(r: int, c: int) -> None:
            grid[r][c] = PATH
            directions = [(0, 2), (0, -2), (2, 0), (-2, 0)]
            self._rng.shuffle(directions)
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 1 <= nr < self.rows - 1 and 1 <= nc < self.cols - 1 and grid[nr][nc] == WALL:
                    grid[r + dr // 2][c + dc // 2] = PATH
                    carve(nr, nc)

        carve(1, 1)
        # Ensure goal cell is reachable
        grid[self.rows - 2][self.cols - 2] = PATH
        return grid

    def _bfs_path(
        self,
        grid: List[List[int]],
        start: Tuple[int, int],
        goal: Tuple[int, int],
    ) -> List[Tuple[int, int]]:
        queue: deque[Tuple[int, int]] = deque([start])
        parents = {start: None}
        while queue:
            r, c = queue.popleft()
            if (r, c) == goal:
                break
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    if grid[nr][nc] == PATH and (nr, nc) not in parents:
                        parents[(nr, nc)] = (r, c)
                        queue.append((nr, nc))

        if goal not in parents:
            return []
        node = goal
        result: List[Tuple[int, int]] = []
        while node is not None:
            result.append(node)
            node = parents[node]
        result.reverse()
        return result

    def _compute_padding(self) -> Tuple[int, int, int, int, Tuple[int, int]]:
        base_width = self.cols * self.cell_size
        base_height = self.rows * self.cell_size
        if self.aspect_ratio is None:
            return 0, 0, 0, 0, (base_width, base_height)
        ratio = float(self.aspect_ratio)
        if ratio <= 0:
            raise ValueError("aspect_ratio must be positive")
        base_ratio = base_width / base_height
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
        canvas_width = base_width + pad_left + pad_right
        canvas_height = base_height + pad_top + pad_bottom
        return pad_left, pad_top, pad_right, pad_bottom, (canvas_width, canvas_height)

    def _compute_cell_bboxes(self, pad_left: int, pad_top: int) -> List[List[Tuple[int, int, int, int]]]:
        bboxes: List[List[Tuple[int, int, int, int]]] = []
        for r in range(self.rows):
            row_boxes: List[Tuple[int, int, int, int]] = []
            for c in range(self.cols):
                left = pad_left + c * self.cell_size
                top = pad_top + r * self.cell_size
                row_boxes.append((left, top, left + self.cell_size, top + self.cell_size))
            bboxes.append(row_boxes)
        return bboxes

    def _render_maze(
        self,
        grid: Sequence[Sequence[int]],
        *,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        path: Optional[Sequence[Tuple[int, int]]],
        padding: Tuple[int, int],
        canvas_dims: Tuple[int, int],
    ) -> Image.Image:
        pad_left, pad_top = padding
        canvas = Image.new("RGB", canvas_dims, BACKGROUND_COLOR)
        draw = ImageDraw.Draw(canvas)

        # Draw cells
        for r in range(self.rows):
            for c in range(self.cols):
                left = pad_left + c * self.cell_size
                top = pad_top + r * self.cell_size
                right = left + self.cell_size
                bottom = top + self.cell_size
                if (r, c) == start:
                    fill = START_COLOR
                elif (r, c) == goal:
                    fill = GOAL_COLOR
                else:
                    fill = PATH_COLOR if grid[r][c] == PATH else WALL_COLOR
                draw.rectangle((left, top, right - 1, bottom - 1), fill=fill)

        if path:
            thickness = max(2, self.cell_size // 3)
            points = [
                (
                    pad_left + c * self.cell_size + self.cell_size / 2,
                    pad_top + r * self.cell_size + self.cell_size / 2,
                )
                for r, c in path
            ]
            if len(points) >= 2:
                draw.line(points, fill=LINE_COLOR, width=thickness, joint="curve")
            elif points:
                x, y = points[0]
                draw.ellipse(
                    (
                        x - thickness // 2,
                        y - thickness // 2,
                        x + thickness // 2,
                        y + thickness // 2,
                    ),
                    fill=LINE_COLOR,
                )
            # Reinforce start/goal colors on top of the line for clarity
            self._draw_cell(draw, start, pad_left, pad_top, START_COLOR)
            self._draw_cell(draw, goal, pad_left, pad_top, GOAL_COLOR)
            # Draw a thin red overlay within start/goal to keep the line visible
            self._draw_path_marker(draw, start, pad_left, pad_top, thickness - 2)
            self._draw_path_marker(draw, goal, pad_left, pad_top, thickness - 2)
        return canvas

    def _draw_cell(
        self,
        draw: ImageDraw.ImageDraw,
        cell: Tuple[int, int],
        pad_left: int,
        pad_top: int,
        color: Tuple[int, int, int],
    ) -> None:
        r, c = cell
        left = pad_left + c * self.cell_size
        top = pad_top + r * self.cell_size
        right = left + self.cell_size
        bottom = top + self.cell_size
        draw.rectangle((left, top, right - 1, bottom - 1), fill=color)

    def _draw_path_marker(
        self,
        draw: ImageDraw.ImageDraw,
        cell: Tuple[int, int],
        pad_left: int,
        pad_top: int,
        thickness: int,
    ) -> None:
        if thickness <= 0:
            return
        r, c = cell
        cx = pad_left + c * self.cell_size + self.cell_size / 2
        cy = pad_top + r * self.cell_size + self.cell_size / 2
        draw.ellipse(
            (
                cx - thickness / 2,
                cy - thickness / 2,
                cx + thickness / 2,
                cy + thickness / 2,
            ),
            fill=LINE_COLOR,
        )


__all__ = ["MazeGenerator", "MazePuzzleRecord"]


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate maze puzzles for VLM training")
    parser.add_argument("count", type=int, help="Number of puzzles to generate")
    parser.add_argument("--output-dir", type=Path, default=Path("data/maze"), help="Where to save assets")
    parser.add_argument("--rows", type=int, default=15)
    parser.add_argument("--cols", type=int, default=15)
    parser.add_argument("--cell-size", type=int, default=32)
    parser.add_argument(
        "--aspect-ratio",
        type=float,
        default=None,
        help="Optional width/height ratio for the final image (adds black padding on outer edges only)",
    )
    parser.add_argument("--prompt", type=str, default="Draw a red line from the red start cell to the green goal cell, avoiding the black walls.")
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)
    generator = MazeGenerator(
        output_dir=args.output_dir,
        rows=args.rows,
        cols=args.cols,
        cell_size=args.cell_size,
        prompt=args.prompt,
        aspect_ratio=args.aspect_ratio,
        seed=args.seed,
    )
    metadata_path = generator.output_dir / "puzzles.json"
    generator.generate_dataset(args.count, metadata_path=metadata_path)


if __name__ == "__main__":
    main()
