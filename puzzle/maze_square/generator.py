"""Maze puzzle generator for grid-based path drawing tasks."""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont

from ..maze_base import MazePuzzleGenerator, MazePuzzleRecord, draw_path_line

WALL = 1
PATH = 0

WALL_COLOR = (0, 0, 0)
PATH_COLOR = (255, 255, 255)
START_COLOR = (220, 30, 30)
GOAL_COLOR = START_COLOR #(40, 180, 80)
LINE_COLOR = (220, 0, 0)
BACKGROUND_COLOR = (0, 0, 0)
TEXT_COLOR = (0, 0, 255)


class MazeGenerator(MazePuzzleGenerator):
    """Generate maze puzzles that require drawing a path from start to goal."""

    DEFAULT_ROWS = 15
    DEFAULT_COLS = 15
    DEFAULT_CELL_SIZE = 32

    def __init__(
        self,
        output_dir: Optional[str | Path] = None,
        *,
        canvas_width: Optional[int] = None,
        aspect: Optional[float] = None,
        size: Optional[int] = None,
        rows: int = DEFAULT_ROWS,
        cols: int = DEFAULT_COLS,
        cell_size: Optional[int] = None,
        aspect_ratio: Optional[float] = None,
        seed: Optional[int] = None,
        prompt: Optional[str] = None,
        show_cell_id: bool = False,
        video: bool = False,
    ) -> None:
        if rows < 5 or cols < 5:
            raise ValueError("rows and cols must be at least 5")
        adjusted_rows = rows if rows % 2 == 1 else rows + 1
        adjusted_cols = cols if cols % 2 == 1 else cols + 1
        effective_cell_size = int(
            cell_size if cell_size is not None else (size if size is not None else self.DEFAULT_CELL_SIZE)
        )
        if effective_cell_size <= 0:
            raise ValueError("cell_size must be positive")
        ratio = aspect_ratio if aspect_ratio is not None else aspect
        pad_left, pad_top, pad_right, pad_bottom, canvas_dims = self._layout_for(
            adjusted_rows,
            adjusted_cols,
            effective_cell_size,
            ratio,
        )
        if canvas_width is not None and canvas_width > canvas_dims[0]:
            extra = canvas_width - canvas_dims[0]
            left_extra = extra // 2
            right_extra = extra - left_extra
            pad_left += left_extra
            pad_right += right_extra
            canvas_dims = (canvas_width, canvas_dims[1])
        final_width, final_height = canvas_dims
        aspect_for_super = (final_width / final_height) if final_height else None
        resolved_output = output_dir if output_dir is not None else self.DEFAULT_OUTPUT_DIR
        super().__init__(
            resolved_output,
            canvas_width=final_width,
            aspect=aspect_for_super,
            size=effective_cell_size,
            seed=seed,
            prompt=prompt,
            show_cell_id=show_cell_id,
            video=video,
        )
        self.rows = adjusted_rows
        self.cols = adjusted_cols
        self.cell_size = effective_cell_size
        self.aspect_ratio = ratio
        self.padding = (pad_left, pad_top, pad_right, pad_bottom)

    def create_puzzle(self, *, puzzle_id: Optional[str] = None) -> MazePuzzleRecord:
        puzzle_uuid = puzzle_id or self.next_id()
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

        puzzle_path, solution_path = self.save_images(puzzle_uuid, puzzle_image, solution_image)

        if self.video:
            path_points = [self._cell_center(self._get_cell_id(*cell)) for cell in path]
            self.save_video(puzzle_uuid, puzzle_image, path_points, thickness=max(3, self.cell_size // 3))

        start_point = self._cell_center(self._get_cell_id(*start))
        goal_point = self._cell_center(self._get_cell_id(*goal))
        extra_payload: Dict[str, object] = {
            "grid_size": [self.rows, self.cols],
            "cell_size": self.cell_size,
            "maze_grid": maze_grid,
            "start": list(start),
            "goal": list(goal),
            "cell_bboxes": cell_bboxes,
            "padding": [pad_left, pad_top, pad_right, pad_bottom],
            "solution_path_cell_ids": [self._get_cell_id(*cell) for cell in path],
        }
        return self.build_record(
            puzzle_uuid,
            start_point=start_point,
            goal_point=goal_point,
            puzzle_path=puzzle_path,
            solution_path=solution_path,
            prompt=self.prompt,
            extra=extra_payload,
        )

    # ------------------------------------------------------------------

    def _generate_maze(self) -> List[List[int]]:
        grid = [[WALL for _ in range(self.cols)] for _ in range(self.rows)]

        def carve(r: int, c: int) -> None:
            grid[r][c] = PATH
            directions = [(0, 2), (0, -2), (2, 0), (-2, 0)]
            self.rng.shuffle(directions)
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

    def _get_cell_id(self, r: int, c: int) -> int:
        return r * self.cols + c

    def _compute_padding(self) -> Tuple[int, int, int, int, Tuple[int, int]]:
        return self._layout_for(self.rows, self.cols, self.cell_size, self.aspect_ratio)

    @staticmethod
    def _layout_for(
        rows: int,
        cols: int,
        cell_size: int,
        aspect_ratio: Optional[float],
    ) -> Tuple[int, int, int, int, Tuple[int, int]]:
        base_width = cols * cell_size
        base_height = rows * cell_size
        if aspect_ratio is None:
            return 0, 0, 0, 0, (base_width, base_height)
        ratio = float(aspect_ratio)
        if ratio <= 0:
            raise ValueError("aspect ratio must be positive")
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
            draw_path_line(canvas, points, LINE_COLOR, thickness)

            # Reinforce start/goal colors on top of the line for clarity
            self._draw_cell(draw, start, pad_left, pad_top, START_COLOR)
            self._draw_cell(draw, goal, pad_left, pad_top, GOAL_COLOR)
            # Draw a thin red overlay within start/goal to keep the line visible
            self._draw_path_marker(draw, start, pad_left, pad_top, thickness - 2)
            self._draw_path_marker(draw, goal, pad_left, pad_top, thickness - 2)
        
        if self.show_cell_id:
            self._draw_cell_ids(draw, pad_left, pad_top)

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

    def _cell_center(self, cell_id: int) -> Tuple[float, float]:
        row = cell_id // self.cols
        col = cell_id % self.cols
        pad_left, pad_top, _, _, _ = self._compute_padding()
        x = pad_left + col * self.cell_size + self.cell_size / 2.0
        y = pad_top + row * self.cell_size + self.cell_size / 2.0
        return (x, y)

    def _draw_cell_ids(self, draw: ImageDraw.ImageDraw, pad_left: int, pad_top: int) -> None:
        font_size = max(8, int(self.cell_size * 0.4))
        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
        except OSError:
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except OSError:
                font = ImageFont.load_default()

        for r in range(self.rows):
            for c in range(self.cols):
                 text = str(self._get_cell_id(r, c))
                 center_x = pad_left + c * self.cell_size + self.cell_size / 2.0
                 center_y = pad_top + r * self.cell_size + self.cell_size / 2.0
                 draw.text((center_x, center_y), text, fill=TEXT_COLOR, anchor="mm", font=font)

    @classmethod
    def _parse_args(cls, argv: Optional[List[str]] = None) -> argparse.Namespace:
        parser = argparse.ArgumentParser(description="Generate maze puzzles for VLM training")
        parser.add_argument("count", type=int, help="Number of puzzles to generate")
        parser.add_argument("--output-dir", type=Path, default=None, help="Where to save assets")
        parser.add_argument("--rows", type=int, default=cls.DEFAULT_ROWS)
        parser.add_argument("--cols", type=int, default=cls.DEFAULT_COLS)
        parser.add_argument("--cell-size", type=int, default=None, help="Cell size in pixels; overrides --size")
        parser.add_argument("--canvas-width", type=int, default=None, help="Optional override for final canvas width")
        parser.add_argument("--aspect", type=float, default=None, help="Optional width/height ratio for the final image")
        parser.add_argument("--size", type=int, default=None, help="Alias for --cell-size to match base interface")
        parser.add_argument("--prompt", type=str, default=None)
        parser.add_argument("--seed", type=int, default=None)
        parser.add_argument("--show-cell-id", action="store_true", help="Draw cell IDs on the maze")
        parser.add_argument("--use-gpt-5", action="store_true", help="Same as --show-cell-id")
        parser.add_argument("--video", action="store_true", help="Generate solution video")
        namespace=parser.parse_args(argv)
        if namespace.use_gpt_5:
            namespace.show_cell_id = True
        return namespace

    @classmethod
    def main(cls, argv: Optional[List[str]] = None) -> None:
        args = cls._parse_args(argv)
        cell_size = args.cell_size if args.cell_size is not None else (args.size if args.size is not None else cls.DEFAULT_CELL_SIZE)
        prompt_arg = args.prompt if args.prompt is not None else cls.DEFAULT_PROMPT
        generator = cls(
            output_dir=args.output_dir,
            canvas_width=args.canvas_width,
            aspect=args.aspect,
            size=cell_size,
            rows=args.rows,
            cols=args.cols,
            cell_size=cell_size,
            seed=args.seed,
            prompt=prompt_arg,
            show_cell_id=args.show_cell_id,
            video=args.video,
        )
        records = [generator.create_random_puzzle() for _ in range(max(1, args.count))]
        generator.write_metadata(records, generator.output_dir / "data.json")


__all__ = ["MazeGenerator"]


def main(argv: Optional[List[str]] = None) -> None:
    MazeGenerator.main(argv)


if __name__ == "__main__":
    MazeGenerator.main()
