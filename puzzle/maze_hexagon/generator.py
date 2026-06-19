"""Hexagonal maze generator using concentric rings on an axial grid."""

from __future__ import annotations

import argparse
import math
from collections import deque
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from PIL import Image, ImageDraw, ImageFont

from ..maze_base import MazePuzzleGenerator, MazePuzzleRecord, draw_path_line

# Rendering palette
PATH_COLOR = (240, 240, 240)
WALL_COLOR = (0, 0, 0)
START_COLOR = (220, 30, 30)
GOAL_COLOR = START_COLOR
LINE_COLOR = (220, 0, 0)
BACKGROUND_COLOR = (16, 16, 16)
TEXT_COLOR = (0, 0, 255)

Axial = Tuple[int, int]

# Axial direction vectors for pointy-top hexagons (q, r)
DIRECTIONS: Tuple[Axial, ...] = (
    (1, -1),
    (1, 0),
    (0, 1),
    (-1, 1),
    (-1, 0),
    (0, -1),
)

EDGE_CORNER_INDICES: Tuple[Tuple[int, int], ...] = (
    (5, 0),
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (4, 5),
)

SQRT_THREE = math.sqrt(3.0)


class MazeHexagonGenerator(MazePuzzleGenerator):
    """Generate mazes laid out on a finite hexagonal grid."""

    DEFAULT_OUTPUT_DIR = "data/maze_hexagon"

    DEFAULT_RADIUS = 4
    DEFAULT_CELL_RADIUS = 38

    def __init__(
        self,
        output_dir: Optional[str | Path] = None,
        *,
        radius: int = DEFAULT_RADIUS,
        cell_radius: Optional[int] = None,
        wall_thickness: Optional[int] = None,
        size: Optional[int] = None,
        seed: Optional[int] = None,
        prompt: Optional[str] = None,
        canvas_width: Optional[int] = None,
        aspect: Optional[float] = None,
        show_cell_id: bool = False,
        video: bool = False,
    ) -> None:
        if radius < 2:
            raise ValueError("radius must be at least 2")

        self.radius = int(radius)
        self.cells: List[Axial] = []
        for q in range(-self.radius, self.radius + 1):
            for r in range(-self.radius, self.radius + 1):
                s = -q - r
                if max(abs(q), abs(r), abs(s)) <= self.radius:
                    self.cells.append((q, r))
        if not self.cells:
            raise ValueError("No cells generated for the requested radius")
        self.cell_set: Set[Axial] = set(self.cells)
        self.cell_to_id: Dict[Axial, int] = {cell: i for i, cell in enumerate(self.cells)}

        # Precompute unit grid spread for dimension calculations.
        min_uq, max_uq, min_ur, max_ur = float("inf"), float("-inf"), float("inf"), float("-inf")
        for q, r in self.cells:
            ux = 1.5 * q
            uy = SQRT_THREE * (r + q / 2.0)
            min_uq = min(min_uq, ux)
            max_uq = max(max_uq, ux)
            min_ur = min(min_ur, uy)
            max_ur = max(max_ur, uy)
        spread_x = max_uq - min_uq
        spread_y = max_ur - min_ur

        target_cr = int(
            cell_radius if cell_radius is not None else (size if size is not None else self.DEFAULT_CELL_RADIUS)
        )
        is_user_set = (cell_radius is not None) or (size is not None)

        def _get_layout_size(cr: int) -> Tuple[int, int]:
            wt = int(wall_thickness if wall_thickness is not None else max(6, cr // 6))
            # Total width = (spread + 2)*cr + 2*(cr+wt)
            w = int(math.ceil((spread_x + 2.0) * cr + 2.0 * wt))
            h = int(math.ceil((spread_y + 2.0) * cr + 2.0 * wt))
            return w, h

        def _check_fits(cr: int) -> bool:
            if canvas_width is None:
                return True
            bw, bh = _get_layout_size(cr)
            w_limit = int(canvas_width)
            if aspect is None:
                return w_limit >= bw - 1
            asp = float(aspect)
            min_w_needed = max(bw, int(math.ceil(bh * asp)))
            return w_limit >= min_w_needed - 1

        if not is_user_set and canvas_width is not None and not _check_fits(target_cr):
             for cr in range(target_cr - 1, 11, -1):
                 if _check_fits(cr):
                     target_cr = cr
                     break

        if target_cr < 12:
            raise ValueError("cell_radius (or auto-sized) must be at least 12 pixels to preserve visible walls")

        self.cell_radius = target_cr
        self.wall_thickness = int(wall_thickness if wall_thickness is not None else max(6, self.cell_radius // 6))
        
        if self.wall_thickness <= 0:
            raise ValueError("wall_thickness must be positive")

        self.walk_radius = max(4.0, self.cell_radius - self.wall_thickness * 0.8)
        self.outer_margin = self.cell_radius + self.wall_thickness * 3

        self._centers = {cell: self._axial_to_pixel(cell) for cell in self.cells}
        
        # We rely on the precomputed formula matching strictly:
        base_width, base_height = _get_layout_size(self.cell_radius)

        final_width, final_height = self._resolve_canvas_dimensions(
            base_width,
            base_height,
            canvas_width,
            aspect,
        )
        aspect_for_super = final_width / final_height

        resolved_output = output_dir if output_dir is not None else self.DEFAULT_OUTPUT_DIR
        super().__init__(
            resolved_output,
            canvas_width=final_width,
            aspect=aspect_for_super,
            size=self.cell_radius,
            seed=seed,
            prompt=prompt,
            show_cell_id=show_cell_id,
            video=video,
        )

        self.canvas_width_px = final_width
        self.canvas_height_px = final_height
        self.center = (final_width / 2.0, final_height / 2.0)
        self.cells_perimeter: List[Axial] = [cell for cell in self.cells if self._distance(cell) == self.radius]

    # ------------------------------------------------------------------
    # Canvas sizing helpers

    def _resolve_canvas_dimensions(
        self,
        natural_width: int,
        natural_height: int,
        canvas_width: Optional[int],
        aspect: Optional[float],
    ) -> Tuple[int, int]:
        if natural_width <= 0 or natural_height <= 0:
            raise ValueError("Natural canvas dimensions must be positive")

        width: int
        height: int

        if aspect is not None:
            aspect_value = float(aspect)
            if aspect_value <= 0:
                raise ValueError("aspect must be positive")
            min_width = max(natural_width, int(math.ceil(natural_height * aspect_value)))
            if canvas_width is not None:
                width = int(canvas_width)
                if width < min_width:
                    raise ValueError("canvas_width is too small for the requested aspect ratio")
            else:
                width = min_width
            height = int(math.ceil(width / aspect_value))
            if height < natural_height:
                height = natural_height
                width = int(math.ceil(height * aspect_value))
        else:
            if canvas_width is not None:
                width = int(canvas_width)
                if width < natural_width:
                    raise ValueError("canvas_width is too small for the hexagon layout")
                scale = width / float(natural_width)
                height = int(math.ceil(natural_height * scale))
            else:
                width = natural_width
                height = natural_height
        return max(width, natural_width), max(height, natural_height)

    # ------------------------------------------------------------------
    # Public API

    def create_puzzle(self, *, puzzle_id: Optional[str] = None) -> MazePuzzleRecord:
        passages = self._generate_passages()
        start_cell = self.rng.choice(self.cells_perimeter)
        goal_cell: Axial = (0, 0)
        solution = self._shortest_path(passages, start_cell, goal_cell)
        if not solution:
            raise RuntimeError("Failed to compute a connecting path in the hex maze")

        puzzle_image = self._render_maze(passages, start_cell, goal_cell, path=None)
        solution_image = self._render_maze(passages, start_cell, goal_cell, path=solution)

        record_id = puzzle_id or self.next_id()
        puzzle_path, solution_path = self.save_images(record_id, puzzle_image, solution_image)

        if self.video:
            path_points = [self._cell_center_from_cell(cell) for cell in solution]
            thickness = max(4, int(self.cell_radius * 0.35))
            self.save_video(record_id, puzzle_image, path_points, thickness=thickness)

        start_point = self._cell_center_from_cell(start_cell)
        goal_point = self._cell_center_from_cell(goal_cell)

        return self.build_record(
            record_id,
            start_point=start_point,
            goal_point=goal_point,
            puzzle_path=puzzle_path,
            solution_path=solution_path,
            prompt=self.prompt,
            extra={
                "radius": self.radius,
                "cell_radius": self.cell_radius,
                "wall_thickness": self.wall_thickness,
                "start_cell": list(start_cell),
                "goal_cell": list(goal_cell),
                "solution_path_cell_ids": [self.cell_to_id[cell] for cell in solution],
            },
        )

    # ------------------------------------------------------------------
    # Maze construction

    def _generate_passages(self) -> Dict[Axial, Set[Axial]]:
        passages: Dict[Axial, Set[Axial]] = {cell: set() for cell in self.cells}
        visited: Set[Axial] = set()

        def dfs(cell: Axial) -> None:
            visited.add(cell)
            neighbors = list(self._neighbors(cell))
            self.rng.shuffle(neighbors)
            for neighbor in neighbors:
                if neighbor not in visited:
                    passages[cell].add(neighbor)
                    passages[neighbor].add(cell)
                    dfs(neighbor)

        dfs((0, 0))
        return passages

    def _neighbors(self, cell: Axial) -> Iterable[Axial]:
        q, r = cell
        for dq, dr in DIRECTIONS:
            neighbor = (q + dq, r + dr)
            if neighbor in self.cell_set:
                yield neighbor

    def _shortest_path(
        self,
        passages: Dict[Axial, Set[Axial]],
        start: Axial,
        goal: Axial,
    ) -> List[Axial]:
        queue: Deque[Axial] = deque([start])
        parents: Dict[Axial, Optional[Axial]] = {start: None}
        while queue:
            current = queue.popleft()
            if current == goal:
                break
            for neighbor in passages[current]:
                if neighbor not in parents:
                    parents[neighbor] = current
                    queue.append(neighbor)
        if goal not in parents:
            return []
        path: List[Axial] = []
        step: Optional[Axial] = goal
        while step is not None:
            path.append(step)
            step = parents[step]
        path.reverse()
        return path

    # ------------------------------------------------------------------
    # Rendering

    def _render_maze(
        self,
        passages: Dict[Axial, Set[Axial]],
        start_cell: Axial,
        goal_cell: Axial,
        *,
        path: Optional[Sequence[Axial]],
    ) -> Image.Image:
        canvas = Image.new("RGB", (self.canvas_width_px, self.canvas_height_px), BACKGROUND_COLOR)
        draw = ImageDraw.Draw(canvas)

        for cell in self.cells:
            center = self._cell_center_from_cell(cell)
            corners = self._hex_corners(center, self.walk_radius*1.5)
            draw.polygon(corners, fill=PATH_COLOR)

        for cell in self.cells:
            self._draw_cell_walls(draw, cell, passages)

        self._draw_marker(draw, start_cell, START_COLOR)
        self._draw_marker(draw, goal_cell, GOAL_COLOR)
        if path:
            self._draw_solution(canvas, path)

        if self.show_cell_id:
            self._draw_cell_ids(draw)

        return canvas

    def _draw_cell_walls(
        self,
        draw: ImageDraw.ImageDraw,
        cell: Axial,
        passages: Dict[Axial, Set[Axial]],
    ) -> None:
        center = self._cell_center_from_cell(cell)
        outer_corners = self._hex_corners(center, self.cell_radius)
        for direction_index, delta in enumerate(DIRECTIONS):
            neighbor = (cell[0] + delta[0], cell[1] + delta[1])
            has_neighbor = neighbor in self.cell_set
            connected = has_neighbor and neighbor in passages[cell]
            if has_neighbor and connected:
                continue
            a_idx, b_idx = EDGE_CORNER_INDICES[direction_index]
            a = outer_corners[a_idx]
            b = outer_corners[b_idx]
            draw.line([a, b], fill=WALL_COLOR, width=self.wall_thickness)

    def _draw_marker(self, draw: ImageDraw.ImageDraw, cell: Axial, color: Tuple[int, int, int]) -> None:
        x, y = self._cell_center_from_cell(cell)
        radius = max(6, int(self.cell_radius * 0.45))
        bbox = (x - radius, y - radius, x + radius, y + radius)
        draw.ellipse(bbox, fill=color)

    def _draw_solution(self, image: Image.Image, path: Sequence[Axial]) -> None:
        if len(path) < 2:
            return
        points = [self._cell_center_from_cell(cell) for cell in path]
        thickness = max(4, int(self.cell_radius * 0.35))
        draw_path_line(image, points, LINE_COLOR, thickness)

    def _draw_cell_ids(self, draw: ImageDraw.ImageDraw) -> None:
        font_size = max(8, int(self.cell_radius * 0.4))
        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
        except OSError:
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except OSError:
                font = ImageFont.load_default()

        for cell in self.cells:
            text = str(self.cell_to_id[cell])
            cx, cy = self._cell_center_from_cell(cell)
            draw.text((cx, cy), text, fill=TEXT_COLOR, anchor="mm", font=font)

    # ------------------------------------------------------------------
    # Geometry helpers

    def _axial_to_pixel(self, cell: Axial) -> Tuple[float, float]:
        q, r = cell
        x = self.cell_radius * (1.5 * q)
        y = self.cell_radius * (SQRT_THREE * (r + q / 2.0))
        return x, y

    def _cell_center(self, cell_id: int) -> Tuple[float, float]:
        cell = self.cells[cell_id]
        return self._cell_center_from_cell(cell)

    def _cell_center_from_cell(self, cell: Axial) -> Tuple[float, float]:
        rel_x, rel_y = self._centers[cell]
        return (self.center[0] + rel_x, self.center[1] + rel_y)

    def _hex_corners(self, center: Tuple[float, float], radius: float) -> List[Tuple[float, float]]:
        cx, cy = center
        corners: List[Tuple[float, float]] = []
        for i in range(6):
            angle = math.radians(60 * i)
            corners.append((cx + radius * math.cos(angle), cy + radius * math.sin(angle)))
        return corners

    def _bounds(self) -> Tuple[float, float, float, float]:
        min_x = float("inf")
        min_y = float("inf")
        max_x = float("-inf")
        max_y = float("-inf")
        offset = self.cell_radius + self.wall_thickness
        for center in self._centers.values():
            x, y = center
            min_x = min(min_x, x - offset)
            min_y = min(min_y, y - offset)
            max_x = max(max_x, x + offset)
            max_y = max(max_y, y + offset)
        return min_x, min_y, max_x, max_y

    def _distance(self, cell: Axial) -> int:
        q, r = cell
        s = -q - r
        return max(abs(q), abs(r), abs(s))

    # ------------------------------------------------------------------
    # CLI integration

    @classmethod
    def _parse_args(cls, argv: Optional[List[str]] = None) -> argparse.Namespace:
        parser = argparse.ArgumentParser(description="Generate hexagonal maze puzzles")
        parser.add_argument("count", type=int, help="Number of puzzles to generate")
        parser.add_argument("--output-dir", type=str, default=None)
        parser.add_argument("--radius", type=int, default=cls.DEFAULT_RADIUS, help="Number of rings from center to edge")
        parser.add_argument("--cell-radius", type=int, default=None, help="Pixel radius of each hex cell")
        parser.add_argument("--wall-thickness", type=int, default=None, help="Thickness of wall lines in pixels")
        parser.add_argument("--size", type=int, default=None, help="Alias for --cell-radius to align with shared interface")
        parser.add_argument("--canvas-width", type=int, default=None, help="Optional override for final canvas width")
        parser.add_argument("--aspect", type=float, default=None, help="Desired width/height ratio for final canvas")
        parser.add_argument("--seed", type=int, default=None)
        parser.add_argument("--prompt", type=str, default=None)
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
        cell_radius = (
            args.cell_radius if args.cell_radius is not None else args.size
        )
        prompt_arg = args.prompt if args.prompt is not None else cls.DEFAULT_PROMPT
        generator = cls(
            output_dir=args.output_dir,
            radius=args.radius,
            cell_radius=cell_radius,
            wall_thickness=args.wall_thickness,
            seed=args.seed,
            prompt=prompt_arg,
            canvas_width=args.canvas_width,
            aspect=args.aspect,
            show_cell_id=args.show_cell_id,
            video=args.video,
        )
        records = [generator.create_random_puzzle() for _ in range(max(1, args.count))]
        generator.write_metadata(records, generator.output_dir / "data.json")


__all__ = ["MazeHexagonGenerator"]


def main(argv: Optional[List[str]] = None) -> None:
    MazeHexagonGenerator.main(argv)


if __name__ == "__main__":
    MazeHexagonGenerator.main()
