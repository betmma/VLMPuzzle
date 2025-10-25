"""Shared scaffolding for maze-style puzzle generators and evaluators.

These base classes implement reusable logic for maze puzzles where a solver must
trace a red path from a designated start to a goal while avoiding walls drawn in
black. Subclasses can focus on maze layout generation and rendering while
inheriting dataset serialization, CLI wiring, and pixel-level evaluation.
"""

from __future__ import annotations

import argparse
import json
import random
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image

from .base import AbstractPuzzleEvaluator, AbstractPuzzleGenerator, PathLike


@dataclass
class MazePuzzleRecord:
    """Serializable metadata for a maze puzzle asset pair."""

    id: str
    prompt: str
    canvas_dimensions: Tuple[int, int]
    start_point: Tuple[float, float]
    goal_point: Tuple[float, float]
    image: str
    solution_image_path: str
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "id": self.id,
            "prompt": self.prompt,
            "canvas_dimensions": [int(self.canvas_dimensions[0]), int(self.canvas_dimensions[1])],
            "start_point": [float(self.start_point[0]), float(self.start_point[1])],
            "goal_point": [float(self.goal_point[0]), float(self.goal_point[1])],
            "image": self.image,
            "solution_image_path": self.solution_image_path,
        }
        for key, value in self.extra.items():
            if key not in payload:
                payload[key] = value
        return payload


class MazePuzzleGenerator(AbstractPuzzleGenerator[MazePuzzleRecord]):
    """Base generator providing canvas configuration and asset management."""

    DEFAULT_OUTPUT_DIR: Optional[PathLike] = "data/maze"
    DEFAULT_PROMPT: Optional[str] = "Draw a red path connecting two red dots without touching the black walls. In portrait. Static camera."

    def __init__(
        self,
        output_dir: Optional[PathLike] = None,
        *,
        canvas_width: int = 512,
        aspect: Optional[float] = None,
        size: int = 32,
        seed: Optional[int] = None,
        prompt: Optional[str] = None,
    ) -> None:
        resolved_output = output_dir if output_dir is not None else self.DEFAULT_OUTPUT_DIR
        if resolved_output is None:
            raise ValueError("output_dir must be provided either via argument or DEFAULT_OUTPUT_DIR")
        super().__init__(resolved_output)

        if canvas_width <= 0:
            raise ValueError("canvas_width must be positive")
        width = int(canvas_width)
        if aspect is not None and aspect <= 0:
            raise ValueError("aspect must be positive when provided")
        if aspect is None:
            height = width
        else:
            height = int(round(width / float(aspect)))
            if height <= 0:
                raise ValueError("Derived canvas height must be positive")
        self.canvas_dimensions: Tuple[int, int] = (width, height)

        if size <= 0:
            raise ValueError("size must be positive")
        self.size = int(size)
        self.prompt = prompt if prompt is not None else (self.DEFAULT_PROMPT or "")
        self._rng = random.Random(seed)

        root = Path(self.output_dir)
        self.puzzle_dir = root / "puzzles"
        self.solution_dir = root / "solutions"
        self.puzzle_dir.mkdir(parents=True, exist_ok=True)
        self.solution_dir.mkdir(parents=True, exist_ok=True)

    @property
    def rng(self) -> random.Random:
        return self._rng

    @property
    def canvas_width(self) -> int:
        return self.canvas_dimensions[0]

    @property
    def canvas_height(self) -> int:
        return self.canvas_dimensions[1]

    def next_id(self) -> str:
        return str(uuid.uuid4())

    def save_images(
        self,
        record_id: str,
        puzzle_image: Image.Image,
        solution_image: Image.Image,
    ) -> Tuple[Path, Path]:
        puzzle_path = self.puzzle_dir / f"{record_id}_puzzle.png"
        solution_path = self.solution_dir / f"{record_id}_solution.png"
        puzzle_image.save(puzzle_path)
        solution_image.save(solution_path)
        return puzzle_path, solution_path

    def build_record(
        self,
        record_id: str,
        *,
        start_point: Tuple[float, float],
        goal_point: Tuple[float, float],
        puzzle_path: Path,
        solution_path: Path,
        prompt: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> MazePuzzleRecord:
        record_prompt = prompt if prompt is not None else self.prompt
        extra_payload = extra if extra is not None else {}
        return MazePuzzleRecord(
            id=record_id,
            prompt=record_prompt,
            canvas_dimensions=self.canvas_dimensions,
            start_point=start_point,
            goal_point=goal_point,
            image=self.relativize_path(puzzle_path),
            solution_image_path=self.relativize_path(solution_path),
            extra=extra_payload,
        )

    @classmethod
    def _parse_args(cls, argv: Optional[List[str]] = None) -> argparse.Namespace:
        parser = argparse.ArgumentParser(description="Generate maze puzzles")
        parser.add_argument("count", type=int, help="Number of puzzles to create")
        parser.add_argument("--output-dir", type=Path, default=None)
        parser.add_argument("--canvas-width", type=int, default=550)
        parser.add_argument("--aspect", type=float, default=0.55)
        parser.add_argument("--size", type=int, default=32, help="Primary maze sizing parameter (e.g., cell size or radius)")
        parser.add_argument("--seed", type=int, default=None)
        parser.add_argument("--prompt", type=str, default=None)
        return parser.parse_args(argv)

    @classmethod
    def main(cls, argv: Optional[List[str]] = None) -> None:
        args = cls._parse_args(argv)
        prompt_arg = args.prompt if args.prompt is not None else cls.DEFAULT_PROMPT
        output_arg = args.output_dir if args.output_dir is not None else cls.DEFAULT_OUTPUT_DIR
        generator = cls(
            output_dir=output_arg,
            canvas_width=args.canvas_width,
            aspect=args.aspect,
            size=args.size,
            seed=args.seed,
            prompt=prompt_arg,
        )
        records = [generator.create_random_puzzle() for _ in range(max(1, args.count))]
        generator.write_metadata(records, generator.output_dir / "data.json")


@dataclass
class MazeEvaluationResult:
    puzzle_id: str
    red_pixel_count: int
    overlaps_walls: bool
    touches_start: bool
    touches_goal: bool
    connected: bool
    message: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "puzzle_id": self.puzzle_id,
            "red_pixel_count": self.red_pixel_count,
            "overlaps_walls": self.overlaps_walls,
            "touches_start": self.touches_start,
            "touches_goal": self.touches_goal,
            "connected": self.connected,
            "message": self.message,
        }


class MazePuzzleEvaluator(AbstractPuzzleEvaluator):
    """Pixel-based evaluation for maze puzzles."""

    RED_THRESHOLD: int = 150
    RED_DOMINANCE: int = 70
    WALL_VALUE_THRESHOLD: int = 40
    ENDPOINT_SEARCH_RADIUS: int = 4

    def evaluate(
        self,
        puzzle_id: str,
        candidate_image: PathLike,
    ) -> MazeEvaluationResult:
        record = self.get_record(puzzle_id)
        candidate_path = Path(candidate_image)
        if not candidate_path.exists():
            raise FileNotFoundError(f"Candidate image not found: {candidate_path}")

        source_path = self.resolve_path(record.get("image"))
        if not source_path.exists():
            raise FileNotFoundError(f"Puzzle image not found: {source_path}")

        with Image.open(source_path) as src_image:
            puzzle_image = src_image.convert("RGB")
        source_canvas_size = puzzle_image.size
        with Image.open(candidate_path) as cand_image:
            candidate_image_rgb = cand_image.convert("RGB")

        if candidate_image_rgb.size != puzzle_image.size:
            puzzle_image = puzzle_image.resize(candidate_image_rgb.size, Image.NEAREST)

        puzzle_pixels = np.asarray(puzzle_image, dtype=np.uint8)
        candidate_pixels = np.asarray(candidate_image_rgb, dtype=np.uint8)

        red_mask = self._red_mask(candidate_pixels)
        red_pixel_count = int(red_mask.sum())
        if red_pixel_count == 0:
            self._resolve_endpoint(record, "start", source_canvas_size, candidate_image_rgb.size)
            self._resolve_endpoint(record, "goal", source_canvas_size, candidate_image_rgb.size)
            return MazeEvaluationResult(
                puzzle_id=puzzle_id,
                red_pixel_count=0,
                overlaps_walls=False,
                touches_start=False,
                touches_goal=False,
                connected=False,
                message="No red path detected.",
            )

        wall_mask = self._wall_mask(puzzle_pixels)
        overlaps_walls = bool(np.any(red_mask & wall_mask))

        start_point = self._resolve_endpoint(record, "start", source_canvas_size, candidate_image_rgb.size)
        goal_point = self._resolve_endpoint(record, "goal", source_canvas_size, candidate_image_rgb.size)

        start_seed = self._nearest_red(red_mask, start_point)
        goal_seed = self._nearest_red(red_mask, goal_point)
        touches_start = start_seed is not None
        touches_goal = goal_seed is not None

        connected = False
        if touches_start and touches_goal and not overlaps_walls:
            connected = self._connected(red_mask, start_seed, goal_seed)

        if overlaps_walls:
            message = "Red path overlaps walls."
        elif not touches_start:
            message = "Red path does not reach the start."
        elif not touches_goal:
            message = "Red path does not reach the goal."
        elif not connected:
            message = "Red path is not continuous between start and goal."
        else:
            message = "Red path successfully connects start to goal."

        return MazeEvaluationResult(
            puzzle_id=puzzle_id,
            red_pixel_count=red_pixel_count,
            overlaps_walls=overlaps_walls,
            touches_start=touches_start,
            touches_goal=touches_goal,
            connected=connected,
            message=message,
        )

    def _red_mask(self, pixels: np.ndarray) -> np.ndarray:
        red = pixels[:, :, 0].astype(np.int32)
        green = pixels[:, :, 1].astype(np.int32)
        blue = pixels[:, :, 2].astype(np.int32)
        dominance = red - np.maximum(green, blue)
        mask = (red >= self.RED_THRESHOLD) & (dominance >= self.RED_DOMINANCE)
        return mask

    def _wall_mask(self, pixels: np.ndarray) -> np.ndarray:
        max_channel = pixels.max(axis=2)
        return max_channel <= self.WALL_VALUE_THRESHOLD

    def _resolve_endpoint(
        self,
        record: Dict[str, Any],
        label: str,
        puzzle_size: Tuple[int, int],
        candidate_size: Tuple[int, int],
    ) -> Tuple[float, float]:
        pixel_key = f"{label}_pixel"
        if pixel_key in record:
            point = self._coerce_point(record[pixel_key])
            return self._scale_point(point, puzzle_size, candidate_size)
        point_key = f"{label}_point"
        if point_key in record:
            point = self._coerce_point(record[point_key])
            return self._scale_point(point, puzzle_size, candidate_size)
        if label in record and "cell_bboxes" in record:
            cell = record[label]
            if not isinstance(cell, Iterable):
                raise ValueError(f"Record contains invalid '{label}' entry")
            cell_list = list(cell)
            if len(cell_list) < 2:
                raise ValueError(f"Record '{label}' does not contain row and column")
            row = int(cell_list[0])
            col = int(cell_list[1])
            bbox_rows = record["cell_bboxes"]
            if not isinstance(bbox_rows, Iterable):
                raise ValueError("cell_bboxes must be iterable")
            bbox_list = list(bbox_rows)
            if row < 0 or row >= len(bbox_list):
                raise ValueError("start or goal row index out of range")
            row_data = bbox_list[row]
            if not isinstance(row_data, Iterable):
                raise ValueError("cell row is not iterable")
            row_cells = list(row_data)
            if col < 0 or col >= len(row_cells):
                raise ValueError("start or goal column index out of range")
            bbox = row_cells[col]
            if not isinstance(bbox, Iterable):
                raise ValueError("cell bbox is not iterable")
            bbox_values = list(bbox)
            if len(bbox_values) < 4:
                raise ValueError("cell bbox must contain four coordinates")
            left = float(bbox_values[0])
            top = float(bbox_values[1])
            right = float(bbox_values[2])
            bottom = float(bbox_values[3])
            center = ((left + right) * 0.5, (top + bottom) * 0.5)
            return self._scale_point(center, puzzle_size, candidate_size)
        raise KeyError(f"Record is missing endpoint information for '{label}'")

    def _coerce_point(self, value: Any) -> Tuple[float, float]:
        if isinstance(value, dict):
            if "x" in value and "y" in value:
                return float(value["x"]), float(value["y"])
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            return float(value[0]), float(value[1])
        raise ValueError("Endpoint entries must contain two numeric values")

    def _scale_point(
        self,
        point: Tuple[float, float],
        source_size: Tuple[int, int],
        target_size: Tuple[int, int],
    ) -> Tuple[float, float]:
        source_width, source_height = source_size
        target_width, target_height = target_size
        if source_width <= 0 or source_height <= 0:
            raise ValueError("Source dimensions must be positive")
        scale_x = target_width / float(source_width)
        scale_y = target_height / float(source_height)
        return point[0] * scale_x, point[1] * scale_y

    def _nearest_red(
        self,
        red_mask: np.ndarray,
        point: Tuple[float, float],
    ) -> Optional[Tuple[int, int]]:
        height, width = red_mask.shape
        cx = int(round(point[0]))
        cy = int(round(point[1]))
        cx = max(0, min(width - 1, cx))
        cy = max(0, min(height - 1, cy))
        if red_mask[cy, cx]:
            return (cy, cx)
        radius = self.ENDPOINT_SEARCH_RADIUS
        for delta in range(1, radius + 1):
            min_x = max(0, cx - delta)
            max_x = min(width - 1, cx + delta)
            min_y = max(0, cy - delta)
            max_y = min(height - 1, cy + delta)
            for y in range(min_y, max_y + 1):
                if red_mask[y, min_x]:
                    return (y, min_x)
                if red_mask[y, max_x]:
                    return (y, max_x)
            for x in range(min_x + 1, max_x):
                if red_mask[min_y, x]:
                    return (min_y, x)
                if red_mask[max_y, x]:
                    return (max_y, x)
        return None

    def _connected(
        self,
        red_mask: np.ndarray,
        start_seed: Tuple[int, int],
        goal_seed: Tuple[int, int],
    ) -> bool:
        height, width = red_mask.shape
        visited = np.zeros((height, width), dtype=bool)
        queue: deque[Tuple[int, int]] = deque([start_seed])
        visited[start_seed[0], start_seed[1]] = True
        while queue:
            y, x = queue.popleft()
            if (y, x) == goal_seed:
                return True
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny = y + dy
                nx = x + dx
                if 0 <= ny < height and 0 <= nx < width:
                    if red_mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        queue.append((ny, nx))
        return False

    @classmethod
    def _parse_args(cls, argv: Optional[List[str]] = None) -> argparse.Namespace:
        parser = argparse.ArgumentParser(description="Evaluate maze puzzles")
        parser.add_argument("metadata", type=Path, help="Path to maze metadata JSON")
        parser.add_argument("puzzle_id", type=str, help="Identifier of the puzzle to evaluate")
        parser.add_argument("candidate", type=Path, help="Candidate solution image path")
        parser.add_argument("--base-dir", type=Path, default=None)
        return parser.parse_args(argv)

    @classmethod
    def main(cls, argv: Optional[List[str]] = None) -> None:
        args = cls._parse_args(argv)
        evaluator = cls(args.metadata, base_dir=args.base_dir)
        result = evaluator.evaluate(args.puzzle_id, args.candidate)
        print(json.dumps(result.to_dict(), indent=2))


__all__ = [
    "MazePuzzleRecord",
    "MazePuzzleGenerator",
    "MazeEvaluationResult",
    "MazePuzzleEvaluator",
]
