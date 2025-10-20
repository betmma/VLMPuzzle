"""Circle count puzzle generator.

Produces a simple scene with multiple colored circles on a blank canvas.
Solvers are instructed to speak the total number of circles they see.
"""

from __future__ import annotations

import argparse
import random
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont

from ..base import AbstractPuzzleGenerator, PathLike


@dataclass
class CircleSpec:
    cx: float
    cy: float
    radius: float
    color: Tuple[int, int, int]

    def bbox(self) -> Tuple[int, int, int, int]:
        return (
            int(round(self.cx - self.radius)),
            int(round(self.cy - self.radius)),
            int(round(self.cx + self.radius)),
            int(round(self.cy + self.radius)),
        )

    def to_dict(self) -> dict:
        return {
            "cx": self.cx,
            "cy": self.cy,
            "radius": self.radius,
            "color": list(self.color),
        }


@dataclass
class CircleCountPuzzleRecord:
    id: str
    prompt: str
    canvas_dimensions: Tuple[int, int]
    circle_count: int
    circles: List[CircleSpec]
    image: str
    solution_image_path: str

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "prompt": self.prompt,
            "canvas_dimensions": list(self.canvas_dimensions),
            "circle_count": self.circle_count,
            "circles": [circle.to_dict() for circle in self.circles],
            "image": self.image,
            "solution_image_path": self.solution_image_path,
            "type": "circle_count",
        }


class CircleCountGenerator(AbstractPuzzleGenerator[CircleCountPuzzleRecord]):
    def __init__(
        self,
        output_dir: PathLike = "data/circle_count",
        *,
        canvas_width: int = 480,
        aspect: Optional[float] = None,
        prompt: Optional[str] = None,
        seed: Optional[int] = None,
        circle_range: Tuple[int, int] = (6, 10),
    ) -> None:
        super().__init__(output_dir)

        width = int(canvas_width)
        if width < 160:
            width = 160
        if aspect and aspect > 0:
            height = int(round(width / float(aspect)))
        else:
            height = width
        self.canvas_dimensions = (width, height)
        self.margin = max(12, int(round(min(width, height) * 0.05)))
        self._rng = random.Random(seed)
        self.circle_range = circle_range
        if prompt is None:
            prompt = "Speak out how many circles are in the image"
        self.prompt = prompt

        out_root = Path(self.output_dir)
        self.puzzle_dir = out_root / "puzzles"
        self.solution_dir = out_root / "solutions"
        self.puzzle_dir.mkdir(parents=True, exist_ok=True)
        self.solution_dir.mkdir(parents=True, exist_ok=True)

    def create_puzzle(self, *, puzzle_id: Optional[str] = None) -> CircleCountPuzzleRecord:
        puzzle_id = puzzle_id or str(uuid.uuid4())
        circle_count = self._rng.randint(self.circle_range[0], self.circle_range[1])
        circles = self._build_circles(circle_count)

        puzzle_img = self._render(circles)
        solution_img = self._render_solution(puzzle_img, circle_count)

        puzzle_path = self.puzzle_dir / f"{puzzle_id}_puzzle.png"
        solution_path = self.solution_dir / f"{puzzle_id}_solution.png"
        puzzle_img.save(puzzle_path)
        solution_img.save(solution_path)

        return CircleCountPuzzleRecord(
            id=puzzle_id,
            prompt=self.prompt,
            canvas_dimensions=self.canvas_dimensions,
            circle_count=circle_count,
            circles=circles,
            image=self.relativize_path(puzzle_path),
            solution_image_path=self.relativize_path(solution_path),
        )

    def create_random_puzzle(self) -> CircleCountPuzzleRecord:
        return self.create_puzzle()

    def _build_circles(self, count: int) -> List[CircleSpec]:
        width, height = self.canvas_dimensions
        min_radius = max(12.0, 0.15 * min(width, height))
        max_radius = max(min_radius + 4.0, 0.4 * min(width, height))
        circles: List[CircleSpec] = []
        for _ in range(count):
            radius = self._rng.uniform(min_radius, max_radius)
            cx = self._rng.uniform(self.margin + radius, width - self.margin - radius)
            cy = self._rng.uniform(self.margin + radius, height - self.margin - radius)
            color = self._random_color()
            circles.append(CircleSpec(cx=cx, cy=cy, radius=radius, color=color))
        return circles

    def _random_color(self) -> Tuple[int, int, int]:
        palette = [
            (66, 135, 245),
            (250, 128, 114),
            (60, 179, 113),
            (238, 130, 238),
            (255, 215, 0),
            (255, 140, 0),
        ]
        return palette[self._rng.randrange(len(palette))]

    def _render(self, circles: Sequence[CircleSpec]) -> Image.Image:
        width, height = self.canvas_dimensions
        image = Image.new("RGB", (width, height), (255, 255, 255))
        draw = ImageDraw.Draw(image)
        for circle in circles:
            bbox = circle.bbox()
            draw.ellipse(bbox, outline=(30, 30, 30), width=2) # fill=circle.color, 
        return image

    def _render_solution(self, puzzle_image: Image.Image, circle_count: int) -> Image.Image:
        width, height = puzzle_image.size
        solution = puzzle_image.copy()
        overlay = ImageDraw.Draw(solution)
        font = ImageFont.load_default(20)
        label = f"Total circles: {circle_count}"
        bbox = overlay.textbbox((0, 0), label, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        padding = max(8, int(round(min(width, height) * 0.02)))
        rect_width = text_width + 2 * padding
        rect_height = text_height + 2 * padding
        rect_left = (width - rect_width) // 2
        rect_top = height - rect_height - padding
        rect = (
            rect_left,
            rect_top,
            rect_left + rect_width,
            rect_top + rect_height,
        )
        overlay.rectangle(rect, fill=(255, 255, 255))
        text_position = (
            rect_left + padding,
            rect_top + padding,
        )
        overlay.text(text_position, label, fill=(20, 20, 20), font=font)
        return solution


__all__ = [
    "CircleCountGenerator",
    "CircleCountPuzzleRecord",
    "CircleSpec",
]


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate circle count puzzles")
    parser.add_argument("count", type=int, help="Number of puzzles to create")
    parser.add_argument("--output-dir", type=Path, default=Path("data/circle_count"))
    parser.add_argument("--canvas-width", type=int, default=480)
    parser.add_argument("--aspect", type=float, default=None, help="Canvas aspect ratio W/H")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--prompt", type=str, default=None)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)
    generator = CircleCountGenerator(
        args.output_dir,
        canvas_width=args.canvas_width,
        aspect=args.aspect,
        prompt=args.prompt,
        seed=args.seed,
    )
    records = [generator.create_random_puzzle() for _ in range(max(1, args.count))]
    generator.write_metadata(records, Path(args.output_dir) / "data.json")


if __name__ == "__main__":
    main()
