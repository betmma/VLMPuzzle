"""Ray intersection puzzle generator.

Three partial rays originate from a hidden intersection point near the canvas
center. Only the edge-adjacent segments of the rays are drawn so solvers must
extend them mentally to locate the true intersection. Five circled options
(A–E) are rendered near the hidden point; exactly one is positioned at the
actual intersection. Prompt instructs respondents to extend the lines, mark the
intersection in red, and report the option using the phonetic alphabet.
"""

from __future__ import annotations

import argparse
import math
import random
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont

from ..base import AbstractPuzzleGenerator, PathLike


@dataclass
class RaySegment:
    angle: float
    start: Tuple[float, float]
    end: Tuple[float, float]

    def to_dict(self) -> dict:
        return {
            "angle": self.angle,
            "start": list(self.start),
            "end": list(self.end),
        }


@dataclass
class CandidatePoint:
    x: float
    y: float
    label: str

    def to_dict(self) -> dict:
        return {"x": self.x, "y": self.y, "label": self.label}


@dataclass
class RayIntersectionPuzzleRecord:
    id: str
    prompt: str
    canvas_dimensions: Tuple[int, int]
    margin: int
    intersection: Tuple[float, float]
    rays: List[RaySegment]
    candidates: List[CandidatePoint]
    point_radius: int
    correct_option: str
    puzzle_image_path: str
    solution_image_path: str

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "prompt": self.prompt,
            "canvas_dimensions": list(self.canvas_dimensions),
            "margin": self.margin,
            "intersection": list(self.intersection),
            "rays": [ray.to_dict() for ray in self.rays],
            "candidates": [c.to_dict() for c in self.candidates],
            "point_radius": self.point_radius,
            "correct_option": self.correct_option,
            "puzzle_image_path": self.puzzle_image_path,
            "solution_image_path": self.solution_image_path,
            "type": "ray_intersection",
        }


class RayIntersectionGenerator(AbstractPuzzleGenerator[RayIntersectionPuzzleRecord]):
    """Generate puzzles with partially hidden ray intersections."""

    def __init__(
        self,
        output_dir: PathLike = "data/ray_intersection",
        *,
        canvas_width: int = 480,
        aspect: Optional[float] = None,
        seed: Optional[int] = None,
        prompt: Optional[str] = None,
    ) -> None:
        super().__init__(output_dir)

        width = int(canvas_width)
        if aspect and aspect > 0:
            height = int(round(width / float(aspect)))
        else:
            height = width
        self.canvas_dimensions = (width, height)
        self.margin = max(18, int(round(min(width, height) * 0.06)))
        self._rng = random.Random(seed)

        if prompt is None:
            prompt = (
                "Extend the three lines and mark the intersection point as red. "
                "Speak out which option is the intersection point using phonetics alphabet"
            )
        self.prompt = prompt

        out_root = Path(self.output_dir)
        self.puzzle_dir = out_root / "puzzles"
        self.solution_dir = out_root / "solutions"
        self.puzzle_dir.mkdir(parents=True, exist_ok=True)
        self.solution_dir.mkdir(parents=True, exist_ok=True)

    def create_puzzle(self, *, puzzle_id: Optional[str] = None) -> RayIntersectionPuzzleRecord:
        width, height = self.canvas_dimensions
        left = self.margin
        top = self.margin
        right = width - self.margin
        bottom = height - self.margin

        intersection = self._pick_intersection(left, top, right, bottom)
        rays = self._build_rays(intersection, left, top, right, bottom)
        point_radius = 10
        candidates, correct_label = self._place_candidates(intersection, point_radius, left, top, right, bottom)

        pid = puzzle_id or str(uuid.uuid4())
        puzzle_img = self._render(
            intersection=intersection,
            rays=rays,
            candidates=candidates,
            point_radius=point_radius,
            highlight_label=None,
        )
        solution_img = self._render(
            intersection=intersection,
            rays=rays,
            candidates=candidates,
            point_radius=point_radius,
            highlight_label=correct_label,
        )

        puzzle_path = self.puzzle_dir / f"{pid}_puzzle.png"
        solution_path = self.solution_dir / f"{pid}_solution.png"
        puzzle_img.save(puzzle_path)
        solution_img.save(solution_path)

        return RayIntersectionPuzzleRecord(
            id=pid,
            prompt=self.prompt,
            canvas_dimensions=self.canvas_dimensions,
            margin=self.margin,
            intersection=intersection,
            rays=rays,
            candidates=candidates,
            point_radius=point_radius,
            correct_option=correct_label,
            puzzle_image_path=self.relativize_path(puzzle_path),
            solution_image_path=self.relativize_path(solution_path),
        )

    def create_random_puzzle(self) -> RayIntersectionPuzzleRecord:
        return self.create_puzzle()

    def _pick_intersection(self, left: int, top: int, right: int, bottom: int) -> Tuple[float, float]:
        width = right - left
        height = bottom - top
        center_x = left + width * 0.5
        center_y = top + height * 0.5
        jitter_x = self._rng.uniform(-0.18 * width, 0.18 * width)
        jitter_y = self._rng.uniform(-0.18 * height, 0.18 * height)
        x = center_x + jitter_x
        y = center_y + jitter_y
        x = min(max(left + width * 0.1, x), right - width * 0.1)
        y = min(max(top + height * 0.1, y), bottom - height * 0.1)
        return (x, y)

    def _build_rays(
        self,
        intersection: Tuple[float, float],
        left: int,
        top: int,
        right: int,
        bottom: int,
    ) -> List[RaySegment]:
        min_sep = math.radians(35.0)
        angles: List[float] = []
        attempts = 0
        while len(angles) < 3 and attempts < 400:
            attempts += 1
            angle = self._rng.uniform(0.0, math.tau)
            if all(self._angle_distance(angle, existing) >= min_sep for existing in angles):
                angles.append(angle)
        if len(angles) < 3:
            base = [0.0, 2.09439510239, 4.18879020479]
            shift = self._rng.uniform(-0.3, 0.3)
            angles = [(value + shift) % math.tau for value in base]

        segments: List[RaySegment] = []
        for angle in angles:
            end_point = self._ray_to_bounds(intersection, angle, left, top, right, bottom)
            start_point = self._edge_segment_start(intersection, end_point)
            segments.append(RaySegment(angle=angle, start=start_point, end=end_point))
        return segments

    def _ray_to_bounds(
        self,
        origin: Tuple[float, float],
        angle: float,
        left: int,
        top: int,
        right: int,
        bottom: int,
    ) -> Tuple[float, float]:
        ox, oy = origin
        dx = math.cos(angle)
        dy = math.sin(angle)
        best_t = float("inf")
        hit_x = ox
        hit_y = oy

        if dx > 0:
            t = (right - ox) / dx
            y_intercept = oy + t * dy
            if t > 0 and top <= y_intercept <= bottom and t < best_t:
                best_t = t
                hit_x = right
                hit_y = y_intercept
        if dx < 0:
            t = (left - ox) / dx
            y_intercept = oy + t * dy
            if t > 0 and top <= y_intercept <= bottom and t < best_t:
                best_t = t
                hit_x = left
                hit_y = y_intercept
        if dy > 0:
            t = (bottom - oy) / dy
            x_intercept = ox + t * dx
            if t > 0 and left <= x_intercept <= right and t < best_t:
                best_t = t
                hit_x = x_intercept
                hit_y = bottom
        if dy < 0:
            t = (top - oy) / dy
            x_intercept = ox + t * dx
            if t > 0 and left <= x_intercept <= right and t < best_t:
                best_t = t
                hit_x = x_intercept
                hit_y = top
        return (hit_x, hit_y)

    def _edge_segment_start(self, origin: Tuple[float, float], end_point: Tuple[float, float]) -> Tuple[float, float]:
        ox, oy = origin
        ex, ey = end_point
        dx = ex - ox
        dy = ey - oy
        span = math.hypot(dx, dy)
        if span <= 1.0:
            return (ox, oy)
        draw_fraction = self._rng.uniform(0.25, 0.35)
        start_dist = span * (1.0 - draw_fraction)
        ratio = start_dist / span
        sx = ox + dx * ratio
        sy = oy + dy * ratio
        return (sx, sy)

    def _place_candidates(
        self,
        intersection: Tuple[float, float],
        radius: int,
        left: int,
        top: int,
        right: int,
        bottom: int,
    ) -> Tuple[List[CandidatePoint], str]:
        base_x, base_y = intersection
        letters = list("ABCDE")
        self._rng.shuffle(letters)
        correct_label = letters[0]
        candidates: List[CandidatePoint] = []
        candidates.append(CandidatePoint(x=base_x, y=base_y, label=correct_label))

        max_attempts = 600
        attempt = 0
        spread = max(18.0, 0.9 * radius)
        while len(candidates) < 5 and attempt < max_attempts:
            attempt += 1
            angle = self._rng.uniform(0.0, math.tau)
            distance = self._rng.uniform(spread * 0.8, spread * 1.8)
            cx = base_x + math.cos(angle) * distance
            cy = base_y + math.sin(angle) * distance
            if not (left + radius <= cx <= right - radius and top + radius <= cy <= bottom - radius):
                continue
            too_close = False
            for existing in candidates:
                if math.hypot(existing.x - cx, existing.y - cy) < radius * 1.2:
                    too_close = True
                    break
            if too_close:
                continue
            label = letters[len(candidates)]
            candidates.append(CandidatePoint(x=cx, y=cy, label=label))
            if attempt==1 and self._rng.random()<0.8:
                base_x, base_y = cx, cy
        if len(candidates) < 5:
            padding = radius * 1.6
            needed = 5 - len(candidates)
            for i in range(needed):
                shift_x = padding * (1 if i % 2 == 0 else -1)
                shift_y = padding * ((i // 2) % 2 * 2 - 1)
                cx = min(max(left + radius, base_x + shift_x), right - radius)
                cy = min(max(top + radius, base_y + shift_y), bottom - radius)
                label = letters[len(candidates)]
                candidates.append(CandidatePoint(x=cx, y=cy, label=label))
        return candidates, correct_label

    def _render(
        self,
        *,
        intersection: Tuple[float, float],
        rays: Sequence[RaySegment],
        candidates: Sequence[CandidatePoint],
        point_radius: int,
        highlight_label: Optional[str],
    ) -> Image.Image:
        width, height = self.canvas_dimensions
        base = Image.new("RGB", (width, height), (255, 255, 255))
        draw = ImageDraw.Draw(base)

        stroke_color = (40, 40, 40)
        stroke_width = max(3, int(round(min(width, height) * 0.015)))

        for ray in rays:
            draw.line(
                [
                    (int(round(ray.start[0])), int(round(ray.start[1]))),
                    (int(round(ray.end[0])), int(round(ray.end[1]))),
                ],
                fill=stroke_color,
                width=stroke_width,
            )

        font = ImageFont.load_default(15)
        outline_color = (32, 32, 32)
        text_color = (0, 0, 0)
        highlight_color = (198, 24, 24)

        for candidate in sorted(candidates, key=lambda c: c.label):
            cx = int(round(candidate.x))
            cy = int(round(candidate.y))
            bbox = (cx - point_radius, cy - point_radius, cx + point_radius, cy + point_radius)
            if highlight_label is not None and candidate.label == highlight_label:
                fill_color = (255, 220, 220)
                draw.ellipse(bbox, fill=fill_color)
                draw.ellipse(bbox, outline=highlight_color, width=max(3, stroke_width // 2))
            else:
                draw.ellipse(bbox, outline=outline_color, width=max(3, stroke_width // 2), fill=(255, 255, 255))
            text_bbox = font.getbbox(candidate.label)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            tx = cx - text_width // 2
            ty = cy - text_height 
            draw.text((tx, ty), candidate.label, fill=text_color, font=font)

        return base

    @staticmethod
    def _angle_distance(a: float, b: float) -> float:
        diff = abs(a - b) % math.tau
        if diff > math.pi:
            diff = math.tau - diff
        return diff


__all__ = [
    "RayIntersectionGenerator",
    "RayIntersectionPuzzleRecord",
    "RaySegment",
    "CandidatePoint",
]


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ray intersection puzzles")
    parser.add_argument("count", type=int, help="Number of puzzles to create")
    parser.add_argument("--output-dir", type=Path, default=Path("data/ray_intersection"))
    parser.add_argument("--canvas-width", type=int, default=480)
    parser.add_argument("--aspect", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--prompt", type=str, default=None)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)
    generator = RayIntersectionGenerator(
        args.output_dir,
        canvas_width=args.canvas_width,
        aspect=args.aspect,
        seed=args.seed,
        prompt=args.prompt,
    )
    records = [generator.create_random_puzzle() for _ in range(max(1, args.count))]
    generator.write_metadata(records, Path(args.output_dir) / "puzzles.json")


if __name__ == "__main__":
    main()
