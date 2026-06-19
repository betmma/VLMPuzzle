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
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont

from ..base import PathLike
from ..point_target_base import PointTargetPuzzleGenerator, PointCandidate


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
class RayIntersectionPuzzleRecord:
    id: str
    prompt: str
    canvas_dimensions: Tuple[int, int]
    margin: int
    intersection: Tuple[float, float]
    rays: List[RaySegment]
    candidates: List[PointCandidate]
    point_radius: int
    correct_option: str
    image: str
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
            "image": self.image,
            "solution_image_path": self.solution_image_path,
            "type": "ray_intersection",
        }


CandidatePoint = PointCandidate


class RayIntersectionGenerator(PointTargetPuzzleGenerator):
    """Generate puzzles with partially hidden ray intersections."""
    DEFAULT_OUTPUT_DIR="data/ray_intersection"
    DEFAULT_PROMPT="Extend the three black lines and mark the intersection point as red. Speak out which option is the intersection point using phonetics alphabet. In portrait, no zoom, no pan."
    DEFAULT_GPT5_PROMPT="Which option is the intersection point of the three lines? Answer an option in A-E."

    def create_puzzle(self, *, puzzle_id: Optional[str] = None) -> RayIntersectionPuzzleRecord:
        intersection = self.pick_target_point()
        rays = self._build_rays(intersection.to_list())
        point_radius = self.point_radius
        self.place_candidates(intersection)
        intersection=intersection.to_list()
        
        # Store for _render usage
        self._intersection_coords = intersection
        self._rays = rays

        pid = puzzle_id or str(uuid.uuid4())
        puzzle_img = self._render(
            highlight_label=None,
        )
        solution_img = self._render(
            highlight_label=self.correct_label,
        )

        puzzle_path = self.puzzle_dir / f"{pid}_puzzle.png"
        solution_path = self.solution_dir / f"{pid}_solution.png"
        puzzle_img.save(puzzle_path)
        solution_img.save(solution_path)

        if self.record_video:
            try:
                self.save_video_solution(pid)
            except Exception as e:
                pass

        return RayIntersectionPuzzleRecord(
            id=pid,
            prompt=self.prompt,
            canvas_dimensions=self.canvas_dimensions,
            margin=self.margin,
            intersection=intersection,
            rays=rays,
            candidates=self.candidates,
            point_radius=point_radius,
            correct_option=self.correct_label,
            image=self.relativize_path(puzzle_path),
            solution_image_path=self.relativize_path(solution_path),
        )

    def create_random_puzzle(self) -> RayIntersectionPuzzleRecord:
        return self.create_puzzle()

    def _build_rays(
        self,
        intersection: Tuple[float, float],
    ) -> List[RaySegment]:
        left, top, right, bottom = self.canvas_bounds()
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
            end_point = self._ray_to_bounds(intersection, angle)
            start_point = self._edge_segment_start(intersection, end_point)
            segments.append(RaySegment(angle=angle, start=start_point, end=end_point))
        return segments

    def _ray_to_bounds(
        self,
        origin: Tuple[float, float],
        angle: float,
    ) -> Tuple[float, float]:
        left, top, right, bottom = self.canvas_bounds()
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

    def _render(
        self,
        *,
        highlight_label: Optional[str],
        # Legacy/Testing optional args
        intersection: Optional[Tuple[float, float]] = None,
        rays: Optional[Sequence[RaySegment]] = None,
    ) -> Image.Image:
        width, height = self.canvas_dimensions
        draw, base = self.get_draw_base()

        # Resolve args or state
        intersection_val = intersection if intersection is not None else getattr(self, "_intersection_coords", None)
        rays_val = rays if rays is not None else getattr(self, "_rays", None)
        
        if intersection_val is None or rays_val is None:
             raise RuntimeError("Render called without geometry state")

        stroke_color = (40, 40, 40)
        stroke_width = max(3, int(round(min(width, height) * 0.015)))

        if highlight_label:
            for ray in rays_val:
                draw.line(
                    [
                        (int(round(ray.start[0])), int(round(ray.start[1]))),
                        (int(round(intersection_val[0])), int(round(intersection_val[1]))),
                    ],
                    fill=stroke_color,
                    width=stroke_width,
                )
        
        for ray in rays_val:
            draw.line(
                [
                    (int(round(ray.start[0])), int(round(ray.start[1]))),
                    (int(round(ray.end[0])), int(round(ray.end[1]))),
                ],
                fill=stroke_color,
                width=stroke_width,
            )

        self.draw_candidates(
            draw,
            highlight_label=highlight_label,
        )

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
    parser.add_argument("--video", action="store_true", help="Generate video solution")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)
    generator = RayIntersectionGenerator(
        args.output_dir,
        canvas_width=args.canvas_width,
        aspect=args.aspect,
        seed=args.seed,
        prompt=args.prompt,
        record_video=args.video,
    )
    records = [generator.create_random_puzzle() for _ in range(max(1, args.count))]
    generator.write_metadata(records, Path(args.output_dir) / "data.json")


if __name__ == "__main__":
    main()
