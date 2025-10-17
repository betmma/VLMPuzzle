"""Midpoint puzzle generator.

Two anchor points are placed symmetrically around a hidden midpoint. Solvers
must imagine or sketch the segment connecting them, identify the midpoint, and
select the correct labeled option nearby.
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
from ..point_target_base import PointCandidate, PointTargetPuzzleGenerator


@dataclass
class Segment:
    start: Tuple[float, float]
    end: Tuple[float, float]

    def to_dict(self) -> dict:
        return {
            "start": list(self.start),
            "end": list(self.end),
        }


@dataclass
class MidpointPuzzleRecord:
    id: str
    prompt: str
    canvas_dimensions: Tuple[int, int]
    margin: int
    midpoint: Tuple[float, float]
    segment: Segment
    candidates: List[PointCandidate]
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
            "midpoint": list(self.midpoint),
            "segment": self.segment.to_dict(),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "point_radius": self.point_radius,
            "correct_option": self.correct_option,
            "puzzle_image_path": self.puzzle_image_path,
            "solution_image_path": self.solution_image_path,
            "type": "midpoint",
        }


CandidatePoint = PointCandidate


class MidpointGenerator(PointTargetPuzzleGenerator[MidpointPuzzleRecord]):
    """Generate puzzles that hide the midpoint of a segment."""

    def __init__(
        self,
        output_dir: PathLike = "data/midpoint",
        *,
        canvas_width: int = 480,
        aspect: Optional[float] = None,
        seed: Optional[int] = None,
        prompt: Optional[str] = None,
        option_labels: Sequence[str] = ("A", "B", "C", "D", "E"),
    ) -> None:
        if prompt is None:
            prompt = (
                "Connect the two large circles and mark the midpoint as red. "
                "Speak out which option is the midpoint using phonetics alphabet"
            )
        super().__init__(
            output_dir,
            canvas_width=canvas_width,
            aspect=aspect,
            seed=seed,
            prompt=prompt,
            option_labels=option_labels,
        )
        out_root = Path(self.output_dir)
        self.puzzle_dir = out_root / "puzzles"
        self.solution_dir = out_root / "solutions"
        self.puzzle_dir.mkdir(parents=True, exist_ok=True)
        self.solution_dir.mkdir(parents=True, exist_ok=True)

    def create_puzzle(self, *, puzzle_id: Optional[str] = None) -> MidpointPuzzleRecord:
        midpoint = self.pick_target_point()
        segment = self._build_segment(midpoint)
        point_radius = self.point_radius
        candidates, correct_label = self.place_candidates(midpoint)

        pid = puzzle_id or str(uuid.uuid4())
        puzzle_img = self._render(
            midpoint=midpoint,
            segment=segment,
            candidates=candidates,
            highlight_label=None,
        )
        solution_img = self._render(
            midpoint=midpoint,
            segment=segment,
            candidates=candidates,
            highlight_label=correct_label,
        )

        puzzle_path = self.puzzle_dir / f"{pid}_puzzle.png"
        solution_path = self.solution_dir / f"{pid}_solution.png"
        puzzle_img.save(puzzle_path)
        solution_img.save(solution_path)

        return MidpointPuzzleRecord(
            id=pid,
            prompt=self.prompt,
            canvas_dimensions=self.canvas_dimensions,
            margin=self.margin,
            midpoint=midpoint,
            segment=segment,
            candidates=candidates,
            point_radius=point_radius,
            correct_option=correct_label,
            puzzle_image_path=self.relativize_path(puzzle_path),
            solution_image_path=self.relativize_path(solution_path),
        )

    def create_random_puzzle(self) -> MidpointPuzzleRecord:
        return self.create_puzzle()

    def _build_segment(
        self,
        midpoint: Tuple[float, float],
    ) -> Segment:
        left, top, right, bottom = self.canvas_bounds()
        mx, my = midpoint
        attempts = 0
        while attempts < 200:
            attempts += 1
            angle = self.rng.uniform(0.0, math.tau)
            dx = math.cos(angle)
            dy = math.sin(angle)
            max_extent = float("inf")
            if abs(dx) > 1e-6:
                bound_x_pos = (right - mx) / dx if dx > 0 else (left - mx) / dx
                max_extent = min(max_extent, abs(bound_x_pos))
            if abs(dy) > 1e-6:
                bound_y_pos = (bottom - my) / dy if dy > 0 else (top - my) / dy
                max_extent = min(max_extent, abs(bound_y_pos))
            max_extent = float(max(max_extent, 0.0))
            max_extent *= 0.9
            min_extent = max(40.0, 0.12 * min(right - left, bottom - top))
            if max_extent < min_extent:
                continue
            half_length = self.rng.uniform(min_extent, max_extent)
            start = (mx - dx * half_length, my - dy * half_length)
            end = (mx + dx * half_length, my + dy * half_length)
            if self._inside_bounds(start) and self._inside_bounds(end):
                return Segment(start=start, end=end)
        # Fallback: horizontal segment
        half_length = min(0.3 * (right - left), 0.3 * (bottom - top))
        start = (max(left + 10, mx - half_length), my)
        end = (min(right - 10, mx + half_length), my)
        return Segment(start=start, end=end)

    def _inside_bounds(self, point: Tuple[float, float]) -> bool:
        left, top, right, bottom = self.canvas_bounds()
        x, y = point
        return left <= x <= right and top <= y <= bottom

    def _render(
        self,
        *,
        midpoint: Tuple[float, float],
        segment: Segment,
        candidates: Sequence[PointCandidate],
        highlight_label: Optional[str],
    ) -> Image.Image:
        width, height = self.canvas_dimensions
        base = Image.new("RGB", (width, height), (255, 255, 255))
        draw = ImageDraw.Draw(base)

        anchor_color = (30, 30, 30)
        point_radius = self.point_radius
        anchor_radius = max(point_radius + 6, int(round(min(width, height) * 0.028)))
        draw.ellipse(
            [
                int(round(segment.start[0] - anchor_radius)),
                int(round(segment.start[1] - anchor_radius)),
                int(round(segment.start[0] + anchor_radius)),
                int(round(segment.start[1] + anchor_radius)),
            ],
            fill=(250, 250, 250),
            outline=anchor_color,
            width=max(3, anchor_radius // 3),
        )
        draw.ellipse(
            [
                int(round(segment.end[0] - anchor_radius)),
                int(round(segment.end[1] - anchor_radius)),
                int(round(segment.end[0] + anchor_radius)),
                int(round(segment.end[1] + anchor_radius)),
            ],
            fill=(250, 250, 250),
            outline=anchor_color,
            width=max(3, anchor_radius // 3),
        )
        
        if highlight_label is not None: # Draw segment only on solution image
            draw.line(
                [
                    (int(round(segment.start[0])), int(round(segment.start[1]))),
                    (int(round(segment.end[0])), int(round(segment.end[1]))),
                ],
                fill=(180, 180, 180),
                width=max(2, int(round(min(width, height) * 0.01))),
            )

        self.draw_candidates(
            draw,
            candidates=candidates,
            highlight_label=highlight_label,
        )

        return base


__all__ = [
    "MidpointGenerator",
    "MidpointPuzzleRecord",
    "Segment",
    "CandidatePoint",
]


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate midpoint puzzles")
    parser.add_argument("count", type=int, help="Number of puzzles to create")
    parser.add_argument("--output-dir", type=Path, default=Path("data/midpoint"))
    parser.add_argument("--canvas-width", type=int, default=480)
    parser.add_argument("--aspect", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--prompt", type=str, default=None)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)
    generator = MidpointGenerator(
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
