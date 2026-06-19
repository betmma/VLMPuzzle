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
    image: str
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
            "image": self.image,
            "solution_image_path": self.solution_image_path,
            "type": "midpoint",
        }


CandidatePoint = PointCandidate


class MidpointGenerator(PointTargetPuzzleGenerator):
    """Generate puzzles that hide the midpoint of a segment."""
    DEFAULT_OUTPUT_DIR="data/midpoint"
    DEFAULT_PROMPT="Connect the two large circles and mark the midpoint as red. Speak out which option is the midpoint using phonetics alphabet. In portrait, no zoom, no pan."
    DEFAULT_GPT5_PROMPT="Which option is the midpoint of the two circles? Answer an option in A-E."

    def create_puzzle(self, *, puzzle_id: Optional[str] = None) -> MidpointPuzzleRecord:
        midpoint = self.pick_target_point()
        segment = self._build_segment(midpoint.to_list())
        point_radius = self.point_radius
        self.place_candidates(midpoint)
        
        # Store for _render usage (implicitly required by save_video_solution)
        self._midpoint_coords = midpoint.to_list()
        self._segment = segment

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
                import traceback
                traceback.print_exc()
                print(f"Video error: {e}")


        return MidpointPuzzleRecord(
            id=pid,
            prompt=self.prompt,
            canvas_dimensions=self.canvas_dimensions,
            margin=self.margin,
            midpoint=midpoint.to_list(),
            segment=segment,
            candidates=self.candidates,
            point_radius=point_radius,
            correct_option=self.correct_label,
            image=self.relativize_path(puzzle_path),
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
        highlight_label: Optional[str],
        # Legacy/Testing optional args
        midpoint: Optional[Tuple[float, float]] = None,
        segment: Optional[Segment] = None,
    ) -> Image.Image:
        width, height = self.canvas_dimensions
        draw, base = self.get_draw_base()
        
        # Resolve args or state
        midpoint_val = midpoint if midpoint is not None else getattr(self, "_midpoint_coords", None)
        segment_val = segment if segment is not None else getattr(self, "_segment", None)
        
        if midpoint_val is None or segment_val is None:
             raise RuntimeError("Render called without geometry state")

        anchor_color = (30, 30, 30)
        point_radius = self.point_radius
        anchor_radius = max(point_radius + 6, int(round(min(width, height) * 0.028)))
        draw.ellipse(
            [
                int(round(segment_val.start[0] - anchor_radius)),
                int(round(segment_val.start[1] - anchor_radius)),
                int(round(segment_val.start[0] + anchor_radius)),
                int(round(segment_val.start[1] + anchor_radius)),
            ],
            fill=(250, 250, 250),
            outline=anchor_color,
            width=max(3, anchor_radius // 3),
        )
        draw.ellipse(
            [
                int(round(segment_val.end[0] - anchor_radius)),
                int(round(segment_val.end[1] - anchor_radius)),
                int(round(segment_val.end[0] + anchor_radius)),
                int(round(segment_val.end[1] + anchor_radius)),
            ],
            fill=(250, 250, 250),
            outline=anchor_color,
            width=max(3, anchor_radius // 3),
        )
        
        if highlight_label is not None: # Draw segment only on solution image
            draw.line(
                [
                    (int(round(segment_val.start[0])), int(round(segment_val.start[1]))),
                    (int(round(segment_val.end[0])), int(round(segment_val.end[1]))),
                ],
                fill=(180, 180, 180),
                width=max(2, int(round(min(width, height) * 0.01))),
            )

        self.draw_candidates(
            draw,
            highlight_label=highlight_label,
        )

        return base


__all__ = [
    "MidpointGenerator",
    "MidpointPuzzleRecord",
    "Segment",
    "CandidatePoint",
]


def main(argv: Optional[List[str]] = None) -> None:
    MidpointGenerator.main(MidpointGenerator, argv)


if __name__ == "__main__":
    main()
