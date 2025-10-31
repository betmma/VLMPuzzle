"""Arc connection puzzle generator (masked vertical band, side arcs).

Visual design per request:
- Define mask_left_x and mask_right_x.
- Draw the true circle as arcs only on x < mask_left_x and x > mask_right_x.
- Create four false circles by shifting the true circle up and down with equal gaps.
- For false circles, draw arcs only on x > mask_right_x (right side).
- Label A–E at the mask-right end of each right arc; exactly one matches the
  true circle geometry. Prompt includes NATO phonetic instruction and requests
  portrait.
"""

from __future__ import annotations

import argparse
import math
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageDraw

from ..base import PathLike
from ..point_target_base import (
    PointCandidate,
    PointTargetPuzzleGenerator,
    PointTargetPuzzleRecord,
)


@dataclass
class CircleSpec:
    cx: float
    cy: float
    r: float

    def bbox(self) -> Tuple[int, int, int, int]:
        return (
            int(round(self.cx - self.r)),
            int(round(self.cy - self.r)),
            int(round(self.cx + self.r)),
            int(round(self.cy + self.r)),
        )

    def to_dict(self) -> Dict[str, float]:
        return {"cx": self.cx, "cy": self.cy, "r": self.r}


@dataclass
class CandidateArc:
    circle: CircleSpec
    label: str

    def to_dict(self) -> Dict[str, object]:
        payload: Dict[str, object] = dict(self.circle.to_dict())
        payload["label"] = self.label
        return payload


@dataclass
class ArcConnectPointPuzzleRecord(PointTargetPuzzleRecord):
    mask_rect: Tuple[int, int, int, int]
    left_arc: CircleSpec
    candidate_arcs: List[CandidateArc]
    branch_upper: bool
    mask_right: int
    arc_span_deg: float

    def to_dict(self) -> Dict[str, object]:
        return {
            "id": self.id,
            "prompt": self.prompt,
            "canvas_dimensions": list(self.canvas_dimensions),
            "margin": self.margin,
            "candidates": [c.to_dict() for c in self.candidates],
            "correct_option": self.correct_option,
            "image": self.image,
            "solution_image_path": self.solution_image_path,
            "mask_rect": list(self.mask_rect),
            "left_arc": self.left_arc.to_dict(),
            "candidate_arcs": [c.to_dict() for c in self.candidate_arcs],
            "branch_upper": self.branch_upper,
            "mask_right": self.mask_right,
            "arc_span_deg": self.arc_span_deg,
            "type": "arc_connect_point_ver",
        }


class ArcConnectGenerator(PointTargetPuzzleGenerator):
    DEFAULT_OUTPUT_DIR = "data/arc_connect_point_ver"
    DEFAULT_PROMPT = "One arc on the left continues across the masked band to one of the arcs on the right. Which labeled arc matches? Remove the masked band quickly while keeping the arcs still, then paint the correct option label (not arc) red. Speak out the answer in phonetic alphabet. In portrait. Static Camera. No zoom."
    DEFAULT_GPT5_PROMPT = "One arc on the left continues across the masked band to one of the arcs on the right. Which labeled arc matches? Answer with A-E."

    def __init__(
        self,
        output_dir: PathLike = None,
        *,
        canvas_width: int = 480,
        aspect: Optional[float] = None,
        mask_fraction: float = 0.18,
        arc_span_deg: float = 20.0,
        prompt: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(
            output_dir=output_dir,
            canvas_width=canvas_width,
            aspect=aspect,
            seed=seed,
            prompt=prompt,
        )

        self.mask_fraction = max(0.08, min(0.35, float(mask_fraction)))
        self.arc_span_deg = max(2.0, min(90.0, float(arc_span_deg)))
        self._span_rad = math.radians(self.arc_span_deg)
        self._forced_id: Optional[str] = None
        self._mask_rect: Optional[Tuple[int, int, int, int]] = None
        self._mask_right: int = 0
        self._left_circle: Optional[CircleSpec] = None
        self._candidate_arcs: List[CandidateArc] = []
        self._branch_upper: bool = True

    def create_puzzle(self, *, puzzle_id: Optional[str] = None) -> ArcConnectPointPuzzleRecord:
        if puzzle_id:
            self._forced_id = puzzle_id

        width, height = self.canvas_dimensions
        left, top, right, bottom = self.canvas_bounds()

        mask_w = int(round(width * self.mask_fraction))
        mask_cx = width // 2
        mask_left = max(left + 8, mask_cx - mask_w // 2)
        mask_right = min(right - 8, mask_cx + mask_w // 2)
        mask_rect = (mask_left, top, mask_right, bottom)

        base_circle = self._pick_true_circle(mask_left, mask_right, left, right, top, bottom)

        n_up = self.rng.randint(0, 4)
        n_down = 4 - n_up

        max_up_space = max(0.0, (base_circle.cy - (top + 1.5 * base_circle.r)))
        max_down_space = max(0.0, ((bottom - 1.5 * base_circle.r) - base_circle.cy))
        gap_guess = 0.12 * height
        gap_min = 24.0
        gap_bounds: List[float] = [gap_guess]
        if n_up > 0:
            gap_bounds.append(max_up_space / n_up)
        if n_down > 0:
            gap_bounds.append(max_down_space / n_down)
        gap = max(gap_min, min(gap_bounds))

        circles: List[CircleSpec] = [base_circle]
        for i in range(1, n_up + 1):
            circles.append(CircleSpec(base_circle.cx, base_circle.cy - i * gap, base_circle.r))
        for i in range(1, n_down + 1):
            circles.append(CircleSpec(base_circle.cx, base_circle.cy + i * gap, base_circle.r))

        branch_upper = bool(self.rng.getrandbits(1))
        labeled: List[Tuple[CircleSpec, float]] = []
        for circle in circles:
            ys = self._crossing_ys(circle, mask_right)
            y_for_label = ys[0] if ys and branch_upper else ys[1] if ys else circle.cy
            labeled.append((circle, y_for_label))
        labeled.sort(key=lambda entry: entry[1])

        labels = list("ABCDE")
        candidate_arcs: List[CandidateArc] = []
        correct_label: Optional[str] = None
        for idx, (circle, _y) in enumerate(labeled):
            label = labels[idx]
            candidate_arcs.append(CandidateArc(circle=circle, label=label))
            if circle is base_circle:
                correct_label = label
        if correct_label is None:
            for idx, (circle, _y) in enumerate(labeled):
                if (circle.cx, circle.cy, circle.r) == (base_circle.cx, base_circle.cy, base_circle.r):
                    correct_label = labels[idx]
                    break
        if correct_label is None:
            raise RuntimeError("Unable to determine correct candidate arc")

        self._span_rad = math.radians(self.arc_span_deg)
        self._mask_rect = mask_rect
        self._mask_right = mask_right
        self._left_circle = base_circle
        self._candidate_arcs = candidate_arcs
        self._branch_upper = branch_upper

        point_candidates: List[PointCandidate] = []
        for candidate in candidate_arcs:
            label_x, label_y = self._label_center(candidate.circle, mask_right, branch_upper)
            point_candidates.append(PointCandidate(x=label_x, y=label_y, label=candidate.label))

        self.candidates = point_candidates
        self.correct_label = correct_label

        record = self.save_puzzle()
        return record

    def save_puzzle(self) -> ArcConnectPointPuzzleRecord:
        if self._mask_rect is None or self._left_circle is None:
            raise RuntimeError("Puzzle geometry not initialized before saving")

        pid = self._forced_id or str(uuid.uuid4())
        self._forced_id = None

        puzzle_img = self._render(highlight_label=None)
        solution_img = self._render(highlight_label=self.correct_label)

        puzzle_path = self.puzzle_dir / f"{pid}_puzzle.png"
        solution_path = self.solution_dir / f"{pid}_solution.png"
        puzzle_img.save(puzzle_path)
        solution_img.save(solution_path)

        record = ArcConnectPointPuzzleRecord(
            id=pid,
            prompt=self.prompt,
            canvas_dimensions=self.canvas_dimensions,
            margin=self.margin,
            candidates=self.candidates,
            correct_option=self.correct_label,
            image=self.relativize_path(puzzle_path),
            solution_image_path=self.relativize_path(solution_path),
            mask_rect=self._mask_rect,
            left_arc=self._left_circle,
            candidate_arcs=list(self._candidate_arcs),
            branch_upper=self._branch_upper,
            mask_right=self._mask_right,
            arc_span_deg=self.arc_span_deg,
        )

        self.pid = pid
        self.puzzle_path = puzzle_path
        self.solution_path = solution_path
        return record

    # --------------- internals ---------------
    def _pick_true_circle(
        self,
        mask_left: int,
        mask_right: int,
        left: int,
        right: int,
        top: int,
        bottom: int,
    ) -> CircleSpec:
        width, height = self.canvas_dimensions
        for _ in range(200):
            radius = self.rng.uniform(0.38, 0.55) * min(width, height)
            cx = self.rng.uniform(mask_left - 0.2 * radius, mask_right + 0.2 * radius)
            cy = self.rng.uniform(top + 1.5 * radius, bottom - 1.5 * radius)
            if abs(mask_right - cx) * 1.2 < radius and abs(mask_left - cx) * 1.2 < radius:
                return CircleSpec(cx, cy, radius)
        fallback_radius = 0.45 * min(width, height)
        fallback_cx = (mask_left + mask_right) / 2
        fallback_cy = (top + bottom) / 2
        return CircleSpec(fallback_cx, fallback_cy, fallback_radius)

    @staticmethod
    def _crossing_ys(circle: CircleSpec, x: float) -> Optional[Tuple[float, float]]:
        dx = x - circle.cx
        value = circle.r * circle.r - dx * dx
        if value <= 1e-6:
            return None
        root = math.sqrt(value)
        return (circle.cy - root, circle.cy + root)

    @staticmethod
    def _crossing_angles(circle: CircleSpec, x_line: float) -> List[float]:
        dx = x_line - circle.cx
        value = circle.r * circle.r - dx * dx
        if value <= 1e-6:
            return []
        root = math.sqrt(value)
        y1 = circle.cy - root
        y2 = circle.cy + root
        angle_one = math.atan2(y1 - circle.cy, dx)
        angle_two = math.atan2(y2 - circle.cy, dx)
        return [angle_one, angle_two]

    @staticmethod
    def _deg(angle_rad: float) -> float:
        value = math.degrees(angle_rad) % 360.0
        if value < 0:
            value += 360.0
        return value

    def _arc_end_point(self, circle: CircleSpec, mask_right: int, branch_upper: bool) -> Tuple[float, float]:
        dx = mask_right - circle.cx
        value = circle.r * circle.r - dx * dx
        if value <= 1e-6:
            return mask_right + self.point_radius * 1.2, circle.cy
        root = math.sqrt(value)
        crossing_y = circle.cy - root if branch_upper else circle.cy + root
        theta = math.atan2(crossing_y - circle.cy, dx)
        direction = 1.0 if math.sin(theta) < 0 else -1.0
        end_angle = theta + direction * self._span_rad
        end_x = circle.cx + circle.r * math.cos(end_angle)
        end_y = circle.cy + circle.r * math.sin(end_angle)
        return end_x, end_y

    def _label_center(self, circle: CircleSpec, mask_right: int, branch_upper: bool) -> Tuple[float, float]:
        end_x, end_y = self._arc_end_point(circle, mask_right, branch_upper)
        offset = max(float(self.point_radius) * 2.2, 12.0)
        min_x = mask_right + self.point_radius + 6
        max_x = self.canvas_dimensions[0] - self.margin - self.point_radius
        centered_x = max(min_x, min(max_x, end_x + offset))
        min_y = self.margin + self.point_radius
        max_y = self.canvas_dimensions[1] - self.margin - self.point_radius
        centered_y = max(min_y, min(max_y, end_y))
        return centered_x, centered_y

    def _candidate_by_label(self, label: str) -> Optional[PointCandidate]:
        target = label.upper()
        for candidate in self.candidates:
            if candidate.label.upper() == target:
                return candidate
        return None

    def _render(self, highlight_label: Optional[str]) -> Image.Image:
        if self._mask_rect is None or self._left_circle is None:
            raise RuntimeError("Render requested before puzzle geometry prepared")

        width, height = self.canvas_dimensions
        base = Image.new("RGB", (width, height), (255, 255, 255))
        draw = ImageDraw.Draw(base)

        mask_rect = self._mask_rect
        mask_right = self._mask_right
        branch_upper = self._branch_upper
        span = self._span_rad

        arc_color = (40, 40, 40, 255)
        left_color = (10, 10, 10, 255)
        stroke = max(3, int(round(min(width, height) * 0.015)))

        right_layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        right_draw = ImageDraw.Draw(right_layer)
        for candidate in self._candidate_arcs:
            angles = self._crossing_angles(candidate.circle, mask_right)
            if not angles:
                continue
            theta = angles[0] if branch_upper else angles[1]
            direction = 1.0 if math.sin(theta) < 0 else -1.0
            start = theta
            if highlight_label:
                start=start- direction * (math.pi/2-span)
            end = theta + direction * span
            deg_start = self._deg(start)
            deg_end = self._deg(end)
            if deg_end < deg_start:
                deg_start, deg_end = deg_end, deg_start
            right_draw.arc(candidate.circle.bbox(), start=deg_start, end=deg_end, fill=arc_color, width=stroke)
        right_draw.rectangle((0, 0, width//2-1 if highlight_label else mask_right, height), fill=(0, 0, 0, 0))
        base.paste(right_layer, (0, 0), right_layer)

        left_layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        left_draw = ImageDraw.Draw(left_layer)
        left_angles = self._crossing_angles(self._left_circle, mask_rect[0])
        if left_angles:
            theta = left_angles[0] if branch_upper else left_angles[1]
            direction = -1.0 if math.sin(theta) < 0 else 1.0
            start = theta
            if highlight_label:
                start=start - direction * (math.pi/2-span)
            end = theta + direction * span
            deg_start = self._deg(start)
            deg_end = self._deg(end)
            if deg_end < deg_start:
                deg_start, deg_end = deg_end, deg_start
            left_draw.arc(self._left_circle.bbox(), start=deg_start, end=deg_end, fill=left_color, width=stroke)
        left_draw.rectangle(( width//2 if highlight_label else mask_rect[0], 0, width, height), fill=(0, 0, 0, 0))
        base.paste(left_layer, (0, 0), left_layer)

        if highlight_label is None:
            edge_color = (200, 200, 200)
            draw.rectangle(mask_rect, fill=(240, 240, 240))
            draw.line((mask_rect[0], 0, mask_rect[0], height), fill=edge_color, width=5)
            draw.line((mask_rect[2], 0, mask_rect[2], height), fill=edge_color, width=5)

        self.draw_candidates(draw, highlight_label=highlight_label)

        return base


__all__ = [
    "ArcConnectGenerator",
    "ArcConnectPointPuzzleRecord",
    "CircleSpec",
    "CandidateArc",
]


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate masked arc-connection puzzles with point targets")
    parser.add_argument("count", type=int, help="Number of puzzles to create")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--canvas-width", type=int, default=480)
    parser.add_argument("--aspect", type=float, default=None, help="Canvas aspect ratio W/H (e.g., 3/4=0.75 portrait)")
    parser.add_argument("--mask-fraction", type=float, default=0.5)
    parser.add_argument("--arc-span-deg", type=float, default=20.0, help="Arc length in degrees from each mask crossing")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--use-gpt-5", action="store_true", help="Use the GPT-5 oriented prompt template")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)
    prompt_value = args.prompt
    if args.use_gpt_5 and prompt_value is None:
        prompt_value = ArcConnectGenerator.DEFAULT_GPT5_PROMPT

    generator = ArcConnectGenerator(
        output_dir=args.output_dir,
        canvas_width=args.canvas_width,
        aspect=args.aspect,
        mask_fraction=args.mask_fraction,
        arc_span_deg=args.arc_span_deg,
        seed=args.seed,
        prompt=prompt_value,
    )
    records = [generator.create_random_puzzle() for _ in range(max(1, args.count))]
    generator.write_metadata(records, generator.output_dir / "data.json")


if __name__ == "__main__":
    main()
