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
    r: float

    def bbox(self) -> Tuple[int, int, int, int]:
        return (
            int(round(self.cx - self.r)),
            int(round(self.cy - self.r)),
            int(round(self.cx + self.r)),
            int(round(self.cy + self.r)),
        )

    def to_dict(self) -> dict:
        return {"cx": self.cx, "cy": self.cy, "r": self.r}


@dataclass
class CandidateArc:
    circle: CircleSpec
    label: str  # 'A'..'E'

    def to_dict(self) -> dict:
        d = self.circle.to_dict()
        d.update({"label": self.label})
        return d


@dataclass
class ArcConnectPuzzleRecord:
    id: str
    prompt: str
    canvas_dimensions: Tuple[int, int]
    margin: int
    mask_rect: Tuple[int, int, int, int]
    left_arc: CircleSpec
    candidates: List[CandidateArc]
    correct_option: str
    image: str
    solution_image_path: str

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "prompt": self.prompt,
            "canvas_dimensions": list(self.canvas_dimensions),
            "margin": self.margin,
            "mask_rect": list(self.mask_rect),
            "left_arc": self.left_arc.to_dict(),
            "candidates": [c.to_dict() for c in self.candidates],
            "correct_option": self.correct_option,
            "image": self.image,
            "solution_image_path": self.solution_image_path,
            "type": "arc_connect",
        }


class ArcConnectGenerator(AbstractPuzzleGenerator[ArcConnectPuzzleRecord]):
    def __init__(
        self,
        output_dir: PathLike = "data/arc_connect",
        *,
        canvas_width: int = 480,
        aspect: Optional[float] = None,  # W/H; <1 => portrait
        mask_fraction: float = 0.18,  # band width fraction of W
        arc_span_deg: float = 20.0,   # degrees to extend from crossing
        prompt: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(output_dir)

        W = int(canvas_width)
        if aspect and aspect > 0:
            H = int(round(W / float(aspect)))
        else:
            H = W
        self.canvas_dimensions = (W, H)
        self.margin = max(16, int(round(min(W, H) * 0.05)))
        self._rng = random.Random(seed)

        self.mask_fraction = max(0.08, min(0.35, float(mask_fraction)))
        self.arc_span_deg = max(2.0, min(90.0, float(arc_span_deg)))

        if prompt is None:
            prompt = "One arc on the left continues across the masked band to one of the arcs on the right. Which labeled arc matches? Remove the masked band quickly while keeping the arcs still. Speak out the answer in phonetic alphabet. In portrait. Static Camera. No zoom."
        self.prompt = prompt

        out = Path(self.output_dir)
        self.puzzle_dir = out / "puzzles"
        self.solution_dir = out / "solutions"
        self.puzzle_dir.mkdir(parents=True, exist_ok=True)
        self.solution_dir.mkdir(parents=True, exist_ok=True)

    # --------------- public API ---------------
    def create_puzzle(self, *, puzzle_id: Optional[str] = None) -> ArcConnectPuzzleRecord:
        W, H = self.canvas_dimensions
        left = self.margin
        right = W - self.margin
        top = self.margin
        bottom = H - self.margin

        # mask band
        mask_w = int(round(W * self.mask_fraction))
        mask_cx = W // 2
        mask_left = max(left + 8, mask_cx - mask_w // 2)
        mask_right = min(right - 8, mask_cx + mask_w // 2)
        mask_rect = (mask_left, top, mask_right, bottom)

        # choose a base (true) circle that crosses mask_right (for label position)
        base = self._pick_true_circle(mask_left, mask_right, left, right, top, bottom)

        # decide how many go above and below (sum=4)
        n_up = self._rng.randint(0, 4)
        n_down = 4 - n_up

        # compute equal gap so n_up and n_down fit within the canvas
        max_up_space = max(0.0, (base.cy - (top + 1.5 * base.r)))
        max_down_space = max(0.0, ((bottom - 1.5 * base.r) - base.cy))
        # initial gap guess
        gap_guess = 0.12 * H
        gap_min = 24.0
        gap_bounds: List[float] = [gap_guess]
        if n_up > 0:
            gap_bounds.append(max_up_space / n_up)
        if n_down > 0:
            gap_bounds.append(max_down_space / n_down)
        gap = max(gap_min, min(gap_bounds))

        # build shifted circles
        circles: List[CircleSpec] = [base]
        for i in range(1, n_up + 1):
            cy_up = base.cy - i * gap
            circles.append(CircleSpec(base.cx, cy_up, base.r))
        for i in range(1, n_down + 1):
            cy_down = base.cy + i * gap
            circles.append(CircleSpec(base.cx, cy_down, base.r))

        # choose one branch (upper/lower) for the whole puzzle
        branch_upper = bool(self._rng.getrandbits(1))
        # sort by the chosen branch crossing y (top to bottom) and label A..E accordingly
        labeled: List[Tuple[CircleSpec, float]] = []
        for c in circles:
            ys = self._crossing_ys(c, mask_right)
            if ys:
                y_for_label = ys[0] if branch_upper else ys[1]
            else:
                y_for_label = c.cy
            labeled.append((c, y_for_label))
        labeled.sort(key=lambda t: t[1])

        letters = list("ABCDE")
        candidates: List[CandidateArc] = []
        correct_label = None
        for idx, (c, _y) in enumerate(labeled):
            lab = letters[idx]
            candidates.append(CandidateArc(circle=c, label=lab))
            if c is base:
                correct_label = lab
        if correct_label is None:
            # fallback via value equality
            for idx, (c, _y) in enumerate(labeled):
                if (c.cx, c.cy, c.r) == (base.cx, base.cy, base.r):
                    correct_label = letters[idx]
                    break
        assert correct_label is not None

        # render
        pid = puzzle_id or str(uuid.uuid4())
        puzzle_img = self._render(
            left_circle=base,
            candidates=candidates,
            mask_rect=mask_rect,
            mask_right=mask_right,
            branch_upper=branch_upper,
            show_solution=False,
            highlight_label=None,
        )
        solution_img = self._render(
            left_circle=base,
            candidates=candidates,
            mask_rect=mask_rect,
            mask_right=mask_right,
            branch_upper=branch_upper,
            show_solution=True,
            highlight_label=correct_label,
        )

        puzzle_path = self.puzzle_dir / f"{pid}_puzzle.png"
        solution_path = self.solution_dir / f"{pid}_solution.png"
        puzzle_img.save(puzzle_path)
        solution_img.save(solution_path)

        return ArcConnectPuzzleRecord(
            id=pid,
            prompt=self.prompt,
            canvas_dimensions=self.canvas_dimensions,
            margin=self.margin,
            mask_rect=mask_rect,
            left_arc=base,
            candidates=candidates,
            correct_option=correct_label or "A",
            image=self.relativize_path(puzzle_path),
            solution_image_path=self.relativize_path(solution_path),
        )

    def create_random_puzzle(self) -> ArcConnectPuzzleRecord:
        return self.create_puzzle()

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
        W, H = self.canvas_dimensions
        for _ in range(200):
            r = self._rng.uniform(0.38, 0.55) * min(W, H)
            cx = self._rng.uniform(mask_left - 0.2 * r, mask_right + 0.2 * r)
            cy = self._rng.uniform(top + 1.5 * r, bottom - 1.5 * r)
            # ensure mask_right intersects circle for label placement
            if abs(mask_right - cx)*1.2 < r and abs(mask_left - cx)*1.2 < r:
                return CircleSpec(cx, cy, r)
        # fallback centered
        r = 0.45 * min(W, H)
        cx = (mask_left + mask_right) / 2
        cy = (top + bottom) / 2
        return CircleSpec(cx, cy, r)

    @staticmethod
    def _crossing_ys(circle: CircleSpec, x: float) -> Optional[Tuple[float, float]]:
        dx = x - circle.cx
        val = circle.r * circle.r - dx * dx
        if val <= 1e-6:
            return None
        root = math.sqrt(val)
        y1 = circle.cy - root
        y2 = circle.cy + root
        return (y1, y2)

    def _render(
        self,
        *,
        left_circle: CircleSpec,
        candidates: Sequence[CandidateArc],
        mask_rect: Tuple[int, int, int, int],
        mask_right: int,
        branch_upper: bool,
        show_solution: bool,
        highlight_label: Optional[str],
    ) -> Image.Image:
        W, H = self.canvas_dimensions
        base = Image.new("RGB", (W, H), (255, 255, 255))
        base_draw = ImageDraw.Draw(base)

        arc_color = (40, 40, 40, 255)
        left_color = (10, 10, 10, 255)
        width = max(3, int(round(min(W, H) * 0.015)))

        # helper to compute crossing angles in radians at a vertical line
        def crossing_angles(circle: CircleSpec, x_line: float) -> List[float]:
            dx = x_line - circle.cx
            val = circle.r * circle.r - dx * dx
            if val <= 1e-6:
                return []
            root = math.sqrt(val)
            y1 = circle.cy - root
            y2 = circle.cy + root
            th1 = math.atan2(y1 - circle.cy, dx)
            th2 = math.atan2(y2 - circle.cy, dx)
            return [th1, th2]

        def deg(angle_rad: float) -> float:
            d = math.degrees(angle_rad) % 360.0
            if d < 0:
                d += 360.0
            return d

        span = math.radians(self.arc_span_deg)

        # 1) Right-side layer: draw arc segments away from mask, keep only x > mask_right
        right_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        rd = ImageDraw.Draw(right_layer)
        for cand in candidates:
            ths = crossing_angles(cand.circle, mask_right)
            if not ths:
                continue
            th = ths[0] if branch_upper else ths[1]
            # choose direction that moves into x > mask_right
            sign = 1.0 if math.sin(th) < 0 else -1.0
            t0 = th
            t1 = th + sign * span
            d0, d1 = deg(t0), deg(t1)
            if d1 < d0:
                d0, d1 = d1, d0
            rd.arc(cand.circle.bbox(), start=d0, end=d1, fill=arc_color, width=width)
        # erase everything to the left of the right mask edge
        rd.rectangle((0, 0, mask_right, H), fill=(0, 0, 0, 0))
        base.paste(right_layer, (0, 0), right_layer)

        # 2) Left-side layer: draw base arc segments away from mask, keep only x < mask_left
        left_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        ld = ImageDraw.Draw(left_layer)
        ths_left = crossing_angles(left_circle, mask_rect[0])
        if ths_left:
            th = ths_left[0] if branch_upper else ths_left[1]
            # choose direction that moves into x < mask_left
            sign = -1.0 if math.sin(th) < 0 else 1.0
            t0 = th
            t1 = th + sign * span
            d0, d1 = deg(t0), deg(t1)
            if d1 < d0:
                d0, d1 = d1, d0
            ld.arc(left_circle.bbox(), start=d0, end=d1, fill=left_color, width=width)
        # erase everything to the right of the left mask edge
        ld.rectangle((mask_rect[0], 0, W, H), fill=(0, 0, 0, 0))
        base.paste(left_layer, (0, 0), left_layer)

        # 3) Mask band edge lines to visualize the hidden middle
        edge_color = (200, 200, 200)
        base_draw.rectangle(mask_rect, fill=(240, 240, 240))
        base_draw.line((mask_rect[0], 0, mask_rect[0], H), fill=edge_color, width=5)
        base_draw.line((mask_rect[2], 0, mask_rect[2], H), fill=edge_color, width=5)

        # 4) Labels at mask_right upper crossing points for each candidate (A–E top to bottom)
        try:
            font = ImageFont.load_default(24)
        except Exception:
            font = None
        # compute consistent top-to-bottom order based on crossing y
        order: List[Tuple[CandidateArc, int]] = []
        for idx, cand in enumerate(candidates):
            ys = self._crossing_ys(cand.circle, mask_right)
            y_for_order = (ys[0] if ys else cand.circle.cy) if branch_upper else (ys[1] if ys else cand.circle.cy)
            order.append((cand, int(round(max(self.margin, min(H - self.margin, y_for_order))))))
        # candidates already labeled A–E in top-to-bottom order; draw accordingly
        order.sort(key=lambda t: t[1])
        for cand, _ in order:
            # compute the end point of the drawn right arc segment and place label there
            dx = mask_right - cand.circle.cx
            val = cand.circle.r * cand.circle.r - dx * dx
            if val <= 1e-6:
                ex = mask_right + 8
                ey = cand.circle.cy
            else:
                root = math.sqrt(val)
                y1 = cand.circle.cy - root
                y2 = cand.circle.cy + root
                th = math.atan2(y1 - cand.circle.cy, dx) if branch_upper else math.atan2(y2 - cand.circle.cy, dx)
                sign = 1.0 if math.sin(th) < 0 else -1.0
                t1 = th + sign * span
                ex = cand.circle.cx + cand.circle.r * math.cos(t1)
                ey = cand.circle.cy + cand.circle.r * math.sin(t1)
            lx = int(round(min(W - self.margin, max(mask_right + 4, ex + 6))))
            ly = int(round(max(self.margin, min(H - self.margin, ey - 6))))
            base_draw.text((lx, ly), cand.label, fill=(0, 0, 160), font=font)

        # 5) Solution highlight
        if show_solution and highlight_label:
            # find the label position again to draw a ring
            for cand, _ in order:
                if cand.label != highlight_label:
                    continue
                dx = mask_right - cand.circle.cx
                val = cand.circle.r * cand.circle.r - dx * dx
                if val <= 1e-6:
                    ex = mask_right + 8
                    ey = cand.circle.cy
                else:
                    root = math.sqrt(val)
                    y1 = cand.circle.cy - root
                    y2 = cand.circle.cy + root
                    th = math.atan2(y1 - cand.circle.cy, dx) if branch_upper else math.atan2(y2 - cand.circle.cy, dx)
                    sign = 1.0 if math.sin(th) < 0 else -1.0
                    t1 = th + sign * span
                    ex = cand.circle.cx + cand.circle.r * math.cos(t1)
                    ey = cand.circle.cy + cand.circle.r * math.sin(t1)
                cx, cy = int(round(ex + 12)), int(round(ey + 0))
                r = max(10, int(round(min(W, H) * 0.03)))
                base_draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline=(20, 180, 20), width=3)
                break

        return base


__all__ = [
    "ArcConnectGenerator",
    "ArcConnectPuzzleRecord",
    "CircleSpec",
    "CandidateArc",
]


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate masked arc-connection puzzles")
    parser.add_argument("count", type=int, help="Number of puzzles to create")
    parser.add_argument("--output-dir", type=Path, default=Path("data/arc_connect"))
    parser.add_argument("--canvas-width", type=int, default=480)
    parser.add_argument("--aspect", type=float, default=None, help="Canvas aspect ratio W/H (e.g., 3/4=0.75 portrait)")
    parser.add_argument("--mask-fraction", type=float, default=0.5)
    parser.add_argument("--arc-span-deg", type=float, default=20.0, help="Arc length in degrees from each mask crossing")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--prompt",type=str, default=None)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)
    gen = ArcConnectGenerator(
        args.output_dir,
        canvas_width=args.canvas_width,
        aspect=args.aspect,
        mask_fraction=args.mask_fraction,
        arc_span_deg=args.arc_span_deg,
        seed=args.seed,
        prompt=args.prompt,
    )
    records = [gen.create_random_puzzle() for _ in range(max(1, args.count))]
    gen.write_metadata(records, Path(args.output_dir) / "data.json")


if __name__ == "__main__":
    main()
