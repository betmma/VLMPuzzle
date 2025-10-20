"""Generator for colored-rectangles stacking order puzzles."""

from __future__ import annotations

import argparse
import math
import random
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw

from ..base import AbstractPuzzleGenerator, PathLike


Color = Tuple[int, int, int]
PaletteEntry = Tuple[str, Color]


BASE_COLOR_PALETTE: Tuple[PaletteEntry, ...] = (
    ("red", (229, 68, 68)),
    ("orange", (229, 149, 68)),
    ("yellow", (229, 229, 68)),
    ("green", (149, 229, 68)),
    ("teal", (68, 229, 195)),
    ("blue", (68, 149, 229)),
    ("purple", (149, 68, 229)),
    ("magenta", (229, 68, 195)),
)


@dataclass
class RectSpec:
    x: int
    y: int
    w: int
    h: int
    color: Color
    name: str
    z: int  # 0 is bottom, higher is closer to viewer

    def to_dict(self) -> dict:
        return {
            "x": self.x,
            "y": self.y,
            "w": self.w,
            "h": self.h,
            "color": list(self.color),
            "color_name": self.name,
            "z": self.z,
        }


@dataclass
class RectsPuzzleRecord:
    id: str
    prompt: str
    canvas_dimensions: Tuple[int, int]
    rectangles: List[RectSpec]
    color_palette: List[PaletteEntry]
    image: str
    solution_image_path: str

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "prompt": self.prompt,
            "canvas_dimensions": list(self.canvas_dimensions),
            "rectangles": [r.to_dict() for r in self.rectangles],
            "color_palette": [
                {"name": name, "color": list(color)} for name, color in self.color_palette
            ],
            "image": self.image,
            "solution_image_path": self.solution_image_path,
        }


class RectsGenerator(AbstractPuzzleGenerator[RectsPuzzleRecord]):
    """Generate overlapping colored rectangles with a definable z-order.

    The puzzle (input) image shows overlapping rectangles. The solution image
    shows an "exploded view" where rectangles are ordered from top to bottom
    according to z (highest/covering nearer the top of the image).
    """

    def __init__(
        self,
        output_dir: PathLike = "data/rects",
        *,
        rect_count: int = 4,
        canvas_size: int = 384,
        canvas_aspect_ratio: Optional[float] = None,
        require_strong_order: bool = True,
        prompt: str = None,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(output_dir)
        self.rect_count = max(2, min(rect_count, 8))
        self.canvas_size = int(canvas_size)
        self.canvas_aspect_ratio = float(canvas_aspect_ratio) if canvas_aspect_ratio else None
        self.require_strong_order = bool(require_strong_order)
        self._active_palette: List[PaletteEntry] = list(BASE_COLOR_PALETTE[: self.rect_count])
        color_names_clause = ", ".join(name for name, _ in self._active_palette)
        if prompt is None:
            prompt = (
                "Explode the colored rectangles vertically. The highest layer should be at the top of the image, "
                "and the lowest layer at the bottom. Speak out the colors of rectangles in layer order, from top "
                "to bottom, using these names: "
                f"{color_names_clause}. In portrait."
            )
        elif "{color_names}" in prompt:
            prompt = prompt.format(color_names=color_names_clause)
        else:
            base_prompt = prompt.strip()
            if base_prompt and base_prompt[-1] not in ".!?":
                base_prompt = f"{base_prompt}."
            prompt = f"{base_prompt} Colors in this puzzle: {color_names_clause}."
        self.prompt = prompt
        self._rng = random.Random(seed)

        # Compute canvas dimensions from square base size and optional aspect ratio
        base_w = base_h = self.canvas_size
        if self.canvas_aspect_ratio and self.canvas_aspect_ratio > 0:
            ratio = self.canvas_aspect_ratio
            if ratio >= 1.0:
                w = int(round(base_h * ratio))
                h = base_h
            else:
                w = base_w
                h = int(round(base_w / ratio))
            self.canvas_dimensions = (w, h)
        else:
            self.canvas_dimensions = (base_w, base_h)

        self.puzzle_dir = Path(self.output_dir) / "puzzles"
        self.solution_dir = Path(self.output_dir) / "solutions"
        self.puzzle_dir.mkdir(parents=True, exist_ok=True)
        self.solution_dir.mkdir(parents=True, exist_ok=True)

    def create_puzzle(self, *, puzzle_id: Optional[str] = None) -> RectsPuzzleRecord:
        puzzle_uuid = puzzle_id or str(uuid.uuid4())
        rects = self._generate_rects()

        puzzle_img = self._render_puzzle(rects)
        solution_img = self._render_solution(rects)

        puzzle_path = self.puzzle_dir / f"{puzzle_uuid}_puzzle.png"
        solution_path = self.solution_dir / f"{puzzle_uuid}_solution.png"
        puzzle_img.save(puzzle_path)
        solution_img.save(solution_path)

        return RectsPuzzleRecord(
            id=puzzle_uuid,
            prompt=self.prompt,
            canvas_dimensions=self.canvas_dimensions,
            rectangles=rects,
            color_palette=list(self._active_palette),
            image=self.relativize_path(puzzle_path),
            solution_image_path=self.relativize_path(solution_path),
        )

    def create_random_puzzle(self) -> RectsPuzzleRecord:
        return self.create_puzzle()

    # --- Generation internals -------------------------------------------------------

    def _generate_colors(self, n: int) -> List[PaletteEntry]:
        colors = list(self._active_palette[:n])
        self._rng.shuffle(colors)
        return colors

    def _generate_rects(self) -> List[RectSpec]:
        W, full_H = self.canvas_dimensions
        H = max(24, full_H // 3)
        colors = self._generate_colors(self.rect_count)

        cx = W // 2
        cy = H // 2
        anchor_span = max(16, int(min(W, H) * 0.2))
        anchor_w = anchor_span
        anchor_h = anchor_span
        ax0 = cx - anchor_w // 2
        ay0 = cy - anchor_h // 2
        ax1 = ax0 + anchor_w
        ay1 = ay0 + anchor_h

        n = self.rect_count
        min_dx = float(W) / (n * 5.0)
        min_dy = float(H) / (n * 5.0)
        step_x = max(8, int(math.ceil(min_dx)))
        step_y = max(6, int(math.ceil(min_dy)))

        left_space = ax0
        right_space = W - ax1
        top_space = ay0
        bottom_space = H - ay1

        def _offsets(space: int, step: int) -> List[int]:
            values=list(range(0, space + 1, step))
            return self._rng.sample(values, min(len(values), n))

        left_offsets = _offsets(left_space, step_x)
        right_offsets = _offsets(right_space, step_x)
        top_offsets = _offsets(top_space, step_y)
        bottom_offsets = _offsets(bottom_space, step_y)

        left_indices = list(range(n))
        right_indices = list(range(n))
        top_indices = list(range(n))
        bottom_indices = list(range(n))

        max_attempts = 999999999999
        for attempt in range(max_attempts):
            if attempt:
                self._rng.shuffle(left_indices)
                self._rng.shuffle(right_indices)
                self._rng.shuffle(top_indices)
                self._rng.shuffle(bottom_indices)

            rects: List[RectSpec] = []
            for slot in range(n):
                color_name, color_value = colors[slot]
                li = left_indices[slot]
                ri = right_indices[-(slot + 1)]
                ti = top_indices[(slot + attempt) % n]
                bi = bottom_indices[-((slot + attempt) % n) - 1]

                x0 = ax0 - left_offsets[li]
                x1 = ax1 + right_offsets[ri]
                y0 = ay0 - top_offsets[ti]
                y1 = ay1 + bottom_offsets[bi]

                x0 = max(0, x0)
                x1 = min(W, x1)
                y0 = max(0, y0)
                y1 = min(H, y1)

                w = max(8, x1 - x0)
                h = max(8, y1 - y0)

                rects.append(
                    RectSpec(
                        x=x0,
                        y=y0,
                        w=w,
                        h=h,
                        color=color_value,
                        name=color_name,
                        z=slot,
                    )
                )

            z_perm = list(range(n))
            self._rng.shuffle(z_perm)
            for idx, z in enumerate(z_perm):
                rects[idx].z = z

            if not self._check_min_side_separation(rects, W, H):
                continue

            if self._validate_visible_pairwise(rects, require_strong=self.require_strong_order):
                return rects

        random_attempts = max_attempts * 8
        for _ in range(random_attempts):
            rects = []
            for i in range(n):
                color_name, color_value = colors[i]
                extra_w = self._rng.randint(anchor_w // 2, max(anchor_w, W // 2))
                extra_h = self._rng.randint(anchor_h // 2, max(anchor_h, H // 2))
                x0 = max(0, ax0 - self._rng.randint(0, extra_w))
                y0 = max(0, ay0 - self._rng.randint(0, extra_h))
                x1 = min(W, ax1 + self._rng.randint(0, extra_w))
                y1 = min(H, ay1 + self._rng.randint(0, extra_h))
                w = max(8, x1 - x0)
                h = max(8, y1 - y0)
                jitter_x = self._rng.randint(-min(8, x0), min(8, W - x1))
                jitter_y = self._rng.randint(-min(8, y0), min(8, H - y1))
                x = x0 + jitter_x
                y = y0 + jitter_y
                rects.append(
                    RectSpec(
                        x=x,
                        y=y,
                        w=w,
                        h=h,
                        color=color_value,
                        name=color_name,
                        z=i,
                    )
                )

            z_perm = list(range(n))
            self._rng.shuffle(z_perm)
            for idx, z in enumerate(z_perm):
                rects[idx].z = z

            if not self._check_min_side_separation(rects, W, H):
                continue

            if self._validate_visible_pairwise(rects, require_strong=self.require_strong_order):
                return rects

        raise RuntimeError("failed to generate rects with current configuration")

    # --- Visible shared-boundary computation (user-specified algorithm) ------------

    def _check_min_side_separation(self, rects: Sequence[RectSpec], W: int, H: int) -> bool:
        # There are 2*rect_count vertical sides (x and x+w) and 2*rect_count horizontal sides (y and y+h)
        # Require every pair among each group to differ by at least 1/(rect_count*5) of width/height
        import math

        n = max(1, len(rects))
        min_dx = float(W) / (n * 5.0)
        min_dy = float(H) / (n * 5.0)

        xs: List[float] = []
        ys: List[float] = []
        for r in rects:
            xs.extend([float(r.x), float(r.x + r.w)])
            ys.extend([float(r.y), float(r.y + r.h)])

        def _min_adjacent_diff(vals: List[float]) -> float:
            if len(vals) < 2:
                return float("inf")
            vals_sorted = sorted(vals)
            return min(vals_sorted[i + 1] - vals_sorted[i] for i in range(len(vals_sorted) - 1))

        return _min_adjacent_diff(xs) >= min_dx and _min_adjacent_diff(ys) >= min_dy

    @staticmethod
    def _subtract_intervals(segments: List[Tuple[int, int]], cut: Tuple[int, int]) -> List[Tuple[int, int]]:
        a, b = cut
        if b <= a:
            return segments
        out: List[Tuple[int, int]] = []
        for s0, s1 in segments:
            if s1 <= a or s0 >= b:
                out.append((s0, s1))
                continue
            if s0 < a:
                out.append((s0, a))
            if b < s1:
                out.append((b, s1))
        return [(u0, u1) for (u0, u1) in out if u1 > u0]

    def _visible_shared_segment_count(self, rects: Sequence[RectSpec], i: int, j: int) -> int:
        # Determine which is on top (higher z)
        a, b = rects[i], rects[j]
        top, bottom = (a, b) if a.z > b.z else (b, a)

        ax0, ay0, ax1, ay1 = top.x, top.y, top.x + top.w, top.y + top.h
        bx0, by0, bx1, by1 = bottom.x, bottom.y, bottom.x + bottom.w, bottom.y + bottom.h

        segments_count = 0

        # Collect candidate segments along top's edges that coincide with bottom's interior or border
        # Horizontal edges at y = ay0 (top) and y = ay1 (bottom)
        def horizontal_candidates(y_const: int) -> List[Tuple[int, int]]:
            candidates: List[Tuple[int, int]] = []
            # Interior overlap along x if y inside bottom
            if by0 < y_const < by1:
                x0 = max(ax0, bx0)
                x1 = min(ax1, bx1)
                if x1 > x0:
                    candidates.append((x0, x1))
            # Edge-touching overlap
            if y_const in (by0, by1):
                x0 = max(ax0, bx0)
                x1 = min(ax1, bx1)
                if x1 > x0:
                    candidates.append((x0, x1))
            return candidates

        def vertical_candidates(x_const: int) -> List[Tuple[int, int]]:
            candidates: List[Tuple[int, int]] = []
            # Interior overlap along y if x inside bottom
            if bx0 < x_const < bx1:
                y0 = max(ay0, by0)
                y1 = min(ay1, by1)
                if y1 > y0:
                    candidates.append((y0, y1))
            # Edge-touching overlap
            if x_const in (bx0, bx1):
                y0 = max(ay0, by0)
                y1 = min(ay1, by1)
                if y1 > y0:
                    candidates.append((y0, y1))
            return candidates

        # Assemble candidates for each edge and subtract occlusions from rectangles with z > top.z
        occluders = [r for r in rects if r.z > top.z]

        # Horizontal edges -> x-intervals
        for y_const in (ay0, ay1):
            intervals = horizontal_candidates(y_const)
            if not intervals:
                continue
            # Subtract occlusions along this scanline
            for occ in occluders:
                oy0, oy1 = occ.y, occ.y + occ.h
                if oy0 <= y_const <= oy1:
                    cut = (max(ax0, occ.x), min(ax1, occ.x + occ.w))
                    if cut[1] > cut[0]:
                        new_intervals: List[Tuple[int, int]] = []
                        for seg in intervals:
                            new_intervals.extend(self._subtract_intervals([seg], cut))
                        intervals = new_intervals
                        if not intervals:
                            break
            segments_count += sum(1 for _ in intervals)

        # Vertical edges -> y-intervals
        for x_const in (ax0, ax1):
            intervals = vertical_candidates(x_const)
            if not intervals:
                continue
            for occ in occluders:
                ox0, ox1 = occ.x, occ.x + occ.w
                if ox0 <= x_const <= ox1:
                    cut = (max(ay0, occ.y), min(ay1, occ.y + occ.h))
                    if cut[1] > cut[0]:
                        new_intervals: List[Tuple[int, int]] = []
                        for seg in intervals:
                            new_intervals.extend(self._subtract_intervals([seg], cut))
                        intervals = new_intervals
                        if not intervals:
                            break
            segments_count += sum(1 for _ in intervals)

        return segments_count

    def _validate_visible_pairwise(self, rects: Sequence[RectSpec], *, require_strong: bool) -> bool:
        need = 2 if require_strong else 1
        n = len(rects)
        for i in range(n):
            for j in range(i + 1, n):
                if self._visible_shared_segment_count(rects, i, j) < need:
                    return False
        return True

    # --- Rendering ------------------------------------------------------------------

    def _render_puzzle(self, rects: Sequence[RectSpec]) -> Image.Image:
        W, H = self.canvas_dimensions
        img = Image.new("RGB", (W, H), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        # Draw from bottom to top
        sorted_rects = sorted(rects, key=lambda r: r.z)
        for r in sorted_rects:
            draw.rectangle([r.x, r.y, r.x + r.w, r.y + r.h], fill=r.color, outline=(0, 0, 0))
        return img

    def _render_solution(self, rects: Sequence[RectSpec]) -> Image.Image:
        W, H = self.canvas_dimensions
        img = Image.new("RGB", (W, H), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        order = [r for r in sorted(rects, key=lambda r: r.z, reverse=True)]  # top-most first
        band_h = max(8, H // max(1, len(order) * 2))
        gap = max(4, band_h // 4)
        y = gap
        for r in order:
            if y + band_h > H - gap:
                band_h = max(4, H - gap - y)
            draw.rectangle([gap, y, W - gap, y + band_h], fill=r.color, outline=(0, 0, 0))
            y += band_h + gap
        return img


__all__ = ["RectsGenerator", "RectsPuzzleRecord", "RectSpec"]


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate colored-rectangles puzzles")
    parser.add_argument("count", type=int, help="Number of puzzles to create")
    parser.add_argument("--output-dir", type=Path, default=Path("data/rects"))
    parser.add_argument("--rect-count", type=int, default=4)
    parser.add_argument("--canvas-size", type=int, default=384)
    parser.add_argument("--aspect", type=float, default=None, help="Canvas aspect ratio W/H")
    parser.add_argument("--weak", action="store_true", help="Allow weak (edge-touching) order evidence")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--prompt",type=str, default=None)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)
    gen = RectsGenerator(
        args.output_dir,
        rect_count=args.rect_count,
        canvas_size=args.canvas_size,
        canvas_aspect_ratio=args.aspect,
        require_strong_order=not args.weak,
        seed=args.seed,
        prompt=args.prompt
    )
    records = [gen.create_random_puzzle() for _ in range(max(1, args.count))]
    gen.write_metadata(records, Path(args.output_dir) / "data.json")


if __name__ == "__main__":
    main()
