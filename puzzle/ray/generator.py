"""Ray-and-mirrors puzzle generator (non-grid, continuous geometry).

Random line-segment mirrors placed on a blank canvas (no grid). A ray is shot
from the left edge and reflects off mirrors using geometric reflection. Five
points (A–E) are placed; exactly one sits on the ray path. Prompt requires
speaking the option via NATO phonetic alphabet.
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
class MirrorSpec:
    x0: float
    y0: float
    x1: float
    y1: float

    def to_dict(self) -> dict:
        return {"x0": self.x0, "y0": self.y0, "x1": self.x1, "y1": self.y1}


@dataclass
class PointSpec:
    x: float
    y: float
    label: str  # 'A'..'E'

    def to_dict(self) -> dict:
        return {"x": self.x, "y": self.y, "label": self.label}


@dataclass
class RayStart:
    x: float
    y: float
    dx: float
    dy: float

    def to_dict(self) -> dict:
        return {"x": self.x, "y": self.y, "dx": self.dx, "dy": self.dy}


@dataclass
class RayPuzzleRecord:
    id: str
    prompt: str
    canvas_dimensions: Tuple[int, int]
    margin: int
    start: RayStart
    mirrors: List[MirrorSpec]
    points: List[PointSpec]
    correct_option: str  # 'A'..'E'
    reflections: int
    image: str
    solution_image_path: str

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "prompt": self.prompt,
            "canvas_dimensions": list(self.canvas_dimensions),
            "margin": self.margin,
            "start": self.start.to_dict(),
            "mirrors": [m.to_dict() for m in self.mirrors],
            "points": [p.to_dict() for p in self.points],
            "correct_option": self.correct_option,
            "reflections": self.reflections,
            "image": self.image,
            "solution_image_path": self.solution_image_path,
            "type": "ray",
        }


class RayGenerator(AbstractPuzzleGenerator[RayPuzzleRecord]):
    """Generate continuous ray puzzles with line-segment mirrors and NATO prompt."""

    def __init__(
        self,
        output_dir: PathLike = "data/ray",
        *,
        canvas_size: int = 480,
        aspect: Optional[float] = None,
        mirror_count: int = 12,
        min_reflections: int = 2,
        prompt: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(output_dir)
        self.base_size = int(canvas_size)
        self.aspect = float(aspect) if aspect and aspect > 0 else None
        if self.aspect and self.aspect >= 1.0:
            W = int(round(self.base_size * self.aspect))
            H = self.base_size
        elif self.aspect and self.aspect < 1.0:
            W = self.base_size
            H = int(round(self.base_size / self.aspect))
        else:
            W = H = self.base_size
        self.canvas_dimensions = (W, H)
        self.margin = max(16, int(round(min(W, H) * 0.05)))
        self.play_left = self.margin
        self.play_top = self.margin
        self.play_right = W - self.margin
        self.play_bottom = H - self.margin

        self.mirror_count = max(3, int(mirror_count))
        self.min_reflections = max(0, int(min_reflections))
        self._rng = random.Random(seed)

        if prompt is None:
            prompt = (
                "Physics simulation. A green arrow marks the laser source. Fire a single laser ray from the arrow; it moves straight and reflects perfectly off the mirrors until it exits. Exactly one labeled point lies on the ray’s path."#There is exactly one correct point. Speak out the option in phonetic alphabet. Alpha for A, Bravo for B, Charlie for C, Delta for D, Echo for E."
            )
        self.prompt = prompt

        self.puzzle_dir = Path(self.output_dir) / "puzzles"
        self.solution_dir = Path(self.output_dir) / "solutions"
        self.puzzle_dir.mkdir(parents=True, exist_ok=True)
        self.solution_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------- public API -----------------------
    def create_puzzle(self, *, puzzle_id: Optional[str] = None) -> RayPuzzleRecord:
        puzzle_uuid = puzzle_id or str(uuid.uuid4())

        for _ in range(1000):
            mirrors = self._random_mirrors()
            start = self._random_start()
            path, reflections = self._trace_path(mirrors, start)
            if reflections < self.min_reflections or len(path) < 2:
                continue

            correct_xy = self._choose_correct_point_on_path(path)
            if correct_xy is None:
                continue

            points, correct_label = self._place_points(correct_xy, path)
            if points is None:
                continue

            puzzle_img = self._render(mirrors, points, start, path=None)
            solution_img = self._render(mirrors, points, start, path=path, correct_label=correct_label)

            puzzle_path = self.puzzle_dir / f"{puzzle_uuid}_puzzle.png"
            solution_path = self.solution_dir / f"{puzzle_uuid}_solution.png"
            puzzle_img.save(puzzle_path)
            solution_img.save(solution_path)

            return RayPuzzleRecord(
                id=puzzle_uuid,
                prompt=self.prompt,
                canvas_dimensions=self.canvas_dimensions,
                margin=self.margin,
                start=start,
                mirrors=mirrors,
                points=points,
                correct_option=correct_label,
                reflections=reflections,
                image=self.relativize_path(puzzle_path),
                solution_image_path=self.relativize_path(solution_path),
            )

        raise RuntimeError("Failed to generate a valid ray puzzle after many attempts")

    def create_random_puzzle(self) -> RayPuzzleRecord:
        return self.create_puzzle()

    # ----------------------- internals ------------------------
    def _random_mirrors(self) -> List[MirrorSpec]:
        mirrors: List[MirrorSpec] = []
        W, H = self.canvas_dimensions
        pad = self.margin + 8
        for _ in range(self.mirror_count * 6):
            if len(mirrors) >= self.mirror_count:
                break
            x0 = self._rng.uniform(pad, W - pad)
            y0 = self._rng.uniform(pad, H - pad)
            length = self._rng.uniform(0.12, 0.22) * min(W, H)
            angle = self._rng.uniform(0, 3.14159)
            x1 = x0 + length * float(__import__("math").cos(angle))
            y1 = y0 + length * float(__import__("math").sin(angle))
            # clip to play area
            x1 = max(pad, min(W - pad, x1))
            y1 = max(pad, min(H - pad, y1))
            if (x1 - x0) ** 2 + (y1 - y0) ** 2 < 9.0:
                continue
            mirrors.append(MirrorSpec(x0=x0, y0=y0, x1=x1, y1=y1))
        return mirrors

    def _random_start(self) -> RayStart:
        # Launch from left inner edge, random y, heading right
        W, H = self.canvas_dimensions
        y = self._rng.uniform(self.play_top + 10, self.play_bottom - 10)
        x = self.play_left + 1.0
        dy = self._rng.uniform(-0.5, 0.5)
        dx = 1.0
        import math
        norm = math.hypot(dx, dy) or 1.0
        dx, dy = dx / norm, dy / norm
        return RayStart(x=x, y=y, dx=dx, dy=dy)

    def _trace_path(
        self,
        mirrors: Sequence[MirrorSpec],
        start: RayStart,
        *,
        bounce_limit: int = 64,
    ) -> Tuple[List[Tuple[float, float]], int]:
        import math

        def ray_segment_intersection(p, v, a, b):
            # Solve p + t v = a + u (b-a), with t>0 and u in (0,1)
            px, py = p
            vx, vy = v
            ax, ay = a
            sx, sy = (b[0] - ax), (b[1] - ay)
            # 2D cross products
            def cross(x1, y1, x2, y2):
                return x1 * y2 - y1 * x2
            den = cross(vx, vy, sx, sy)
            if abs(den) < 1e-8:
                return None  # parallel or nearly so
            wx, wy = (ax - px), (ay - py)
            t = cross(wx, wy, sx, sy) / den
            u = cross(wx, wy, vx, vy) / den
            if t <= 1e-6 or u <= 1e-6 or u >= 1 - 1e-6:
                return None
            return (t, u)

        def reflect(v, a, b):
            # Reflect vector v across the line defined by segment ab (tangent)
            vx, vy = v
            tx, ty = (b[0] - a[0], b[1] - a[1])
            norm = math.hypot(tx, ty)
            if norm <= 1e-8:
                return v
            tx, ty = tx / norm, ty / norm
            dot = vx * tx + vy * ty
            rx, ry = 2 * dot * tx - vx, 2 * dot * ty - vy
            # normalize to unit length to avoid drift
            rnorm = math.hypot(rx, ry)
            if rnorm <= 1e-12:
                return (vx, vy)
            return (rx / rnorm, ry / rnorm)

        W, H = self.canvas_dimensions
        left, top, right, bottom = self.play_left, self.play_top, self.play_right, self.play_bottom
        p = (float(start.x), float(start.y))
        v = (float(start.dx), float(start.dy))
        path: List[Tuple[float, float]] = [p]
        reflections = 0

        for _ in range(bounce_limit):
            # compute exit t
            txs = []
            vx, vy = v
            px, py = p
            if vx > 1e-9:
                txs.append(( (right - px) / vx, (right, None)))
            elif vx < -1e-9:
                txs.append(( (left - px) / vx, (left, None)))
            if vy > 1e-9:
                txs.append(( (bottom - py) / vy, (None, bottom)))
            elif vy < -1e-9:
                txs.append(( (top - py) / vy, (None, top)))
            t_exit = min((t for t, _ in txs if t > 1e-6), default=float("inf"))

            # find nearest mirror hit
            best_t = float("inf")
            best_seg = None
            best_pt = None
            for m in mirrors:
                a = (m.x0, m.y0)
                b = (m.x1, m.y1)
                hit = ray_segment_intersection(p, v, a, b)
                if not hit:
                    continue
                t, u = hit
                if 1e-6 < t < best_t:
                    best_t = t
                    best_seg = (a, b)
                    best_pt = (px + vx * t, py + vy * t)

            if best_t < t_exit and best_seg is not None and best_pt is not None:
                # reflect
                p = best_pt
                path.append(p)
                v = reflect(v, *best_seg)
                reflections += 1
                continue

            # no mirror before exit -> exit at boundary
            if not math.isfinite(t_exit) or t_exit <= 1e-6:
                break
            p = (px + vx * t_exit, py + vy * t_exit)
            path.append(p)
            break

        return path, reflections

    def _choose_correct_point_on_path(self, path: Sequence[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
        # Choose a point along an interior segment
        if len(path) < 2:
            return None
        import math
        seg_indices = list(range(len(path)//2,max(1, len(path) - 1)))
        self._rng.shuffle(seg_indices)
        for idx in seg_indices:
            a = path[idx]
            b = path[idx + 1] if idx + 1 < len(path) else None
            if b is None:
                continue
            ax, ay = a
            bx, by = b
            if (ax - bx) ** 2 + (ay - by) ** 2 < 25.0:
                continue
            t = self._rng.uniform(0.2, 0.8)
            x = ax + (bx - ax) * t
            y = ay + (by - ay) * t
            # ensure inside play area
            if not (self.play_left <= x <= self.play_right and self.play_top <= y <= self.play_bottom):
                continue
            return (x, y)
        return None

    def _place_points(
        self,
        correct_xy: Tuple[float, float],
        path: Sequence[Tuple[float, float]],
    ) -> Tuple[Optional[List[PointSpec]], Optional[str]]:
        import math

        labels = ["A", "B", "C", "D", "E"]
        self._rng.shuffle(labels)
        correct_label = labels[0]

        def seg_dist(p, a, b):
            # distance from p to segment ab
            px, py = p
            ax, ay = a
            bx, by = b
            vx, vy = (bx - ax), (by - ay)
            wx, wy = (px - ax), (py - ay)
            vv = vx * vx + vy * vy
            if vv <= 1e-9:
                return math.hypot(px - ax, py - ay)
            t = max(0.0, min(1.0, (wx * vx + wy * vy) / vv))
            cx = ax + t * vx
            cy = ay + t * vy
            return math.hypot(px - cx, py - cy)

        points: List[PointSpec] = []
        points.append(PointSpec(x=correct_xy[0], y=correct_xy[1], label=correct_label))

        # decoys: far from any segment and from the correct point
        W, H = self.canvas_dimensions
        min_sep = max(8.0, 0.03 * min(W, H))
        tries = 0
        while len(points) < 5 and tries < 2000:
            tries += 1
            x = self._rng.uniform(self.play_left + 10, self.play_right - 10)
            y = self._rng.uniform(self.play_top + 10, self.play_bottom - 10)
            # far from path
            ok = True
            for i in range(len(path) - 1):
                if seg_dist((x, y), path[i], path[i + 1]) < min_sep:
                    ok = False
                    break
            if not ok:
                continue
            # far from existing points
            for p in points:
                if math.hypot(p.x - x, p.y - y) < min_sep:
                    ok = False
                    break
            if not ok:
                continue
            label = labels[len(points)]
            points.append(PointSpec(x=x, y=y, label=label))

        if len(points) != 5:
            # Relax constraints progressively
            target = 5
            for relax in (0.5, 0.25):
                while len(points) < target and tries < 4000:
                    tries += 1
                    x = self._rng.uniform(self.play_left + 8, self.play_right - 8)
                    y = self._rng.uniform(self.play_top + 8, self.play_bottom - 8)
                    ok = True
                    # loosen distance to path
                    for i in range(len(path) - 1):
                        if seg_dist((x, y), path[i], path[i + 1]) < (min_sep * relax):
                            ok = False
                            break
                    if not ok:
                        continue
                    for p in points:
                        if math.hypot(p.x - x, p.y - y) < (min_sep * relax):
                            ok = False
                            break
                    if not ok:
                        continue
                    label = labels[len(points)]
                    points.append(PointSpec(x=x, y=y, label=label))
                if len(points) == target:
                    break
        if len(points) != 5:
            return None, None
        self._rng.shuffle(points)
        return points, correct_label

    # ----------------------- rendering ------------------------
    def _draw_border(self, draw: ImageDraw.ImageDraw) -> None:
        left, top, right, bottom = self.play_left, self.play_top, self.play_right, self.play_bottom
        return
        draw.rectangle((left, top, right, bottom), outline=(0, 0, 0), width=2)

    def _render(
        self,
        mirrors: Sequence[MirrorSpec],
        points: Sequence[PointSpec],
        start: RayStart,
        *,
        path: Optional[Sequence[Tuple[int, int]]],
        correct_label: Optional[str] = None,
    ) -> Image.Image:
        W, H = self.canvas_dimensions
        img = Image.new("RGB", (W, H), (255, 255, 255))
        draw = ImageDraw.Draw(img)

        # border
        self._draw_border(draw)

        # mirrors
        for m in mirrors:
            draw.line((m.x0, m.y0, m.x1, m.y1), fill=(60, 60, 60), width=3)

        # start arrow (green), pointing from left margin to first column center
        # start arrow, oriented with (dx, dy)
        import math as _math
        px, py = start.x, start.y
        dx, dy = start.dx, start.dy
        L = 48.0
        ax0x = int(round(px - dx * L))
        ax0y = int(round(py - dy * L))
        ax1x = int(round(px))
        ax1y = int(round(py))
        draw.line((ax0x, ax0y, ax1x, ax1y), fill=(0, 160, 0), width=4)
        # arrow head
        hx = ax1x
        hy = ax1y
        left_dx = -dy
        left_dy = dx
        head_len = 6.0
        p1 = (int(round(hx - dx * head_len + left_dx * 4)), int(round(hy - dy * head_len + left_dy * 4)))
        p2 = (int(round(hx - dx * head_len - left_dx * 4)), int(round(hy - dy * head_len - left_dy * 4)))
        draw.polygon([(hx, hy), p1, p2], fill=(0, 160, 0))

        # points A–E
        for p in points:
            cx, cy = int(round(p.x)), int(round(p.y))
            r = max(6, int(round(min(W, H) * 0.02)))
            draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline=(0, 0, 200), width=2, fill=(230, 240, 255))
            try:
                font = ImageFont.load_default()
            except Exception:
                font = None
            try:
                box = draw.textbbox((0, 0), p.label, font=font)
                tw, th = (box[2] - box[0], box[3] - box[1])
            except Exception:
                tw, th = (8, 8)
            draw.text((cx - tw // 2, cy - th // 2), p.label, fill=(0, 0, 180), font=font)

        # ray path on solution image
        if path:
            pts: List[Tuple[int, int]] = [(ax1x, ax1y)]
            pts.extend([(int(round(x)), int(round(y))) for (x, y) in path])
            draw.line([coord for pt in pts for coord in pt], fill=(220, 30, 30), width=3)

            # highlight correct label if provided
            if correct_label:
                for p in points:
                    if p.label == correct_label:
                        cx, cy = int(round(p.x)), int(round(p.y))
                        r = max(10, int(round(min(W, H) * 0.03)))
                        draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline=(20, 200, 20), width=3)
                        break

        return img


__all__ = [
    "RayGenerator",
    "RayPuzzleRecord",
    "MirrorSpec",
    "PointSpec",
    "RayStart",
]


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate non-grid ray-and-mirrors puzzles")
    parser.add_argument("count", type=int, help="Number of puzzles to create")
    parser.add_argument("--output-dir", type=Path, default=Path("data/ray"))
    parser.add_argument("--canvas-size", type=int, default=480)
    parser.add_argument("--aspect", type=float, default=None, help="Canvas aspect ratio W/H (e.g., 16/9=1.777)")
    parser.add_argument("--mirror-count", type=int, default=12)
    parser.add_argument("--min-reflections", type=int, default=2)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)
    gen = RayGenerator(
        args.output_dir,
        canvas_size=args.canvas_size,
        aspect=args.aspect,
        mirror_count=args.mirror_count,
        min_reflections=args.min_reflections,
        seed=args.seed,
    )
    records = [gen.create_random_puzzle() for _ in range(max(1, args.count))]
    gen.write_metadata(records, Path(args.output_dir) / "data.json")


if __name__ == "__main__":
    main()
