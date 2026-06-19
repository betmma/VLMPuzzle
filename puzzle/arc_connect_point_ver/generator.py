"""Arc connection point puzzle generator.

Simplified version with vertical mask and side arcs.
"""

from __future__ import annotations
import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from PIL import Image, ImageDraw

from ..base import PathLike
from ..point_target_base import (
    PointCandidate,
    PointTargetPuzzleGenerator,
    PointTargetPuzzleRecord,
    VideoRenderer,
)

@dataclass
class CircleSpec:
    cx: float
    cy: float
    r: float

    def bbox(self) -> Tuple[int, int, int, int]:
        return (int(round(self.cx - self.r)), int(round(self.cy - self.r)),
                int(round(self.cx + self.r)), int(round(self.cy + self.r)))

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
        seed: Optional[int] = None,
        prompt: Optional[str] = None,
        record_video: bool = False,
    ) -> None:
        super().__init__(
            output_dir=output_dir,
            canvas_width=canvas_width,
            aspect=aspect,
            seed=seed,
            prompt=prompt,
            record_video=record_video,
        )
        self.mask_fraction = 0.35
        self.arc_span_deg = 20.0
        self.circles: List[CircleSpec] = []
        self.mask_x_center: float = 0.0
        self.mask_width: float = 0.0

    def create_puzzle(self) -> PointTargetPuzzleRecord:
        width, height = self.canvas_dimensions
        self.mask_width = width * self.mask_fraction
        self.mask_x_center = width / 2.0
        span_rad = math.radians(self.arc_span_deg)
        margin = self.margin
        stroke = max(3, int(round(min(width, height) * 0.015)))

        for _ in range(1000):
            # Restore original radius logic: uniform(0.38, 0.55) * min_dimension
            radius = self.rng.uniform(0.38, 0.55) * min(width, height)
            
            mask_left = self.mask_x_center - self.mask_width/2
            mask_right = self.mask_x_center + self.mask_width/2
            
            cx = self.rng.uniform(self.mask_x_center - 0.4 * radius, self.mask_x_center + 0.4 * radius)
            
            # Validate strict crossing: check if valid Y exists at mask boundaries
            check_x_left = mask_left
            check_x_right = mask_right
            if abs(check_x_left - cx) >= radius or abs(check_x_right - cx) >= radius:
                continue

            gap = self.rng.uniform(height * 0.05, height * 0.08)
            
            # We want 5 candidate arcs. Let's space their crossings at mask_right evenly.
            # Pick center crossing Y for the middle candidate (index 2)
            base_crossing_y = self.rng.uniform(height * 0.2, height * 0.8)
            
            # Generate 5 crossing Ys
            crossing_ys = [base_crossing_y + (i - 2) * gap for i in range(5)]
            
            temp_circles = []
            temp_candidates_points = []
            valid_group = True
            
            sign = 1 if self.rng.choice([True, False]) else -1

            for y_cross in crossing_ys:
                dx = mask_right - cx
                dy_term = math.sqrt(max(0, radius**2 - dx**2))
                
                cy = y_cross + sign * dy_term
                
                c = CircleSpec(cx, cy, radius)
                
                # Now validate endpoints
                # Requirement: "visible part has 20 degrees"
                theta_base = math.atan2(y_cross - cy, mask_right - cx)
                
                test_pt = self._point_on_circle(c, theta_base + 0.01)
                direction = 1.0 if test_pt[0] > mask_right else -1.0
                
                theta_end = theta_base + direction * span_rad
                
                # Validate endpoints y-coordinates
                p_start = (mask_right, y_cross)
                p_end = self._point_on_circle(c, theta_end)
                
                # Check right arc containment
                if not (margin < p_start[1] < height - margin): valid_group = False
                if not (margin < p_end[1] < height - margin): valid_group = False
                if p_end[0] <= mask_right: valid_group = False
                
                # Check left arc containment (theoretical left arc for this circle)
                cross_ys_left = self._get_y_at_x(c, mask_left)
                if not cross_ys_left: 
                    valid_group = False
                else:
                    y_cross_left = min(cross_ys_left, key=lambda y: abs(y - y_cross))
                    theta_left_base = math.atan2(y_cross_left - c.cy, mask_left - c.cx)
                    
                    test_pt_left = self._point_on_circle(c, theta_left_base - 0.01)
                    dir_left = -1.0 if test_pt_left[0] < mask_left else 1.0
                    theta_left_end = theta_left_base + dir_left * span_rad
                    
                    p_left_start = (mask_left, y_cross_left)
                    p_left_end = self._point_on_circle(c, theta_left_end)
                    
                    if not (margin < p_left_start[1] < height - margin): valid_group = False
                    if not (margin < p_left_end[1] < height - margin): valid_group = False
                    if p_left_end[0] >= mask_left: valid_group = False

                if not valid_group: break
                
                temp_circles.append(c)
                temp_candidates_points.append(p_end)

            
            if valid_group:
                self.circles = temp_circles
                labels = ["A", "B", "C", "D", "E"]
                self.candidates = [
                    PointCandidate(x=pt[0] + 15, y=pt[1], label=l) 
                    for pt, l in zip(temp_candidates_points, labels)
                ]
                self.correct_label = labels[self.rng.randint(0, 4)]
                break
        else:
             raise RuntimeError("Failed to generate valid geometry")

        # Create unique ID and save
        return self.save_puzzle()

    def _get_y_at_x(self, c: CircleSpec, x: float) -> Optional[List[float]]:
        dist_x = abs(x - c.cx)
        if dist_x > c.r:
            return None
        dy = math.sqrt(c.r**2 - dist_x**2)
        return [c.cy - dy, c.cy + dy]

    def _point_on_circle(self, c: CircleSpec, theta: float) -> Tuple[float, float]:
        return (c.cx + c.r * math.cos(theta), c.cy + c.r * math.sin(theta))

    def _render(self, highlight_label: Optional[str], mask_factor: float = 1.0) -> Image.Image:
        width, height = self.canvas_dimensions
        draw, base = self.get_draw_base()
        
        span_rad = math.radians(self.arc_span_deg)
        stroke = max(3, int(round(min(width, height) * 0.015)))

        mask_left = self.mask_x_center - self.mask_width/2
        mask_right = self.mask_x_center + self.mask_width/2

        for i, c in enumerate(self.circles):
            # Calculate theta at center line for drawing "from center"
            # We need to find the specific crossing that corresponds to our generated arc
            # Our generated arc was based on a specific y crossing at mask_right.
            # To be robust, let's find the crossing at mask_right closest to the candidate point
            cross_ys_right = self._get_y_at_x(c, mask_right)
            if not cross_ys_right: continue # Should not happen
            y_cross_right = min(cross_ys_right, key=lambda y: abs(y - self.candidates[i].y))
            
            theta_right_base = math.atan2(y_cross_right - c.cy, mask_right - c.cx)
            
            # Determine direction towards Right (away from mask)
            test_pt = self._point_on_circle(c, theta_right_base + 0.01)
            direction = 1.0 if test_pt[0] > mask_right else -1.0
            
            # Visual Requirement: "Draw from central vertical line to right end"
            # But the "20 degrees" requirement applies to the "Visible" part (mask_right to end).
            # So End Angle = Angle(mask_right) + direction * 20deg
            # Start Angle = Angle(center_line)
            
            theta_end = theta_right_base + direction * span_rad # This ensures 20 deg visible from mask edge
            
            # Find center crossing
            cross_ys_center = self._get_y_at_x(c, self.mask_x_center)
            if not cross_ys_center: continue
            y_cross_center = min(cross_ys_center, key=lambda y: abs(y - y_cross_right))
            theta_center = math.atan2(y_cross_center - c.cy, self.mask_x_center - c.cx)
            
            deg_center = math.degrees(theta_center)
            deg_end = math.degrees(theta_end)
            
            # Draw Right Arc
            self._draw_arc_safe(draw, c, deg_center, deg_end, (40, 40, 40), stroke)
            
            # Draw Left Arc (Answer only)
            if self.candidates[i].label == self.correct_label:
                # Mirror logic for left side
                # Visible part starts at mask_left
                cross_ys_left = self._get_y_at_x(c, mask_left)
                if cross_ys_left:
                    y_cross_left = min(cross_ys_left, key=lambda y: abs(y - y_cross_center))
                    theta_left_base = math.atan2(y_cross_left - c.cy, mask_left - c.cx)
                    
                    # Direction: towards Left (away from mask towards left)
                    # If direction for right was +1 (clockwise/CCW?), left is usually opposite relative to top/bottom?
                    # Be careful. Let's check test pt.
                    test_pt_left = self._point_on_circle(c, theta_left_base - 0.01)
                    dir_left = -1.0 if test_pt_left[0] < mask_left else 1.0 # Moving towards left means x decreases
                    
                    theta_left_end = theta_left_base + dir_left * span_rad
                    deg_left_end = math.degrees(theta_left_end)
                    
                    self._draw_arc_safe(draw, c, deg_center, deg_left_end, (10, 10, 10), stroke)

        # Draw Mask
        if not highlight_label and mask_factor > 0.01:
            curr_w = self.mask_width * mask_factor
            mx1, mx2 = self.mask_x_center - curr_w / 2, self.mask_x_center + curr_w / 2
            draw.rectangle((mx1, 0, mx2, height), fill=(240, 240, 240))
            draw.line((mx1, 0, mx1, height), fill=(200, 200, 200), width=5)
            draw.line((mx2, 0, mx2, height), fill=(200, 200, 200), width=5)

        self.draw_candidates(draw, highlight_label=highlight_label)
        return base

    def _draw_arc_safe(self, draw: ImageDraw.ImageDraw, c: CircleSpec, d1: float, d2: float, color: Tuple[int, int, int], w: int) -> None:
        d1, d2 = d1 % 360, d2 % 360
        mn, mx = min(d1, d2), max(d1, d2)
        if mx - mn > 180:
            start, end = mx, mn
        else:
            start, end = mn, mx
        draw.arc(c.bbox(), start=start, end=end, fill=color, width=w)

    def save_video_solution(self, pid: str) -> None:
        width, height = self.canvas_dimensions
        renderer = VideoRenderer(width, height, self)
        
        # 1. Hold
        frame1 = self._render(None, 1.0)
        for _ in range(30): renderer.add_pil_frame(frame1)
        
        # 2. Shrink
        steps = 45
        for i in range(steps):
            f = 1.0 - i / (steps - 1)
            renderer.add_pil_frame(self._render(None, f))
            
        # 3. Reveal
        frame2 = self._render(self.correct_label, 0.0)
        for _ in range(45): renderer.add_pil_frame(frame2)
        
        renderer.save(self.solution_dir / f"{pid}_solution.mp4")

def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate arc connect point puzzles")
    parser.add_argument("count", type=int, help="Number of puzzles to create")
    parser.add_argument("--output-dir", type=Path, default=Path(ArcConnectGenerator.DEFAULT_OUTPUT_DIR))
    parser.add_argument("--canvas-width", type=int, default=480)
    parser.add_argument("--aspect", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--use-gpt-5", action="store_true", help="Use GPT5 prompt")
    parser.add_argument("--video", action="store_true", help="Generate video solution")
    return parser.parse_args(argv)

def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)
    generator = ArcConnectGenerator(
        output_dir=args.output_dir,
        canvas_width=args.canvas_width,
        aspect=args.aspect,
        seed=args.seed,
        prompt=ArcConnectGenerator.DEFAULT_GPT5_PROMPT if args.use_gpt_5 and not args.prompt else args.prompt,
        record_video=args.video,
    )
    records = [generator.create_random_puzzle() for _ in range(max(1, args.count))]
    generator.write_metadata(records, generator.output_dir / "data.json")

if __name__ == "__main__":
    main()
