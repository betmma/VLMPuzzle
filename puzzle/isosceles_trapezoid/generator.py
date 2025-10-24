"""Isosceles Trapezoid puzzle generator.
"""

from __future__ import annotations
from typing import List, Optional, Sequence, Tuple
from PIL import Image, ImageDraw, ImageFont
import math
from ..point_target_base import PointTargetPuzzleGenerator, PointTargetPuzzleRecord, Point

class IsoscelesTrapezoidGenerator(PointTargetPuzzleGenerator):
    """Generate puzzles to find the fourth vertex of an isosceles trapezoid."""
    DEFAULT_OUTPUT_DIR="data/isosceles_trapezoid"
    DEFAULT_PROMPT="Find the fourth vertex that completes the isosceles trapezoid. Mark the fourth vertex red. Speak out which option is the fourth vertex using phonetic alphabet. In portrait, no zoom, no pan."
    DEFAULT_GPT5_PROMPT="Which option is the fourth vertex of the isosceles trapezoid? Answer an option in A-E."

    def create_puzzle(self) -> PointTargetPuzzleRecord:
        """
        Creates an isosceles trapezoid puzzle.
        We define a base (p1, p2) and a third point p3. The goal is to find p4 such that
        p1-p2 is parallel to p3-p4, and the non-parallel sides (p1-p3 and p2-p4) are equal length.
        This is achieved by reflecting p3 across the perpendicular bisector of the base p1-p2.
        A simpler vector-based calculation is used here.
        """
        tries = 0
        min_base_len = self.canvas_short_side * 0.3
        min_height = self.canvas_short_side * 0.15

        while tries < 9999:
            # Pick two points for the main base of the trapezoid
            p1, p2 = self.pick_target_point(0.8), self.pick_target_point(0.8)
            
            # Pick the third point, which will form one of the non-parallel legs
            p3 = self.pick_target_point(0.8)

            # --- Vector calculation to find the fourth point (target) ---
            v_base = Point(p2.x - p1.x, p2.y - p1.y)
            v_leg = Point(p3.x - p1.x, p3.y - p1.y)
            
            base_len_sq = v_base.x**2 + v_base.y**2
            
            # Avoid division by zero or a very short base
            if base_len_sq < min_base_len**2:
                tries += 1
                continue

            # Project the leg vector onto the base vector to find the parallel component
            dot_product = v_leg.x * v_base.x + v_leg.y * v_base.y
            proj_factor = dot_product / base_len_sq
            
            # The projected vector component along the base
            v_proj = Point(proj_factor * v_base.x, proj_factor * v_base.y)

            # The perpendicular component (determines the height)
            v_perp = Point(v_leg.x - v_proj.x, v_leg.y - v_proj.y)
            height = math.sqrt(v_perp.x**2 + v_perp.y**2)
            
            # --- Sanity checks for a good puzzle ---
            # 1. Ensure the trapezoid has a reasonable height (not flat)
            if height < min_height:
                tries += 1
                continue
            
            # 2. Ensure legs do not cross (top base length must be positive).
            # The top base length is base_len * (1 - 2 * proj_factor).
            # This means proj_factor must be less than 0.5. We'll use a slightly smaller
            # threshold to avoid a tiny top base.
            if proj_factor >= 0.45 or proj_factor <= 0.05:
                tries += 1
                continue
            
            # Calculate the fourth point (target).
            # p4 = p2 + (p3 - p1) - 2 * v_proj
            target_point = Point(
                x = p2.x + v_leg.x - 2 * v_proj.x,
                y = p2.y + v_leg.y - 2 * v_proj.y,
            )
            
            # 3. Ensure the target point is within the canvas
            if not self.inside_canvas(target_point):
                tries += 1
                continue

            # All checks passed, we have a valid puzzle
            break
        
        if tries >= 9999:
            raise RuntimeError("Failed to generate a valid isosceles trapezoid puzzle.")

        self.trapezoid_points = (p1, p2, p3, target_point)
        self.place_candidates(target_point)
        record = self.save_puzzle()
        record.trapezoid_points = self.trapezoid_points
        return record

    def _render(self, highlight_label: Optional[str]) -> Image.Image:
        draw, base = self.get_draw_base()
        
        p1, p2, p3, target = self.trapezoid_points
        
        # Draw the three given vertices/sides
        self.draw_line(draw, [p3, p1, p2])
        
        # In the solution image, complete the trapezoid
        if highlight_label:
            self.draw_line(draw, [p2, target, p3])

        self.draw_candidates(
            draw,
            highlight_label=highlight_label,
        )
        return base

def main(argv: Optional[List[str]] = None) -> None:
    IsoscelesTrapezoidGenerator.main(IsoscelesTrapezoidGenerator, argv)

if __name__ == "__main__":
    main()