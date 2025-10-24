"""Fermat point puzzle generator.
"""

from __future__ import annotations
from typing import List, Optional, Sequence, Tuple
from PIL import Image, ImageDraw, ImageFont
import math
from ..point_target_base import PointTargetPuzzleGenerator, PointTargetPuzzleRecord, Point

class FermatPointGenerator(PointTargetPuzzleGenerator):
    """Generate puzzles to find the Fermat point of a triangle."""
    DEFAULT_OUTPUT_DIR="data/fermat_point"
    DEFAULT_PROMPT="Find the Fermat point of the triangle. Mark the point red. Speak out which option is the Fermat point using the phonetic alphabet. In portrait, no zoom, no pan."
    DEFAULT_GPT5_PROMPT="Which option is the Fermat point of the triangle? Answer an option in A-E."

    def _dist_sq(self, p1: Point, p2: Point) -> float:
        """Calculates the squared distance between two points."""
        return (p1.x - p2.x)**2 + (p1.y - p2.y)**2

    def _signed_area(self, p1: Point, p2: Point, p3: Point) -> float:
        """Calculates the signed area of a triangle to determine vertex order."""
        return 0.5 * (p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y))

    def _line_intersection(self, p1: Point, p2: Point, p3: Point, p4: Point) -> Optional[Point]:
        """Finds the intersection of two lines defined by (p1, p2) and (p3, p4)."""
        x1, y1 = p1.x, p1.y
        x2, y2 = p2.x, p2.y
        x3, y3 = p3.x, p3.y
        x4, y4 = p4.x, p4.y

        den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(den) < 1e-9: # Lines are parallel or collinear
            return None

        t_num = (x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)
        t = t_num / den
        
        ix = x1 + t * (x2 - x1)
        iy = y1 + t * (y2 - y1)

        return Point(x=ix, y=iy)

    def create_puzzle(self) -> PointTargetPuzzleRecord:
        """
        Creates a puzzle by generating a triangle with no angle > 120 degrees
        and calculating its Fermat point.
        """
        tries = 0
        min_side_len = self.canvas_short_side * 0.4
        cos_120_deg = -0.5

        while tries < 9999:
            tries += 1
            # Pick three candidate points for the triangle vertices
            p1, p2, p3 = (self.pick_target_point(0.8),
                          self.pick_target_point(0.8),
                          self.pick_target_point(0.8))

            # Calculate side lengths squared
            a2 = self._dist_sq(p2, p3)
            b2 = self._dist_sq(p1, p3)
            c2 = self._dist_sq(p1, p2)

            # Ensure the triangle is not too small
            if a2 < min_side_len**2 or b2 < min_side_len**2 or c2 < min_side_len**2:
                continue

            a, b, c = math.sqrt(a2), math.sqrt(b2), math.sqrt(c2)
            
            # Check for degenerate triangle (violates triangle inequality)
            if a + b <= c or a + c <= b or b + c <= a:
                continue

            # Calculate cosines of angles to check the 120-degree constraint
            # This is more robust than calculating angles directly
            cos_A = (b2 + c2 - a2) / (2 * b * c)
            cos_B = (a2 + c2 - b2) / (2 * a * c)
            cos_C = (a2 + b2 - c2) / (2 * a * b)

            if cos_A <= cos_120_deg or cos_B <= cos_120_deg or cos_C <= cos_120_deg:
                continue
            
            # Found a valid triangle
            break
        else:
             raise RuntimeError("Could not generate a valid triangle for Fermat point puzzle after many tries.")

        # Ensure Counter-Clockwise (CCW) order for consistent outward rotation
        if self._signed_area(p1, p2, p3) < 0:
            p2, p3 = p3, p2
        
        # Constants for 60-degree rotation
        cos60 = 0.5
        sin60 = math.sqrt(3) / 2.0

        # For a CCW triangle A,B,C, the outward equilateral vertices are found
        # by rotating by -60 degrees (Clockwise).
        
        # Construct vertex of outward equilateral triangle on side BC (p2, p3) -> A'
        # Rotate C around B by -60 deg. Vector is C-B.
        vx, vy = p3.x - p2.x, p3.y - p2.y
        vx_rot = vx * cos60 + vy * sin60
        vy_rot = -vx * sin60 + vy * cos60
        p_A_prime = Point(p2.x + vx_rot, p2.y + vy_rot)

        # Construct vertex of outward equilateral triangle on side CA (p3, p1) -> B'
        # Rotate A around C by -60 deg. Vector is A-C.
        vx, vy = p1.x - p3.x, p1.y - p3.y
        vx_rot = vx * cos60 + vy * sin60
        vy_rot = -vx * sin60 + vy * cos60
        p_B_prime = Point(p3.x + vx_rot, p3.y + vy_rot)

        # The Fermat point is the intersection of the lines (A, A') and (B, B')
        target_point = self._line_intersection(p1, p_A_prime, p2, p_B_prime)
        self.target_point = target_point
        
        if target_point is None:
            # Fallback if lines are parallel (should not happen for a valid triangle)
            return self.create_puzzle()

        self.triangle_points = (p1, p2, p3)
        self.place_candidates(target_point)
        record = self.save_puzzle()
        record.triangle_points = self.triangle_points
        return record

    def _render(self, highlight_label: Optional[str]) -> Image.Image:
        draw, base = self.get_draw_base()
        
        p1, p2, p3 = self.triangle_points
        # Draw the triangle by connecting its vertices
        self.draw_line(draw, [p1, p2, p3, p1])
        
        if highlight_label:
            # In the solution image, draw lines from vertices to the Fermat point
            # These three lines should meet at 120-degree angles.
            self.draw_line(draw, [p1, self.target_point])
            self.draw_line(draw, [p2, self.target_point])
            self.draw_line(draw, [p3, self.target_point])

        self.draw_candidates(
            draw,
            highlight_label=highlight_label,
        )
        return base

def main(argv: Optional[List[str]] = None) -> None:
    FermatPointGenerator.main(FermatPointGenerator, argv)

if __name__ == "__main__":
    main()