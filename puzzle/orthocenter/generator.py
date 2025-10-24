"""Orthocenter puzzle generator.
"""

from __future__ import annotations
from typing import List, Optional, Sequence, Tuple
from PIL import Image, ImageDraw, ImageFont
import math
from ..point_target_base import PointTargetPuzzleGenerator, PointTargetPuzzleRecord, Point

class OrthocenterGenerator(PointTargetPuzzleGenerator):
    """Generate puzzles to find the orthocenter of a triangle."""
    DEFAULT_OUTPUT_DIR="data/orthocenter"
    DEFAULT_PROMPT="Find the orthocenter (intersection of altitudes) of the triangle and mark it red. Speak out which option is the orthocenter using phonetic alphabet. In portrait, no zoom, no pan."
    DEFAULT_GPT5_PROMPT="Which option is the orthocenter of the triangle? Answer an option in A-E."

    def _calculate_orthocenter(self, p1: Point, p2: Point, p3: Point) -> Optional[Point]:
        """
        Calculates the orthocenter of a triangle defined by three points by solving
        the system of linear equations for two of the triangle's altitudes.
        """
        x1, y1 = p1.x, p1.y
        x2, y2 = p2.x, p2.y
        x3, y3 = p3.x, p3.y

        # We set up a system of linear equations:
        # a1*x + b1*y = d1  (for altitude from p1)
        # a2*x + b2*y = d2  (for altitude from p2)
        
        # The line for the altitude from p1 is perpendicular to side p2-p3.
        # Vector for side p2-p3 is (x3-x2, y3-y2).
        # A perpendicular vector is (y2-y3, x3-x2). Let's use (x2-x3, y2-y3) for coefficients.
        # Equation: (x2-x3)*x + (y2-y3)*y = (x2-x3)*x1 + (y2-y3)*y1
        a1 = x2 - x3
        b1 = y2 - y3
        d1 = a1 * x1 + b1 * y1

        # The line for the altitude from p2 is perpendicular to side p1-p3.
        # Vector for side p1-p3 is (x3-x1, y3-y1).
        # Equation: (x1-x3)*x + (y1-y3)*y = (x1-x3)*x2 + (y1-y3)*y2
        a2 = x1 - x3
        b2 = y1 - y3
        d2 = a2 * x2 + b2 * y2
        
        # Solve using Cramer's rule. The determinant is related to the area of the triangle.
        determinant = a1 * b2 - a2 * b1

        # If determinant is close to zero, the points are collinear, and the
        # orthocenter is undefined.
        if abs(determinant) < 1e-9:
            return None

        ortho_x = (d1 * b2 - d2 * b1) / determinant
        ortho_y = (a1 * d2 - a2 * d1) / determinant

        return Point(x=ortho_x, y=ortho_y)

    def _get_altitude_foot(self, p_vertex: Point, p_base1: Point, p_base2: Point) -> Point:
        """Finds the foot of the altitude from p_vertex to the line defined by p_base1 and p_base2."""
        ap_x = p_vertex.x - p_base1.x
        ap_y = p_vertex.y - p_base1.y
        ab_x = p_base2.x - p_base1.x
        ab_y = p_base2.y - p_base1.y

        # Squared length of the base segment
        dot_ab_ab = ab_x**2 + ab_y**2
        if dot_ab_ab < 1e-9: # Base points are the same
            return p_base1
        
        # Project vector AP onto vector AB
        dot_ap_ab = ap_x * ab_x + ap_y * ab_y
        t = dot_ap_ab / dot_ab_ab
        
        # The foot is p_base1 + t * AB
        foot_x = p_base1.x + t * ab_x
        foot_y = p_base1.y + t * ab_y
        return Point(x=foot_x, y=foot_y)

    def create_puzzle(self) -> PointTargetPuzzleRecord:
        tries = 0
        min_side_length = self.canvas_short_side * 0.4
        while tries < 999:
            # Pick three vertices for the triangle
            p1, p2, p3 = (
                self.pick_target_point(0.8),
                self.pick_target_point(0.8),
                self.pick_target_point(0.8),
            )
            
            # Ensure the triangle is not too small or thin
            if self.distance(p1, p2) < min_side_length or \
               self.distance(p2, p3) < min_side_length or \
               self.distance(p3, p1) < min_side_length:
                tries += 1
                continue

            target_point = self._calculate_orthocenter(p1, p2, p3)
            self.target_point=target_point

            # Check if calculation was successful (non-collinear points)
            # and if the target point is within the canvas bounds.
            # The orthocenter can be outside the triangle for obtuse triangles.
            if target_point is None or not self.inside_canvas(target_point):
                tries += 1
                continue
            
            # Found a valid configuration
            self.triangle_points = (p1, p2, p3)
            self.place_candidates(target_point)
            record = self.save_puzzle()
            record.triangle_points = self.triangle_points
            return record
        
        raise RuntimeError("Failed to generate a valid orthocenter puzzle after many attempts.")

    def _render(self, highlight_label: Optional[str]) -> Image.Image:
        draw, base = self.get_draw_base()
        p1, p2, p3 = self.triangle_points
        
        # Draw the triangle by connecting its vertices
        self.draw_line(draw, [p1, p2, p3, p1])

        if highlight_label:
            # For the solution image, draw the altitudes
            f1 = self._get_altitude_foot(p1, p2, p3)
            self.draw_line(draw, [p1, f1])
            
            f2 = self._get_altitude_foot(p2, p1, p3)
            self.draw_line(draw, [p2, f2])

            f3 = self._get_altitude_foot(p3, p1, p2)
            self.draw_line(draw, [p3, f3])
            
            # possible obtuse angle
            for p in self.triangle_points:
                self.draw_line(draw, [self.target_point, p])

        self.draw_candidates(
            draw,
            highlight_label=highlight_label,
        )
        return base

def main(argv: Optional[List[str]] = None) -> None:
    OrthocenterGenerator.main(OrthocenterGenerator, argv)

if __name__ == "__main__":
    main()