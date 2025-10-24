"""Incenter puzzle generator.
"""

from __future__ import annotations
from typing import List, Optional, Sequence, Tuple
from PIL import Image, ImageDraw, ImageFont
import math
from ..point_target_base import PointTargetPuzzleGenerator, PointTargetPuzzleRecord, Point


def distance(p1: Point, p2: Point) -> float:
    """Calculates the Euclidean distance between two points."""
    return math.sqrt((p2.x - p1.x)**2 + (p2.y - p1.y)**2)

def calculate_incenter_and_inradius(p1: Point, p2: Point, p3: Point) -> Tuple[Point, float]:
    # 1. Calculate the lengths of the sides of the triangle
    # a is the length of the side opposite p1
    a = distance(p2, p3)
    # b is the length of the side opposite p2
    b = distance(p1, p3)
    # c is the length of the side opposite p3
    c = distance(p1, p2)

    # 2. Check for collinearity (degenerate triangle)
    # If one side is greater than or equal to the sum of the other two,
    # the points are in a line. We use a small tolerance for floating point errors.
    epsilon = 1e-9
    if a + b <= c + epsilon or a + c <= b + epsilon or b + c <= a + epsilon:
        return Point(0, 0),0

    # 3. Calculate the perimeter and semi-perimeter
    perimeter = a + b + c
    s = perimeter / 2

    # 4. Calculate the incenter coordinates
    incenter_x = (a * p1.x + b * p2.x + c * p3.x) / perimeter
    incenter_y = (a * p1.y + b * p2.y + c * p3.y) / perimeter
    incenter = Point(incenter_x, incenter_y)

    # 5. Calculate the area of the triangle using Heron's formula
    area = math.sqrt(s * (s - a) * (s - b) * (s - c))

    # 6. Calculate the inradius
    inradius = area / s
    
    return incenter,inradius

class IncenterGenerator(PointTargetPuzzleGenerator):
    """Generate puzzles to find the incenter of a triangle."""
    DEFAULT_OUTPUT_DIR="data/incenter"
    DEFAULT_PROMPT="Mark the incenter of the triangle red. Speak out which option is the incenter using phonetic alphabet. In portrait, no zoom, no pan."
    DEFAULT_GPT5_PROMPT="Which option is the incenter of the triangle? Answer an option in A-E."

    def create_puzzle(self) -> PointTargetPuzzleRecord:
        tries=0
        while tries<999:
            p1,p2,p3=self.pick_target_point(0.8), self.pick_target_point(0.8), self.pick_target_point(0.8)
            incenter,r=calculate_incenter_and_inradius(p1,p2,p3)
            if r<self.canvas_short_side*0.1:
                tries+=1
                continue
            break
        self.triangle_points = (p1, p2, p3)
        self.target_point = incenter
        self.r = r
        self.place_candidates(incenter)
        record = self.save_puzzle()
        record.triangle_points=self.triangle_points
        record.r=self.r
        return record

    def _render(self, highlight_label: Optional[str]) -> Image.Image:
        draw, base = self.get_draw_base()
        
        self.draw_line(draw,list(self.triangle_points)+[self.triangle_points[0]])
        if highlight_label:
            self.draw_circle(draw,self.target_point,self.r)

        self.draw_candidates(
            draw,
            highlight_label=highlight_label,
        )
        return base

def main(argv: Optional[List[str]] = None) -> None:
    IncenterGenerator.main(IncenterGenerator, argv)

if __name__ == "__main__":
    main()
