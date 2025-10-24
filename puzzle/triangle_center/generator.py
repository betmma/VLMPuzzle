"""TriangleCenter puzzle generator.
"""

from __future__ import annotations
from typing import List, Optional, Sequence, Tuple
from PIL import Image, ImageDraw, ImageFont
import math
from ..point_target_base import PointTargetPuzzleGenerator, PointTargetPuzzleRecord, Point

class TriangleCenterGenerator(PointTargetPuzzleGenerator):
    """Generate puzzles to find the center of a triangle."""
    DEFAULT_OUTPUT_DIR="data/triangle_center"
    DEFAULT_PROMPT="Mark the center of the triangle red. Speak out which option is the center using phonetic alphabet. In portrait, no zoom, no pan."
    DEFAULT_GPT5_PROMPT="Which option is the center of the triangle? Answer an option in A-E."

    def create_puzzle(self) -> PointTargetPuzzleRecord:
        tries=0
        while tries<999:
            p1,p2,p3=self.pick_target_point(0.8), self.pick_target_point(0.8), self.pick_target_point(0.8)
            center=Point(
                x=(p1.x + p2.x + p3.x) / 3,
                y=(p1.y + p2.y + p3.y) / 3,
            )
            p12=Point(x=(p1.x + p2.x)/2, y=(p1.y + p2.y)/2)
            p23=Point(x=(p2.x + p3.x)/2, y=(p2.y + p3.y)/2)
            p31=Point(x=(p3.x + p1.x)/2, y=(p3.y + p1.y)/2)
            d1=math.sqrt((center.x - p12.x)**2 + (center.y - p12.y)**2)
            d2=math.sqrt((center.x - p23.x)**2 + (center.y - p23.y)**2)
            d3=math.sqrt((center.x - p31.x)**2 + (center.y - p31.y)**2)
            if min(d1,d2,d3)<self.canvas_short_side*0.3:
                tries+=1
                continue
            break
        self.triangle_points = (p1, p2, p3)
        self.midpoints = (p23, p31, p12)
        self.target_point = center
        self.place_candidates(center)
        record = self.save_puzzle()
        record.triangle_points=self.triangle_points
        return record

    def _render(self, highlight_label: Optional[str]) -> Image.Image:
        draw, base = self.get_draw_base()
        
        self.draw_line(draw,list(self.triangle_points)+[self.triangle_points[0]])
        if highlight_label:
            for vertex, midpoint in zip(self.triangle_points, self.midpoints):
                self.draw_line(draw,[vertex, midpoint])

        self.draw_candidates(
            draw,
            highlight_label=highlight_label,
        )
        return base

def main(argv: Optional[List[str]] = None) -> None:
    TriangleCenterGenerator.main(TriangleCenterGenerator, argv)

if __name__ == "__main__":
    main()
