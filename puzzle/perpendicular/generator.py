"""Perpendicular puzzle generator.
"""

from __future__ import annotations
from typing import List, Optional, Sequence, Tuple
from PIL import Image, ImageDraw, ImageFont
import math
from ..point_target_base import PointTargetPuzzleGenerator, PointTargetPuzzleRecord, Point

class PerpendicularGenerator(PointTargetPuzzleGenerator):
    """Generate puzzles to find the center of a triangle."""
    DEFAULT_OUTPUT_DIR="data/perpendicular"
    DEFAULT_PROMPT="Draw a black line perpendicular to the existing line and passing the small circle. Speak out which option is on the line using phonetic alphabet and mark that red. In portrait, no zoom, no pan."
    DEFAULT_GPT5_PROMPT="Which option is the center of the triangle? Answer an option in A-E."

    def create_puzzle(self) -> PointTargetPuzzleRecord:
        self.margin=50
        tries=0
        while tries<999:
            p1,p2=self.pick_target_point(0.8), self.pick_target_point(0.8)
            angle=math.atan2(p2.y - p1.y, p2.x - p1.x)+math.pi/2*self._rng.choice([-1,1])
            length=self.canvas_short_side*self._rng.uniform(0.3,0.6)
            target=Point(
                x=p1.x+length*math.cos(angle),
                y=p1.y+length*math.sin(angle),
            )
            distance=self.distance(p1,p2)
            if distance<self.canvas_short_side*0.3 or not self.inside_canvas(target):
                tries+=1
                continue
            break
        self.points = (p1, p2)
        self.target_point = target
        self.place_candidates_line(target,angle+math.pi/2)
        record = self.save_puzzle()
        record.points=self.points
        return record

    def _render(self, highlight_label: Optional[str]) -> Image.Image:
        draw, base = self.get_draw_base()
        
        self.draw_line(draw,self.points)
        self.draw_circle(draw,self.points[0],7)
        if highlight_label:
            self.draw_line(draw,[self.points[0],self.target_point])

        self.draw_candidates(
            draw,
            highlight_label=highlight_label,
        )
        return base

def main(argv: Optional[List[str]] = None) -> None:
    PerpendicularGenerator.main(PerpendicularGenerator, argv)

if __name__ == "__main__":
    main()
