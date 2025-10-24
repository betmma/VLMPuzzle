"""AngleBisector puzzle generator.
"""

from __future__ import annotations
from typing import List, Optional, Sequence, Tuple
from PIL import Image, ImageDraw, ImageFont
import math
from ..point_target_base import PointTargetPuzzleGenerator, PointTargetPuzzleRecord, Point

def mid_angle(angle1: float, angle2: float) -> float:
    """Calculate the bisector angle between two angles."""
    x = math.cos(angle1) + math.cos(angle2)
    y = math.sin(angle1) + math.sin(angle2)
    return math.atan2(y, x)

class AngleBisectorGenerator(PointTargetPuzzleGenerator):
    """Generate puzzles to find the center of a triangle."""
    DEFAULT_OUTPUT_DIR="data/angle_bisector"
    DEFAULT_PROMPT="Draw a black line bisecting the angle. Speak out which option is on the bisector using phonetic alphabet and mark that red. In portrait, no zoom, no pan."
    DEFAULT_GPT5_PROMPT="Which option is on the bisector of the angle? Answer an option in A-E."

    def create_puzzle(self) -> PointTargetPuzzleRecord:
        self.margin=50
        tries=0
        distance_threshold = self.canvas_short_side * 0.3
        while tries<999:
            p0, p1, p2 = self.pick_target_point(0.7), self.pick_target_point(0.7), self.pick_target_point(0.7)
            angle1,angle2=math.atan2(p1.y-p0.y,p1.x-p0.x), math.atan2(p2.y-p0.y,p2.x-p0.x)
            mid_angle_value=mid_angle(angle1,angle2)
            length=self.canvas_short_side*self._rng.uniform(0.5,1.0)
            target=Point(
                x=p0.x+length*math.cos(mid_angle_value),
                y=p0.y+length*math.sin(mid_angle_value),
            )
            if not self.inside_canvas(target) or self.distance(p0, p1) < distance_threshold or self.distance(p0, p2) < distance_threshold or abs(math.sin(angle2-angle1))<0.3:
                tries+=1
                continue
            break
        self.points = (p2, p0, p1)
        self.target_point = target
        self.place_candidates_line(target,mid_angle_value+math.pi/2+self._rng.uniform(-0.1,0.1))
        record = self.save_puzzle()
        record.points=self.points
        return record

    def _render(self, highlight_label: Optional[str]) -> Image.Image:
        draw, base = self.get_draw_base()
        
        self.draw_line(draw,self.points)
        if highlight_label:
            self.draw_line(draw,[self.points[1],self.target_point])

        self.draw_candidates(
            draw,
            highlight_label=highlight_label,
        )
        return base

def main(argv: Optional[List[str]] = None) -> None:
    AngleBisectorGenerator.main(AngleBisectorGenerator, argv)

if __name__ == "__main__":
    main()
