"""Parallelogram puzzle generator.
"""

from __future__ import annotations
from typing import List, Optional, Sequence, Tuple
from PIL import Image, ImageDraw, ImageFont
import math
from ..point_target_base import PointTargetPuzzleGenerator, PointTargetPuzzleRecord, Point

class ParallelogramGenerator(PointTargetPuzzleGenerator):
    """Generate puzzles that hide the parallelogram of a segment."""
    DEFAULT_OUTPUT_DIR="data/parallelogram"
    DEFAULT_PROMPT="Draw a black parallelogram with two sides given. Mark the fourth vertex red. Speak out which option is the fourth vertex using phonetics alphabet. In portrait, no zoom, no pan."
    DEFAULT_GPT5_PROMPT="Which option is the fourth vertex of the parallelogram with two sides given? Answer an option in A-E."

    def create_puzzle(self) -> PointTargetPuzzleRecord:
        target_point = self.pick_target_point()
        tries=0
        distance_threshold = self.canvas_short_side * 0.3
        while tries<9999:
            p1, p2 = self.pick_target_point(0.5), self.pick_target_point(0.5)
            # Calculate the fourth point of the parallelogram
            p3 = Point(
                x=p1.x + p2.x - target_point.x,
                y=p1.y + p2.y - target_point.y,
            )
            angle1,angle2=math.atan2(p1.y-target_point.y,p1.x-target_point.x), math.atan2(p2.y-target_point.y,p2.x-target_point.x)
            if not self.inside_canvas(p3) or self.distance(target_point, p1) < distance_threshold or self.distance(target_point, p2) < distance_threshold or abs(math.sin(angle2-angle1))<0.5:
                tries+=1
                continue
            break
        self.parallelogram_points = (p1, p2, p3, target_point)
        self.place_candidates(target_point)
        record = self.save_puzzle()
        record.parallelogram_points=self.parallelogram_points
        return record

    def _render(self, highlight_label: Optional[str]) -> Image.Image:
        draw, base = self.get_draw_base()
        
        p1,p2,p3,target=self.parallelogram_points
        self.draw_line(draw,[p1,p3,p2])
        if highlight_label:
            self.draw_line(draw,[p2,target,p1],)

        self.draw_candidates(
            draw,
            highlight_label=highlight_label,
        )
        return base

def main(argv: Optional[List[str]] = None) -> None:
    ParallelogramGenerator.main(ParallelogramGenerator, argv)

if __name__ == "__main__":
    main()
