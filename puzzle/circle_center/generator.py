"""CircleCenter puzzle generator.
"""

from __future__ import annotations
from typing import List, Optional, Sequence, Tuple
from PIL import Image, ImageDraw, ImageFont
from ..point_target_base import PointTargetPuzzleGenerator, PointTargetPuzzleRecord, Point

class CircleCenterGenerator(PointTargetPuzzleGenerator):
    """Generate puzzles to find the center of a circle."""
    DEFAULT_OUTPUT_DIR="data/circle_center"
    DEFAULT_PROMPT="Mark the center of the circle red. Speak out which option is the center using phonetic alphabet. In portrait, no zoom, no pan."
    DEFAULT_GPT5_PROMPT="Which option is the center of the circle? Answer an option in A-E."

    def create_puzzle(self) -> PointTargetPuzzleRecord:
        target_point = self.pick_target_point()
        self.target_point = target_point
        r_max = min(
            target_point.x,
            target_point.y,
            self.canvas_dimensions[0] - target_point.x,
            self.canvas_dimensions[1] - target_point.y,
        )
        r=(self._rng.random()*0.5+0.25)*r_max
        self.r=r
        self.place_candidates(target_point)
        record = self.save_puzzle()
        record.r=r
        return record

    def _render(self, highlight_label: Optional[str]) -> Image.Image:
        draw, base = self.get_draw_base()
        
        draw.ellipse(
            [
                (self.target_point.x - self.r, self.target_point.y - self.r),
                (self.target_point.x + self.r, self.target_point.y + self.r),
            ],
            outline="black",
            width=3,
        )

        self.draw_candidates(
            draw,
            highlight_label=highlight_label,
        )
        return base

def main(argv: Optional[List[str]] = None) -> None:
    CircleCenterGenerator.main(CircleCenterGenerator, argv)

if __name__ == "__main__":
    main()
