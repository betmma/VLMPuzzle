"""Circle Tangent Line puzzle generator.
"""

from __future__ import annotations
from typing import List, Optional, Sequence, Tuple
from PIL import Image, ImageDraw, ImageFont
import math
from ..point_target_base import PointTargetPuzzleGenerator, PointTargetPuzzleRecord, Point

class CircleTangentLineGenerator(PointTargetPuzzleGenerator):
    """Generate puzzles that require identifying a line tangent to a circle."""
    DEFAULT_OUTPUT_DIR="data/circle_tangent_line"
    DEFAULT_PROMPT="Draw a black line tangent to the circle at the highlighted point. Speak out which option lies on this tangent line in phonetic alphabet and mark that red. In portrait, no zoom, no pan."
    DEFAULT_GPT5_PROMPT="Which option lies on the line that is tangent to the circle at the highlighted point? Answer an option in A-E."

    def create_puzzle(self) -> PointTargetPuzzleRecord:
        tries=0
        while tries<999:
            # 1. Define the circle's center and radius
            center = self.pick_target_point(0.6) # Keep center away from edges
            min_radius = self.canvas_short_side * 0.2
            max_radius = self.canvas_short_side * 0.4
            radius = self._rng.uniform(min_radius, max_radius)

            # 2. Define the point on the circle's circumference
            radius_angle = self._rng.uniform(0, math.tau)
            point_on_circle = Point(
                x=center.x + radius * math.cos(radius_angle),
                y=center.y + radius * math.sin(radius_angle)
            )

            # 3. The tangent line is perpendicular to the radius
            tangent_angle = radius_angle + math.pi / 2

            # 4. Define the target point on the tangent line
            # Choose a distance from the point on the circle to the target
            dist_from_poc = self._rng.uniform(0.2, 0.4) * self.canvas_short_side
            # Randomly pick a direction along the tangent line
            dist_from_poc *= self._rng.choice([-1, 1])

            target_point = Point(
                x=point_on_circle.x + dist_from_poc * math.cos(tangent_angle),
                y=point_on_circle.y + dist_from_poc * math.sin(tangent_angle)
            )

            self.circle_center = center
            self.circle_radius = radius
            self.point_on_circle = point_on_circle
            self.target_point = target_point
            
            self.place_candidates_line(target_point, tangent_angle+math.pi/2+self._rng.uniform(-0.1,0.1))
            # 5. Ensure the target point is within the canvas
            if not self.check_candidates_inside():
                tries += 1
                continue
            
            # If everything is valid, break the loop
            break

        
        record = self.save_puzzle()
        record.circle_center = self.circle_center
        record.circle_radius = self.circle_radius
        record.point_on_circle = self.point_on_circle
        return record

    def _render(self, highlight_label: Optional[str]) -> Image.Image:
        draw, base = self.get_draw_base()
        
        # Draw the main circle
        self.draw_circle(draw, self.circle_center, self.circle_radius)
        # Highlight the point on the circumference where the tangent touches
        self.draw_circle(draw, self.point_on_circle, 7)

        # In the solution image, draw the tangent line segment
        if highlight_label:
            self.draw_line(draw, [self.point_on_circle, self.target_point])

        self.draw_candidates(
            draw,
            highlight_label=highlight_label,
        )
        return base

def main(argv: Optional[List[str]] = None) -> None:
    CircleTangentLineGenerator.main(CircleTangentLineGenerator, argv)

if __name__ == "__main__":
    main()