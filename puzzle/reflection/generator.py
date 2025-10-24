"""Reflection puzzle generator.
"""

from __future__ import annotations
from typing import List, Optional, Sequence, Tuple
from PIL import Image, ImageDraw, ImageFont
import math
from ..point_target_base import PointTargetPuzzleGenerator, PointTargetPuzzleRecord, Point

class ReflectionGenerator(PointTargetPuzzleGenerator):
    """Generate puzzles to find the reflection of a point across a line."""
    DEFAULT_OUTPUT_DIR="data/reflection"
    DEFAULT_PROMPT="Reflect the small circle across the line. Mark the reflection red and speak out which option is the reflected point using phonetic alphabet. In portrait, no zoom, no pan."
    DEFAULT_GPT5_PROMPT="Which option is the reflection of the small circle across the line? Answer an option in A-E."

    def create_puzzle(self) -> PointTargetPuzzleRecord:
        """
        Creates a puzzle by defining a line and a point, then calculating the
        reflection of that point across the line.
        """
        tries = 0
        min_line_dist = self.canvas_short_side * 0.4
        min_point_dist = self.canvas_short_side * 0.2

        while tries < 999:
            # 1. Define the line with two points, ensuring it's reasonably long.
            p1, p2 = self.pick_target_point(0.8), self.pick_target_point(0.8)
            if self.distance(p1, p2) < min_line_dist:
                tries += 1
                continue

            # 2. Define the source point to be reflected.
            source_point = self.pick_target_point(0.8)

            # 3. Calculate the reflection using vector projection.
            # Vector for the line (from p1 to p2)
            v_x, v_y = p2.x - p1.x, p2.y - p1.y
            # Vector from p1 to the source point
            w_x, w_y = source_point.x - p1.x, source_point.y - p1.y
            
            # Dot product of v with itself (squared magnitude)
            dot_vv = v_x * v_x + v_y * v_y
            if dot_vv == 0: # Avoid division by zero if p1 and p2 are the same point
                tries += 1
                continue
            
            # Dot product of w with v
            dot_wv = w_x * v_x + w_y * v_y
            
            # The projection factor 't' determines the closest point on the line.
            t = dot_wv / dot_vv

            # q is the projection of source_point onto the line p1-p2.
            q = Point(x=p1.x + t * v_x, y=p1.y + t * v_y)
            
            # Ensure the source point is not too close to the line.
            if self.distance(source_point, q) < min_point_dist:
                tries += 1
                continue
                
            # The target point is the reflection. The vector from source to target
            # is twice the vector from source to its projection 'q'.
            target_point = Point(
                x=source_point.x + 2 * (q.x - source_point.x),
                y=source_point.y + 2 * (q.y - source_point.y),
            )

            # 4. Ensure the target point is within the canvas bounds.
            if not self.inside_canvas(target_point):
                tries += 1
                continue

            # A valid puzzle configuration has been found.
            break
        
        # Store the geometric elements for rendering.
        self.line_points = (p1, p2)
        self.source_point = source_point
        self.target_point = target_point
        
        self.place_candidates(target_point)
        record = self.save_puzzle()
        record.line_points = self.line_points
        record.source_point = self.source_point
        return record

    def _render(self, highlight_label: Optional[str]) -> Image.Image:
        """Renders the reflection puzzle on an image canvas."""
        draw, base = self.get_draw_base()
        
        # Draw the line of reflection.
        self.draw_line(draw, self.line_points)
        
        # Draw the source point to be reflected.
        self.draw_circle(draw, self.source_point, 7)
        
        # If rendering the solution, draw the line connecting the source to its reflection.
        if highlight_label:
            self.draw_line(draw, [self.source_point, self.target_point])

        # Draw the multiple-choice candidate points.
        self.draw_candidates(
            draw,
            highlight_label=highlight_label,
        )
        return base

def main(argv: Optional[List[str]] = None) -> None:
    ReflectionGenerator.main(ReflectionGenerator, argv)

if __name__ == "__main__":
    main()