"""Square outlier puzzle generator.
"""

from __future__ import annotations
from typing import List, Optional, Sequence, Tuple
from PIL import Image, ImageDraw, ImageFont
import math
from ..point_target_base import PointTargetPuzzleGenerator, PointTargetPuzzleRecord, Point, PointCandidate

class SquareOutlierGenerator(PointTargetPuzzleGenerator):
    """Generate puzzles where four of five points form a square."""
    DEFAULT_OUTPUT_DIR="data/square_outlier"
    DEFAULT_PROMPT="Four of the five options form a square. Mark the fifth point red. Speak out which option is the fifth point using phonetics alphabet. In portrait, no zoom, no pan."
    DEFAULT_GPT5_PROMPT="Four of the five options form a square. Which option is the fifth point? Answer an option in A-E."

    def create_puzzle(self) -> PointTargetPuzzleRecord:
        """
        Creates a puzzle with five points, where four form a square and one is an outlier.
        The outlier is the target.
        """
        tries = 0
        while tries < 999:
            # 1. Generate a square with a random size, rotation, and position.
            
            # The side length of the square.
            side_length = self._rng.uniform(self.canvas_short_side * 0.3, self.canvas_short_side * 0.6)
            half_side = side_length / 2
            
            center = self.pick_target_point(0.8)
            
            # The rotation angle of the square.
            angle = self._rng.uniform(0, 2 * math.pi)
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)
            
            # Vertices of a non-rotated square centered at the origin.
            unrotated_corners = [
                Point(half_side, half_side),
                Point(-half_side, half_side),
                Point(-half_side, -half_side),
                Point(half_side, -half_side),
            ]
            
            square_points = []
            valid_square = True
            for p in unrotated_corners:
                # Apply rotation and then translation to get final vertex coordinates.
                rotated_x = p.x * cos_a - p.y * sin_a
                rotated_y = p.x * sin_a + p.y * cos_a
                final_point = Point(center.x + rotated_x, center.y + rotated_y)
                
                if not self.inside_canvas(final_point):
                    valid_square = False
                    break
                square_points.append(final_point)
            
            if not valid_square:
                tries += 1
                continue

            # 2. Generate the outlier point (the target).
            target_point = None
            outlier_tries = 0
            while outlier_tries < 100:
                potential_outlier = self.pick_target_point()
                
                # Ensure the outlier is not too close to any of the square's vertices.
                min_dist = min(self.distance(potential_outlier, v) for v in square_points)
                if min_dist > side_length * 0.3:
                    target_point = potential_outlier
                    break
                outlier_tries += 1
            
            if target_point is None: # Failed to place an outlier
                tries += 1
                continue

            # A valid configuration has been found.
            self.target_point = target_point
            self.square_points = square_points
            break
        
        if tries >= 999:
             raise RuntimeError("Failed to generate a valid square outlier puzzle.")

        # 3. Manually create the list of candidates.
        # This is required because place_candidates would place distractors
        # according to some rule, but here our distractors are part of the core puzzle.
        all_points = self.square_points + [self.target_point]
        self._rng.shuffle(all_points)
        
        labels = ['A', 'B', 'C', 'D', 'E']
        self.candidates = []
        for i, p in enumerate(all_points):
            label = labels[i]
            self.candidates.append(PointCandidate(x=p.x, y=p.y, label=label))
            if p == self.target_point:
                self.correct_label = label

        # 4. Save the puzzle data and return the record.
        record = self.save_puzzle()
        record.square_points = self.square_points
        return record

    def _render(self, highlight_label: Optional[str]) -> Image.Image:
        """
        Renders the puzzle image. For the solution, it draws the square.
        """
        draw, base = self.get_draw_base()
        
        # For the solution image, draw the square formed by the four non-outlier points.
        if highlight_label:
            # The points are already ordered. Add the first point to the end to close the shape.
            points_to_draw = self.square_points + [self.square_points[0]]
            self.draw_line(draw, points_to_draw)

        # Draw all five candidate points and their labels.
        self.draw_candidates(
            draw,
            highlight_label=highlight_label,
        )
        return base

def main(argv: Optional[List[str]] = None) -> None:
    SquareOutlierGenerator.main(SquareOutlierGenerator, argv)

if __name__ == "__main__":
    main()