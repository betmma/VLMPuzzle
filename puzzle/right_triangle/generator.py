"""Right angle triangle puzzle generator.
"""

from __future__ import annotations
from typing import List, Optional, Sequence, Tuple
from PIL import Image, ImageDraw, ImageFont
import math
import itertools
from ..point_target_base import PointTargetPuzzleGenerator, PointTargetPuzzleRecord, Point, PointCandidate

class RightTriangleGenerator(PointTargetPuzzleGenerator):
    """
    Generate puzzles where 5 points are shown, but only 3 form a right-angled triangle.
    The goal is to identify the vertex with the right angle.
    """
    DEFAULT_OUTPUT_DIR = "data/right_triangle"
    DEFAULT_PROMPT = "Out of the 5 points, 3 form a right-angled triangle. Mark the vertex with the right angle in red. Speak out which option is the right-angle vertex using phonetic alphabet. In portrait, no zoom, no pan."
    DEFAULT_GPT5_PROMPT = "Which option is the vertex of the right angle, given that exactly three of the five options form a right-angled triangle? Answer an option in A-E."

    def _cos_angle(self, p_a: Point, p_b: Point, p_c: Point) -> float:
        """Calculates the cosine of the angle at vertex p_b for triangle p_a, p_b, p_c."""
        v_ba = Point(x=p_a.x - p_b.x, y=p_a.y - p_b.y)
        v_bc = Point(x=p_c.x - p_b.x, y=p_c.y - p_b.y)

        dot_product = v_ba.x * v_bc.x + v_ba.y * v_bc.y
        mag_ba = math.sqrt(v_ba.x**2 + v_ba.y**2)
        mag_bc = math.sqrt(v_bc.x**2 + v_bc.y**2)

        if mag_ba == 0 or mag_bc == 0:
            return 1.0

        return dot_product / (mag_ba * mag_bc)

    def create_puzzle(self) -> PointTargetPuzzleRecord:
        min_side_len = self.canvas_short_side * 0.2
        max_side_len = self.canvas_short_side * 0.8
        min_dist_between_points = self.canvas_short_side * 0.1
        ambiguity_threshold = 0.2

        tries = 0
        while tries < 999999:
            # 1. Generate the right-angled triangle
            p_right = self.pick_target_point(0.7)
            
            angle = self._rng.uniform(0, 2 * math.pi)
            len1 = self._rng.uniform(min_side_len, max_side_len)
            len2 = self._rng.uniform(min_side_len, max_side_len)

            p1 = Point(
                x=p_right.x + len1 * math.cos(angle),
                y=p_right.y + len1 * math.sin(angle),
            )
            p2 = Point(
                x=p_right.x + len2 * math.cos(angle + math.pi / 2),
                y=p_right.y + len2 * math.sin(angle + math.pi / 2),
            )
            
            right_triangle_pts = [p_right, p1, p2]
            
            if not all(self.inside_canvas(p) for p in right_triangle_pts):
                tries += 1
                continue

            # 2. Generate 2 distractor points, ensuring they are not too close to other points
            all_points = list(right_triangle_pts)
            distractors_ok = True
            for _ in range(2):
                for _d_try in range(100):
                    d = self.pick_target_point()
                    if all(self.distance(d, p) > min_dist_between_points for p in all_points):
                        all_points.append(d)
                        break
                else:
                    distractors_ok = False
                    break
            
            if not distractors_ok:
                tries += 1
                continue

            # 3. Check for uniqueness of the right angle
            is_ambiguous = False
            
            for combo in itertools.combinations(all_points, 3):
                p_a, p_b, p_c = combo
                
                if (p_a in right_triangle_pts and p_b in right_triangle_pts and p_c in right_triangle_pts):
                    continue
                
                cosines = [
                    abs(self._cos_angle(p_b, p_a, p_c)),
                    abs(self._cos_angle(p_a, p_b, p_c)),
                    abs(self._cos_angle(p_b, p_c, p_a)),
                ]
                
                if any(c < ambiguity_threshold for c in cosines):
                    is_ambiguous = True
                    break
            
            if is_ambiguous:
                tries += 1
                continue
            
            # Success!
            self.target_point = p_right
            self.right_triangle_points = (p_right, p1, p2)
            
            # 4. Manually create candidates and set the correct label
            self._rng.shuffle(all_points)
            self.candidates = []
            labels = "ABCDE"
            for i, p in enumerate(all_points):
                label = labels[i]
                self.candidates.append(PointCandidate(x=p.x, y=p.y, label=label))
                if p == self.target_point:
                    self.correct_label = label
            
            record = self.save_puzzle()
            record.right_triangle_points = self.right_triangle_points
            return record

        raise RuntimeError("Failed to generate a valid puzzle after many tries.")

    def _render(self, highlight_label: Optional[str]) -> Image.Image:
        draw, base = self.get_draw_base()
        
        self.draw_candidates(
            draw,
            highlight_label=highlight_label,
        )

        if highlight_label:
            p_right, p1, p2 = self.right_triangle_points
            self.draw_line(draw, [p1, p_right, p2, p1])

        return base

def main(argv: Optional[List[str]] = None) -> None:
    RightTriangleGenerator.main(RightTriangleGenerator, argv)

if __name__ == "__main__":
    main()