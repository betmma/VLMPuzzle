"""Ray reflection puzzle generator.
"""

from __future__ import annotations
from typing import List, Optional, Sequence, Tuple
from PIL import Image, ImageDraw, ImageFont
import math
from ..point_target_base import PointTargetPuzzleGenerator, PointTargetPuzzleRecord, Point

class RayReflectGenerator(PointTargetPuzzleGenerator):
    """Generate puzzles where a ray reflects off a line."""
    DEFAULT_OUTPUT_DIR="data/ray_reflect"
    DEFAULT_PROMPT="Draw the ray of light starting from the small circle and reflecting off the line in black. Speak out which option the reflected ray will pass through using phonetic alphabet and mark it red. In portrait, no zoom, no pan."
    DEFAULT_GPT5_PROMPT="A ray of light starts from the small circle and reflects off the line. Which option will the reflected ray pass through? Answer an option in A-E."

    def _reflect_point(self, point_to_reflect: Point, line_p1: Point, line_p2: Point) -> Point:
        """Helper to reflect a point across a line defined by two other points."""
        dx = line_p2.x - line_p1.x
        dy = line_p2.y - line_p1.y
        
        line_len_sq = dx * dx + dy * dy
        # Handle degenerate case where the line is just a point
        if line_len_sq < 1e-9:
            return point_to_reflect
        
        # Project the vector (point_to_reflect - line_p1) onto the line vector (line_p2 - line_p1)
        # t is the projection factor
        t = ((point_to_reflect.x - line_p1.x) * dx + (point_to_reflect.y - line_p1.y) * dy) / line_len_sq
        
        # Find the closest point on the line to point_to_reflect
        closest_point = Point(x=line_p1.x + t * dx, y=line_p1.y + t * dy)

        # The reflected point is P' = P + 2*(Closest - P) = 2*Closest - P
        reflected_point = Point(
            x=2 * closest_point.x - point_to_reflect.x,
            y=2 * closest_point.y - point_to_reflect.y
        )
        return reflected_point

    def create_puzzle(self) -> PointTargetPuzzleRecord:
        self.margin = 50
        tries = 0
        while tries < 999:
            tries += 1
            
            # 1. Define the mirror line segment
            m1 = self.pick_target_point(0.7)
            m2 = self.pick_target_point(0.7)
            if self.distance(m1, m2) < self.canvas_short_side * 0.4:
                continue

            # 2. Define the light source
            source = self.pick_target_point(0.9)

            # 3. Ensure the source is not too close to the mirror line
            dx, dy = m2.x - m1.x, m2.y - m1.y
            dist_from_line = abs(dx * (source.y - m1.y) - dy * (source.x - m1.x)) / math.sqrt(dx*dx + dy*dy)
            if dist_from_line < self.canvas_short_side * 0.2:
                continue

            # 4. Pick a point on the mirror segment for the reflection to occur
            reflect_ratio = self._rng.uniform(0.2, 0.8)  # Avoid the very ends of the mirror
            p_mirror = Point(
                x=m1.x + reflect_ratio * (m2.x - m1.x),
                y=m1.y + reflect_ratio * (m2.y - m1.y),
            )

            # 5. Check the angle of incidence to avoid very shallow angles
            v_in = (p_mirror.x - source.x, p_mirror.y - source.y)
            v_mirror = (m2.x - m1.x, m2.y - m1.y)
            dot_product = v_in[0]*v_mirror[0] + v_in[1]*v_mirror[1]
            mag_in = math.sqrt(v_in[0]**2 + v_in[1]**2)
            mag_mirror = math.sqrt(v_mirror[0]**2 + v_mirror[1]**2)
            if mag_in < 1e-6 or mag_mirror < 1e-6: continue
            cos_angle = abs(dot_product / (mag_in * mag_mirror))
            if cos_angle > 0.95:  # Angle is < ~18 degrees, too shallow
                continue

            # 6. The key insight: the reflected ray appears to come from the reflection of the source.
            s_reflected = self._reflect_point(source, m1, m2)

            # 7. The reflected ray is the line from p_mirror through s_reflected. Find a target on it.
            ray_dir_x = s_reflected.x - p_mirror.x
            ray_dir_y = s_reflected.y - p_mirror.y
            ray_len = math.sqrt(ray_dir_x**2 + ray_dir_y**2)
            if ray_len < 1e-6: continue
            
            # Extend the ray to find a target point within the canvas
            target_dist = self.canvas_short_side * self._rng.uniform(0.4, 0.9)
            target = Point(
                x=p_mirror.x - target_dist * (ray_dir_x / ray_len),
                y=p_mirror.y - target_dist * (ray_dir_y / ray_len),
            )
            
            # 8. Final validation
            if not self.inside_canvas(target):
                continue
            
            # We have a valid puzzle
            break

        # Store the geometric points for rendering
        self.points = (source, p_mirror, m1, m2)
        self.target_point = target
        
        # The candidates lie along the reflected ray
        line_angle = math.atan2(target.y - p_mirror.y, target.x - p_mirror.x)
        self.place_candidates_line(target, line_angle+math.pi/2)

        record = self.save_puzzle()
        record.points = self.points
        return record

    def _render(self, highlight_label: Optional[str]) -> Image.Image:
        draw, base = self.get_draw_base()
        
        source, p_mirror, m1, m2 = self.points
        
        # Draw the mirror line
        self.draw_line(draw, [m1, m2])
        
        # Draw the source point and the incoming ray
        self.draw_circle(draw, source, 7)

        # If rendering the solution, draw the reflected ray
        if highlight_label:
            self.draw_line(draw, [source, p_mirror],0.6)
            self.draw_line(draw, [p_mirror, self.target_point],0.6)
        else:
            half=Point(
                x=(source.x + p_mirror.x)/2,
                y=(source.y + p_mirror.y)/2,
            )
            self.draw_line(draw, [source, half],0.6)
            

        self.draw_candidates(
            draw,
            highlight_label=highlight_label,
        )
        return base

def main(argv: Optional[List[str]] = None) -> None:
    RayReflectGenerator.main(RayReflectGenerator, argv)

if __name__ == "__main__":
    main()