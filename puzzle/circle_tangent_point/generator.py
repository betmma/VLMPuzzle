from __future__ import annotations
from typing import List, Optional, Sequence, Tuple
from PIL import Image, ImageDraw, ImageFont
import math
from ..point_target_base import PointTargetPuzzleGenerator, PointTargetPuzzleRecord, Point, PointCandidate

class CircleTangentPointGenerator(PointTargetPuzzleGenerator):
    """Generate puzzles to find the point on a circle where the tangent line from an external point touches."""
    DEFAULT_OUTPUT_DIR="data/circle_tangent_point"
    DEFAULT_PROMPT="Draw the tangent line from the external point to the circle in black. Paint the point of tangency red. Speak out which option is the point using phonetic alphabet. In portrait, no zoom, no pan."
    DEFAULT_GPT5_PROMPT="Which option is the tangent point on the circle from the external point? Answer an option in A-E."

    def calculate_tangent_point(self, center: Point, R: float, external: Point) -> Point:
        """Calculates one of the two tangent points using geometric projection."""
        
        # 1. Distance D between center C and external point P
        D = self.distance(center, external)
        
        # 2. Distance d_CT along CP to the projection point K
        # d_CK = R^2 / D (from similar triangles in the right triangle CTP)
        d_CK = R**2 / D

        # 3. Calculate Point K (projection of T onto CP)
        # Unit vector u_CP
        ux = (external.x - center.x) / D
        uy = (external.y - center.y) / D
        
        Kx = center.x + d_CK * ux
        Ky = center.y + d_CK * uy

        # 4. Calculate distance h from K to T (height of right triangle)
        # h = sqrt(R^2 - d_CK^2)
        h = math.sqrt(R**2 - d_CK**2)
        
        # 5. Calculate T by moving perpendicularly from K by distance h
        # Perpendicular unit vector v (-uy, ux)
        vx = -uy
        vy = ux

        # We arbitrarily choose the 'upper' solution (K + h*v)
        Tx = Kx + h * vx
        Ty = Ky + h * vy
        
        return Point(Tx, Ty)

    def place_candidates_on_circle(self, center: Point, R: float, true_point: Point) -> None:
        """Generates candidates all lying on the circle, centered angularly around the true point."""
        
        labels = list(self.option_labels)
        correct_index = self._rng.randint(0, len(labels)-1)
        self.correct_label = labels[correct_index]
        
        # Find the angle of the true point
        theta_T = math.atan2(true_point.y - center.y, true_point.x - center.x)
        
        # Define a narrow angular spread (e.g., 40 degrees total) for challenge
        # The wider the spread, the easier the visual distinction.
        angular_spread = math.radians(self._rng.uniform(30.0, 55.0))*3
        
        # Calculate the angular separation between candidates
        angular_step = angular_spread / (len(labels) - 1)
        
        # Calculate the starting angle offset so the true point lands at the correct_index
        theta_start = theta_T - correct_index * angular_step
        
        candidates: List[PointCandidate] = []
        for i in range(len(labels)):
            theta = theta_start + i * angular_step
            
            # Calculate coordinates based on polar form
            cx = center.x + R * math.cos(theta)
            cy = center.y + R * math.sin(theta)
            
            p = Point(cx, cy)
            
            # If this is the correct index, ensure coordinates match the calculated target
            if i == correct_index:
                p = true_point

            candidates.append(PointCandidate(x=p.x, y=p.y, label=labels[i]))
            
        self.candidates = candidates

    def create_puzzle(self) -> PointTargetPuzzleRecord:
        
        min_R_ratio = 0.2
        max_R_ratio = 0.4
        min_dist_ratio = 1.5 # P must be at least 1.5 R from C
        max_dist_ratio = 3.0 # P must be at most 3.0 R from C

        tries=0
        while tries < 999:
            # 1. Define Circle
            R = self.canvas_short_side * self._rng.uniform(min_R_ratio, max_R_ratio)
            center = self.pick_target_point(R / self.canvas_short_side + 0.1)

            # 2. Define External Point P
            D_min = R * min_dist_ratio
            D_max = R * max_dist_ratio
            
            # Find a point P that is D distance away from C and inside the canvas
            dist_CP = self._rng.uniform(D_min, D_max)
            angle = self._rng.uniform(0, math.tau)
            
            external_point = Point(
                x = center.x + dist_CP * math.cos(angle),
                y = center.y + dist_CP * math.sin(angle)
            )

            # 3. Calculate Target Tangent Point T
            target_point = self.calculate_tangent_point(center, R, external_point)
            
            if not self.inside_canvas(external_point) or not self.inside_canvas(target_point):
                tries += 1
                continue
            

            self.center = center
            self.r = R
            self.external_point = external_point
            self.target_point = target_point
            
            # 4. Place candidates on the circle's circumference
            self.place_candidates_on_circle(center, R, target_point)
            for candidate in self.candidates:
                if candidate.label == self.correct_label:
                    continue
                dir1=math.atan2(center.y - candidate.y, center.x - candidate.x)
                dir2=math.atan2(external_point.y - candidate.y, external_point.x - candidate.x)
                if abs(math.cos(dir1 - dir2))<0.2:
                    break # Not tangent enough, retry
            else:
                # Success
                break
        
        record = self.save_puzzle()
        record.circle_center = self.center
        record.radius = self.r
        record.external_point = self.external_point
        return record

    def _render(self, highlight_label: Optional[str]) -> Image.Image:
        draw, base = self.get_draw_base()
        
        # Draw the main circle
        self.draw_circle(draw, self.center, self.r)
        
        # Draw the external point (a smaller solid dot)
        self.draw_circle(draw, self.external_point, 7)

        # Highlight the solution if rendering the answer image
        if highlight_label:
            # Draw the line segment from P to T (the tangent line fragment)
            self.draw_line(draw, [self.external_point, self.target_point])
            
            # Optionally, draw the radius CT to emphasize the perpendicular relationship
            self.draw_line(draw, [self.center, self.target_point])


        # Draw all candidate points
        self.draw_candidates(
            draw,
            highlight_label=highlight_label,
        )
        return base

def main(argv: Optional[List[str]] = None) -> None:
    CircleTangentPointGenerator.main(CircleTangentPointGenerator, argv)

if __name__ == "__main__":
    main()