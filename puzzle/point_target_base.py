"""Shared scaffolding for point-target option puzzles.

These puzzles feature a hidden or implicit key point on the canvas and a fixed
set of labeled candidate markers positioned nearby. Solvers indicate the
correct marker by speaking, writing, or highlighting it in red. Generators and
Evaluators implementing this pattern can derive from the classes here to reuse
candidate placement and scoring logic.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Generic, Iterable, List, Optional, Sequence, Tuple
import argparse
import json
import uuid

import cv2
import numpy as np

from .base import AbstractPuzzleEvaluator, AbstractPuzzleGenerator, PathLike

from PIL import Image, ImageFont, ImageDraw


@dataclass
class Point:
    x: float
    y: float

    def to_list(self) -> List[float]:
        return [self.x, self.y]

@dataclass
class PointCandidate:
    """Serializable representation of a labeled candidate point."""

    x: float
    y: float
    label: str

    def to_dict(self) -> Dict[str, object]:
        return {"x": self.x, "y": self.y, "label": self.label}

@dataclass
class PointTargetPuzzleRecord:
    """Base record fields for point-target puzzles."""

    id: str
    prompt: str
    canvas_dimensions: Tuple[int, int]
    margin: int
    candidates: List[PointCandidate]
    correct_option: str
    image: str
    solution_image_path: str

class PointTargetPuzzleGenerator(AbstractPuzzleGenerator):
    """Base generator providing canvas configuration and candidate placement."""

    POINT_RADIUS: int = 10
    LINE_WIDTH: int = 5
    CANDIDATE_OUTLINE_COLOR: Tuple[int, int, int] = (32, 32, 32)
    CANDIDATE_HIGHLIGHT_COLOR: Tuple[int, int, int] = (198, 24, 24)
    CANDIDATE_TEXT_COLOR: Tuple[int, int, int] = (0, 0, 0)
    CANDIDATE_BASE_FILL: Tuple[int, int, int] = (255, 255, 255)
    CANDIDATE_HIGHLIGHT_FILL: Tuple[int, int, int] = (255, 220, 220)
    CANDIDATE_OUTLINE_WIDTH: int = 4
    CANDIDATE_HIGHLIGHT_OUTLINE_WIDTH: int = 4
    CANDIDATE_LABEL_OFFSET_Y: int = 0
    MAX_VIDEO_FRAMES: int = 193
    DEFAULT_OUTPUT_DIR: str = None
    DEFAULT_PROMPT: str = None
    DEFAULT_GPT5_PROMPT: str = None

    def __init__(
        self,
        output_dir: PathLike,
        *,
        canvas_width: int = 480,
        aspect: Optional[float] = None,
        seed: Optional[int] = None,
        prompt: Optional[str] = None,
        option_labels: Sequence[str] = ("A", "B", "C", "D", "E"),
        margin_ratio: float = 0.06,
        record_video: bool = False,
    ) -> None:
        output_dir = output_dir if output_dir is not None else Path(self.DEFAULT_OUTPUT_DIR)
        prompt = prompt if prompt is not None else self.DEFAULT_PROMPT
        super().__init__(output_dir)
        width = int(canvas_width)
        if width <= 0:
            raise ValueError("canvas_width must be positive")
        if aspect and aspect > 0:
            height = round(width / float(aspect))
        else:
            height = width
        if height <= 0:
            raise ValueError("Derived canvas height must be positive")
        self.canvas_dimensions = (width, height)
        margin_base = min(width, height)
        computed_margin = round(margin_base * max(0.0, margin_ratio))
        self.margin = max(18, computed_margin)
        self._rng = random.Random(seed)
        if not option_labels:
            raise ValueError("option_labels must contain at least one label")
        self.option_labels = tuple(option_labels)
        self.prompt = prompt
        self._candidate_font: Optional[Any] = None
        self.point_radius = int(self.POINT_RADIUS)
        out_root = Path(self.output_dir)
        self.puzzle_dir = out_root / "puzzles"
        self.solution_dir = out_root / "solutions"
        self.puzzle_dir.mkdir(parents=True, exist_ok=True)
        self.solution_dir.mkdir(parents=True, exist_ok=True)
        self.record_video = record_video
        self._recording_active = False
        self._recorder: Optional[DrawingRecorder] = None

    @property
    def rng(self) -> random.Random:
        return self._rng

    def canvas_bounds(self) -> Tuple[int, int, int, int]:
        width, height = self.canvas_dimensions
        left = self.margin
        top = self.margin
        right = width - self.margin
        bottom = height - self.margin
        return left, top, right, bottom
    
    def inside_canvas(
        self,
        point: Point,
    ) -> bool:
        x, y = point.to_list()
        left, top, right, bottom = self.canvas_bounds()
        return (left <= x <= right) and (top <= y <= bottom)
    
    def distance(
        self,
        p1: Point,
        p2: Point,
    ) -> float:
        return math.hypot(p1.x - p2.x, p1.y - p2.y)
    
    @property
    def canvas_short_side(self) -> int:
        width, height = self.canvas_dimensions
        return min(width, height)

    def pick_target_point(
        self,
        jitter_ratio: float = 0.36,
    ) -> Point:
        jitter_ratio/=2 # jitter_ratio = 1 means full spread across the canvas
        left, top, right, bottom = self.canvas_bounds()
        width, height = right - left, bottom - top
        center_x = left + width * 0.5
        center_y = top + height * 0.5
        jitter_x = self._rng.uniform(-jitter_ratio * width, jitter_ratio * width)
        jitter_y = self._rng.uniform(-jitter_ratio * height, jitter_ratio * height)
        x = center_x + jitter_x
        y = center_y + jitter_y
        return Point(x, y)
    
    def place_candidates_line(self,true_point: Point,angle:float|None=None)->None:
        radius = self.point_radius
        base_x, base_y = true_point.x, true_point.y
        labels = list(self.option_labels)
        correct_index = self._rng.randint(0, len(labels)-1)
        correct_label = labels[correct_index]
        candidates: List[PointCandidate] = []
        target_count = len(labels)
        spread = max(18.0, 0.9 * radius)*2
        if angle is None:
            angle = self._rng.uniform(0.0, math.tau)
        dx,dy=math.cos(angle)*spread, math.sin(angle)*spread
        for i in range(target_count):
            cx = base_x + dx*(i-correct_index)
            cy = base_y + dy*(i-correct_index)
            label = labels[i]
            candidates.append(PointCandidate(x=cx, y=cy, label=label))
        self.candidates, self.correct_label= candidates, correct_label

    def check_candidates_inside(self)->bool:
        for candidate in self.candidates:
            point=Point(candidate.x,candidate.y)
            if not self.inside_canvas(point):
                return False
        return True
    
    def place_candidates(
        self,
        true_point: Point,
    ) -> None:
        radius = self.point_radius
        left, top, right, bottom = self.canvas_bounds()
        base_x, base_y = true_point.x, true_point.y
        labels = list(self.option_labels)
        self._rng.shuffle(labels)
        correct_label = labels[0]
        candidates: List[PointCandidate] = []
        candidates.append(PointCandidate(x=base_x, y=base_y, label=correct_label))
        target_count = len(labels)
        max_attempts = 600
        attempt = 0
        spread = max(18.0, 0.9 * radius)*2
        while len(candidates) < target_count and attempt < max_attempts:
            attempt += 1
            angle = self._rng.uniform(0.0, math.tau)
            distance = self._rng.uniform(spread * 0.8, spread * 1.8)
            cx = base_x + math.cos(angle) * distance
            cy = base_y + math.sin(angle) * distance
            inside_bounds = (
                left + radius <= cx <= right - radius and
                top + radius <= cy <= bottom - radius
            )
            if not inside_bounds:
                continue
            too_close = False
            for existing in candidates:
                if math.hypot(existing.x - cx, existing.y - cy) < radius * 1.2:
                    too_close = True
                    break
            if too_close:
                continue
            label = labels[len(candidates)]
            candidates.append(PointCandidate(x=cx, y=cy, label=label))
            if attempt == 1 and self._rng.random() < 0.8:
                base_x = cx
                base_y = cy
        if len(candidates) < target_count:
            padding = radius * 1.6
            needed = target_count - len(candidates)
            for i in range(needed):
                shift_x = padding if i % 2 == 0 else -padding
                shift_y = padding if (i // 2) % 2 == 0 else -padding
                cx = base_x + shift_x
                cy = base_y + shift_y
                if cx < left + radius:
                    cx = left + radius
                elif cx > right - radius:
                    cx = right - radius
                if cy < top + radius:
                    cy = top + radius
                elif cy > bottom - radius:
                    cy = bottom - radius
                label = labels[len(candidates)]
                candidates.append(PointCandidate(x=cx, y=cy, label=label))
        self.candidates, self.correct_label= candidates, correct_label

    def draw_candidates(
        self,
        draw: Any,
        *,
        highlight_label: Optional[str] = None,
    ) -> None:
        if isinstance(draw, DrawingRecorder):
            draw.add_high_level_command("draw_candidates", highlight_label=highlight_label)
            return

        if ImageDraw is None:
            raise RuntimeError("Pillow is required to draw candidates but is not installed")
        font = self._get_candidate_font()
        active_highlight = highlight_label.upper() if isinstance(highlight_label, str) else None

        point_radius = self.point_radius
        for candidate in sorted(self.candidates, key=lambda c: c.label):
            cx = round(candidate.x)
            cy = round(candidate.y)
            bbox = (cx - point_radius, cy - point_radius, cx + point_radius, cy + point_radius)
            is_highlight = active_highlight is not None and candidate.label.upper() == active_highlight
            outline = self.CANDIDATE_HIGHLIGHT_COLOR if is_highlight else self.CANDIDATE_OUTLINE_COLOR
            width = self.CANDIDATE_HIGHLIGHT_OUTLINE_WIDTH if is_highlight else self.CANDIDATE_OUTLINE_WIDTH
            fill = self.CANDIDATE_HIGHLIGHT_FILL if is_highlight else self.CANDIDATE_BASE_FILL
            draw.ellipse(bbox, fill=fill, outline=outline, width=width)
            text_bbox = font.getbbox(candidate.label)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            tx = cx - text_width // 2
            ty = cy - text_height + self.CANDIDATE_LABEL_OFFSET_Y
            draw.text((tx, ty), candidate.label, fill=self.CANDIDATE_TEXT_COLOR, font=font)

    def draw_line(self,draw,points:List[Point],width_factor:float=1)->None:
        if isinstance(draw, DrawingRecorder):
            pts = [[round(p.x), round(p.y)] for p in points]
            draw.add_high_level_command("draw_line", points=pts, width_factor=width_factor,
                                        fill=self.CANDIDATE_OUTLINE_COLOR, width=round(self.LINE_WIDTH*width_factor))
            return

        draw.line(
            [[round(p.x), round(p.y)] for p in points],
            fill=self.CANDIDATE_OUTLINE_COLOR,
            width=round(self.LINE_WIDTH*width_factor),
        )
        
    def draw_circle(self,draw,center:Point,radius:int)->None:
        if isinstance(draw, DrawingRecorder):
            cx, cy = round(center.x), round(center.y)
            bbox = (cx - radius, cy - radius, cx + radius, cy + radius)
            draw.add_high_level_command("draw_circle", bbox=bbox, outline=self.CANDIDATE_OUTLINE_COLOR, width=self.LINE_WIDTH)
            return

        cx,cy=round(center.x), round(center.y)
        bbox = (cx - radius, cy - radius, cx + radius, cy + radius)
        draw.ellipse(bbox, outline=self.CANDIDATE_OUTLINE_COLOR, width=self.LINE_WIDTH)

    def _get_candidate_font(self) -> Any:
        if self._candidate_font is None:
            self._candidate_font = ImageFont.load_default(15)
        return self._candidate_font
    
    def _render(self, highlight_label: Optional[str]) -> Image.Image:
        raise NotImplementedError("Subclasses must implement _render method")
    
    def get_draw_base(self) -> Tuple[ImageDraw.ImageDraw, Image.Image]:
        width, height = self.canvas_dimensions
        if self._recording_active:
            if self._recorder is None:
                self._recorder = DrawingRecorder(width, height)
            return self._recorder, self._recorder.base_image
            
        base = Image.new("RGB", (width, height), (255, 255, 255))
        draw = ImageDraw.Draw(base)
        return draw, base
    
    def save_puzzle(self) -> PointTargetPuzzleRecord:
        pid = str(uuid.uuid4())
        self.pid=pid
        puzzle_img = self._render(
            highlight_label=None,
        )
        solution_img = self._render(
            highlight_label=self.correct_label,
        )

        self.puzzle_path = self.puzzle_dir / f"{pid}_puzzle.png"
        self.solution_path = self.solution_dir / f"{pid}_solution.png"
        puzzle_img.save(self.puzzle_path)
        solution_img.save(self.solution_path)

        if self.record_video:
            try:
                self.save_video_solution(pid)
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"Video generation failed for {pid}: {e}")
                pass

        return PointTargetPuzzleRecord(
            id=self.pid,
            prompt=self.prompt,
            canvas_dimensions=self.canvas_dimensions,
            margin=self.margin,
            candidates=self.candidates,
            correct_option=self.correct_label,
            image=self.relativize_path(self.puzzle_path),
            solution_image_path=self.relativize_path(self.solution_path),
        )

    def save_video_solution(self, pid: str) -> None:
        # Phase 1: Record trace without highlights
        self._recording_active = True
        self._recorder = None # Reset
        self._render(highlight_label=None) # This populates self._recorder
        trace_base = self._recorder.commands if self._recorder else []
        
        # Phase 2: Record trace with highlights
        self._recorder = None # Reset
        self._render(highlight_label=self.correct_label)
        trace_solution = self._recorder.commands if self._recorder else []
        self._recording_active = False

        if not trace_base and not trace_solution:
            # Generator likely doesn't use get_draw_base
            return

        # Diff commands
        # Use simple queue based diffing. 'trace_base' defines the base commands.
        # Any command in 'trace_solution' that matches the head of 'trace_base' queue is skipped (part of base).
        # Any command that doesn't match is considered new solution geometry.
        
        base_cmds = trace_base
        base_queue = list(trace_base)
        solution_diff = []
        
        for cmd in trace_solution:
            if base_queue and cmd == base_queue[0]:
                base_queue.pop(0)
            else:
                solution_diff.append(cmd)
        
        solution_steps = []
        final_candidates_cmd = None
        
        for cmd in solution_diff:
            if cmd['type'] == 'draw_candidates':
                final_candidates_cmd = cmd
            else:
                solution_steps.append(cmd)
        
        # Generate Video
        video_path = self.solution_dir / f"{pid}_solution.mp4"
        width, height = self.canvas_dimensions
        fps = 30
        
        base_hold = 10
        end_hold = 30
        step_frames = 30

        estimated_frames = base_hold + len(solution_steps) * step_frames + end_hold
        if estimated_frames > self.MAX_VIDEO_FRAMES:
            available = self.MAX_VIDEO_FRAMES - base_hold - end_hold
            if len(solution_steps) > 0:
                step_frames = max(1, int(available / len(solution_steps)))
            else:
                step_frames = 1

        # We need a renderer that can paint these commands onto a cv2 frame
        video_renderer = VideoRenderer(width, height, self)
        
        # 1. Base Frame (Static)
        video_renderer.execute_commands(base_cmds)
        # Hold base frame for 1 second
        for _ in range(base_hold):
            video_renderer.write_frame()
            
        # 2. Animate Solution Steps
        for cmd in solution_steps:
             video_renderer.animate_command(cmd, duration_frames=step_frames)
             
        # 3. Animate Answer (Candidates)
        if final_candidates_cmd:
             # We can just switch to the final state or fade it. For now, just execute it.
             # Better: Render the candidates command
             video_renderer.execute_commands([final_candidates_cmd])
             # Hold for result
             for _ in range(end_hold):
                 video_renderer.write_frame()

        video_renderer.save(video_path)

    
    @staticmethod
    def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
        parser = argparse.ArgumentParser(description="Generate point target puzzles")
        parser.add_argument("count", type=int, help="Number of puzzles to create")
        parser.add_argument("--output-dir", type=Path, default=None)
        parser.add_argument("--canvas-width", type=int, default=480)
        parser.add_argument("--aspect", type=float, default=None)
        parser.add_argument("--seed", type=int, default=None)
        parser.add_argument("--prompt", type=str, default=None)
        parser.add_argument("--use-gpt-5", action="store_true", help="Use GPT5_PROMPT defined by puzzle generator. Will be overridden by --prompt if both are provided.")
        parser.add_argument("--video", action="store_true", help="Generate video solution")
        return parser.parse_args(argv)

    @staticmethod
    def main(cls: PointTargetPuzzleGenerator, argv: Optional[List[str]] = None) -> None:
        args = cls._parse_args(argv)
        generator = cls(
            output_dir=args.output_dir,
            canvas_width=args.canvas_width,
            aspect=args.aspect,
            seed=args.seed,
            prompt=cls.DEFAULT_GPT5_PROMPT if args.use_gpt_5 and not args.prompt else args.prompt,
            record_video=args.video,
        )
        records = [generator.create_random_puzzle() for _ in range(max(1, args.count))]
        generator.write_metadata(records, generator.output_dir / "data.json")

class DrawingRecorder:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.commands = []
        self.base_image = Image.new("RGB", (width, height), (255, 255, 255))
        
    def add_high_level_command(self, type_name, **kwargs):
         self.commands.append({"type": type_name, **kwargs})

    def line(self, xy, fill=None, width=0, joint=None):
        self.commands.append({"type": "line", "xy": xy, "fill": fill, "width": width})
        
    def ellipse(self, xy, fill=None, outline=None, width=1):
        self.commands.append({"type": "ellipse", "xy": xy, "fill": fill, "outline": outline, "width": width})

    def text(self, xy, text, fill=None, font=None, anchor=None, spacing=4, align="left", direction=None, features=None, language=None, stroke_width=0, stroke_fill=None, embedded_color=False):
        self.commands.append({"type": "text", "xy": xy, "text": text, "fill": fill, "font": font}) # Simplify capture

    def rectangle(self, xy, fill=None, outline=None, width=1):
        self.commands.append({"type": "rectangle", "xy": xy, "fill": fill, "outline": outline, "width": width})

    def arc(self, xy, start, end, fill=None, width=1):
        self.commands.append({"type": "arc", "xy": xy, "start": start, "end": end, "fill": fill, "width": width})
        
    # Pillow Draw methods proxy
    def point(self, xy, fill=None): pass
    def polygon(self, xy, fill=None, outline=None): pass
    def chord(self, xy, start, end, fill=None, outline=None, width=1): pass
    def pieslice(self, xy, start, end, fill=None, outline=None, width=1): pass


class VideoRenderer:
    def __init__(self, width, height, generator: PointTargetPuzzleGenerator):
        self.width = width
        self.height = height
        self.generator = generator
        self.frames = []
        # Current state as PIL image
        self.canvas = Image.new("RGB", (width, height), (255, 255, 255))
        self.draw = ImageDraw.Draw(self.canvas)
        
    def execute_commands(self, commands):
        for cmd in commands:
            self.execute_command_instant(cmd)
            
    def execute_command_instant(self, cmd):
        t = cmd['type']
        if t == 'draw_candidates':
            self.generator.draw_candidates(self.draw, highlight_label=cmd.get('highlight_label'))
        elif t == 'draw_line':
            # Reconstruct args for draw_line call, but better to just draw it directly here
            # to avoid recursion into recording logic (which shouldn't happen as self.draw is real)
            # cmd keys: points, width_factor, fill, width
             pts = cmd['points']
             # flatten if needed or list of [x,y]
             # PIL line expects [(x,y), (x,y)] or [x,y,x,y]
             flat_list = [tuple(p) for p in pts]
             self.draw.line(flat_list, fill=cmd['fill'], width=cmd['width'])
        elif t == 'draw_circle':
             self.draw.ellipse(cmd['bbox'], outline=cmd['outline'], width=cmd['width'])
        
        # Native PIL commands
        elif t == 'line':
            self.draw.line(cmd['xy'], fill=cmd.get('fill'), width=cmd.get('width', 0))
        elif t == 'ellipse':
            self.draw.ellipse(cmd['xy'], fill=cmd.get('fill'), outline=cmd.get('outline'), width=cmd.get('width', 1))
        # ... other PIL types support if needed for complex animations
    
    def animate_command(self, cmd, duration_frames=30):
        t = cmd['type']
        # Currently only animating lines and circles for smooth effect
        if t == 'draw_line' or t == 'line':
            self.animate_line(cmd, duration_frames)
        elif t == 'draw_circle' or t == 'ellipse':
             self.animate_circle(cmd, duration_frames)
        else:
            self.execute_command_instant(cmd)
            for _ in range(duration_frames):
                self.write_frame()

    def animate_line(self, cmd, frames):
        # Extract points
        if cmd['type'] == 'draw_line':
            points = cmd['points'] # [[x,y], [x,y], ...]
            width = cmd['width']
            fill = cmd['fill']
        else:
            xy = cmd['xy']
            # xy can be [x,y, x,y...] or [(x,y), (x,y)...]
            if isinstance(xy[0], (int, float)):
                points = [[xy[i], xy[i+1]] for i in range(0, len(xy), 2)]
            else:
                points = [[p[0], p[1]] for p in xy]
            width = cmd.get('width', 1)
            fill = cmd.get('fill')

        if len(points) < 2: return

        # Calculate total length
        total_len = 0
        segments = []
        for i in range(len(points)-1):
            p1 = points[i]
            p2 = points[i+1]
            dist = math.hypot(p2[0]-p1[0], p2[1]-p1[1])
            segments.append((dist, p1, p2))
            total_len += dist
        
        if total_len == 0: return

        # Draw progressively
        # We need to save the state BEFORE this line, because we redraw the canvas for each frame?
        # No, simpler: We draw onto self.canvas on each frame, but that accumulates?
        # Yes, PIL Draw modifies in place.
        # To animate, we need to restore the "before command" state each frame, draw partial, then finally draw full.
        base_frame = self.canvas.copy()
        
        for f in range(frames):
            temp_canvas = base_frame.copy()
            temp_draw = ImageDraw.Draw(temp_canvas)
            progress = (f + 1) / frames
            current_len = total_len * progress
            
            drawn_len = 0
            for seg_dist, p1, p2 in segments:
                if drawn_len + seg_dist <= current_len:
                    # Full segment
                    temp_draw.line([tuple(p1), tuple(p2)], fill=fill, width=width)
                    drawn_len += seg_dist
                else:
                    # Partial segment
                    # Fraction of this segment needed
                    remain = current_len - drawn_len
                    ratio = remain / seg_dist
                    nx = p1[0] + (p2[0] - p1[0]) * ratio
                    ny = p1[1] + (p2[1] - p1[1]) * ratio
                    temp_draw.line([tuple(p1), (nx, ny)], fill=fill, width=width)
                    break
            
            self.add_pil_frame(temp_canvas)
        
        # Finally execute permanently on main canvas
        self.execute_command_instant(cmd)
        
    def animate_circle(self, cmd, frames):
        if cmd['type'] == 'draw_circle':
             bbox = cmd['bbox']
             width_px = cmd['width']
             outline = cmd['outline']
        else:
             bbox = cmd['xy'] # [x0, y0, x1, y1]
             width_px = cmd.get('width', 1)
             outline = cmd.get('outline')
        
        # ellipse bbox to center radius
        x0, y0, x1, y1 = bbox
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        rx = (x1 - x0) / 2
        ry = (y1 - y0) / 2
        
        base_frame = self.canvas.copy()

        for f in range(frames):
            temp_canvas = base_frame.copy()
            temp_draw = ImageDraw.Draw(temp_canvas)
            
            # Draw arc
            end_angle = 360 * (f + 1) / frames
            start_angle = 0
            
            temp_draw.arc(bbox, start=start_angle, end=end_angle, fill=outline, width=width_px)
            self.add_pil_frame(temp_canvas)
            
        self.execute_command_instant(cmd)

    def write_frame(self):
        self.add_pil_frame(self.canvas)
        
    def add_pil_frame(self, pil_img):
        # Convert RGB PIL to BGR numpy for opencv
        rgb = np.array(pil_img)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        self.frames.append(bgr)

    def save(self, path):
        if not self.frames: return
        
        # Try avc1 (H.264) for better web/vscode compatibility
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(str(path), fourcc, 30.0, (self.width, self.height))
        
        if not out.isOpened():
            # Try vp09 (VP9) for web compatibility if H.264 is missing
            fourcc = cv2.VideoWriter_fourcc(*'vp09')
            out = cv2.VideoWriter(str(path), fourcc, 30.0, (self.width, self.height))

        if not out.isOpened():
            # Fallback to mp4v if avc1 and vp09 are not supported.
            # This is common on systems without recent codecs.
            print(f"Warning: avc1/vp09 codec not available for {path}, falling back to mp4v", flush=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(path), fourcc, 30.0, (self.width, self.height))

        if not out.isOpened():
            print(f"Error: Failed to open VideoWriter for {path}. Codecs avc1, vp09, mp4v failed.", flush=True)
            print("Please install ffmpeg / libav codecs for OpenCV.", flush=True)
            return
            
        for frame in self.frames:
            out.write(frame)
        out.release()



class PointTargetPuzzleEvaluator(AbstractPuzzleEvaluator):
    """Base evaluator utilities for point-target option puzzles."""

    VIDEO_GLOBS = ("video_*.mp4", "video_*.webm", "video_*.mov", "*.mp4", "*.webm", "*.mov")

    def image_option_from_path(
        self,
        candidate_image: PathLike,
        record: Dict[str, object],
    ) -> Tuple[Optional[str], int, Optional[Tuple[float, float]]]:
        candidate_path = Path(candidate_image)
        loaded_frame = cv2.imread(candidate_path.as_posix(), cv2.IMREAD_COLOR)
        if loaded_frame is None:
            return None, 0, None
        rgb_frame = cv2.cvtColor(loaded_frame, cv2.COLOR_BGR2RGB)
        return self.image_option_from_frame(rgb_frame, record)

    def image_option_from_frame(
        self,
        frame: np.ndarray,
        record: Dict[str, object],
    ) -> Tuple[Optional[str], int, Optional[Tuple[float, float]]]:
        return self._score_red_point(frame, record)

    def video_option_from_attempt(
        self,
        attempt_dir: Path,
        record: Dict[str, object],
        sample_stride: int,
    ) -> Optional[str]:
        stride = sample_stride if sample_stride > 0 else 1
        counts: Dict[str, int] = {}
        for video_path in self._iter_video_files(attempt_dir):
            capture = cv2.VideoCapture(video_path.as_posix())
            if not capture.isOpened():
                capture.release()
                continue
            frame_index = 0
            success, frame = capture.read()
            while success:
                if frame_index % stride == 0:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    label, _, _ = self._score_red_point(rgb_frame, record)
                    if label:
                        key = label.upper()
                        counts[key] = counts.get(key, 0) + 1
                frame_index += 1
                success, frame = capture.read()
            capture.release()
        if not counts:
            return None
        best_count = max(counts.values())
        best_labels = [label for label, count in counts.items() if count == best_count]
        best_labels.sort()
        return best_labels[0]

    def transcript_option_from_attempt(self, attempt_dir: Path) -> Optional[str]:
        transcript_result = self.transcribe_video(attempt_dir)
        value = transcript_result.get("first_nato_word")
        if isinstance(value, str):
            stripped = value.strip()
            if stripped:
                return stripped.upper()
        return None

    def text_option_from_attempt(self, attempt_dir: Path) -> Optional[str]:
        text_path = attempt_dir / "content.txt"
        if not text_path.exists() or not text_path.is_file():
            raise FileNotFoundError(f"Text not found: {text_path}")
        text_payload = text_path.read_text(encoding="utf-8")
        return self.extract_first_nato_word(text_payload)

    def _score_red_point(
        self,
        frame: np.ndarray,
        record: Dict[str, object],
    ) -> Tuple[Optional[str], int, Optional[Tuple[float, float]]]:
        height = frame.shape[0]
        width = frame.shape[1]
        canvas_dims_obj = record.get("canvas_dimensions")
        scale = self._extract_scale(canvas_dims_obj, width, height)
        if scale is None:
            return None, 0, None
        scale_x, scale_y = scale
        red_mask = self._red_mask(frame)
        red_pixels = np.column_stack(np.nonzero(red_mask > 0.5))
        candidates_raw = record.get("candidates") or []
        scaled_candidates: List[Dict[str, object]] = []
        for entry in candidates_raw:
            label = entry.get("label")
            x, y = entry.get("x"), entry.get("y")

            cx = float(x) * scale_x
            cy = float(y) * scale_y
            scaled_candidates.append({"label": label, "x": cx, "y": cy})
        # if want to exclude red pixels far from candidates: 
        # threshold = min(height, width) / 5.0
        # if red_pixels.size and scaled_candidates:
        #     coords = red_pixels.astype(np.float32)
        #     candidate_coords = np.array([[c["y"], c["x"]] for c in scaled_candidates], dtype=np.float32)
        #     if candidate_coords.size:
        #         distance_sq = np.sum((coords[:, None, :] - candidate_coords[None, :, :]) ** 2, axis=2)
        #         mask = np.min(distance_sq, axis=1) <= (threshold * threshold)
        #         red_pixels = red_pixels[mask]
        red_count = int(red_pixels.shape[0])
        if red_count < 20:
            return None, red_count, None
        mean_y = float(red_pixels[:, 0].mean())
        mean_x = float(red_pixels[:, 1].mean())
        red_point = (mean_x, mean_y)
        # print(f"Detected {red_count} red pixels, centroid at ({mean_x:.1f}, {mean_y:.1f})")
        
        best_label: Optional[str] = None
        best_distance: float = math.inf
        for scaled in scaled_candidates:
            label, cx, cy = scaled['label'], scaled['x'], scaled['y']
            distance = math.hypot(cx - mean_x, cy - mean_y)
            # print(f"Candidate {label}: position ({cx:.1f}, {cy:.1f}), distance {distance:.1f}")

            if distance < best_distance:
                best_distance = distance
                best_label = label
        return best_label, red_count, red_point

    def _extract_scale(
        self,
        canvas_dims_obj: object,
        width: int,
        height: int,
    ) -> Optional[Tuple[float, float]]:
        if isinstance(canvas_dims_obj, (list, tuple)) and len(canvas_dims_obj) >= 2:
            raw_width = canvas_dims_obj[0]
            raw_height = canvas_dims_obj[1]
        elif isinstance(canvas_dims_obj, dict) and {"width", "height"} <= set(canvas_dims_obj):
            raw_width = canvas_dims_obj["width"]
            raw_height = canvas_dims_obj["height"]
        else:
            return None
        if not isinstance(raw_width, (int, float)) or not isinstance(raw_height, (int, float)):
            return None
        canvas_width = float(raw_width)
        canvas_height = float(raw_height)
        if canvas_width <= 0 or canvas_height <= 0:
            return None
        scale_x = width / canvas_width
        scale_y = height / canvas_height
        return scale_x, scale_y

    def _iter_video_files(self, attempt_dir: Path) -> List[Path]:
        seen = set()
        videos: List[Path] = []
        for pattern in self.VIDEO_GLOBS:
            for candidate in attempt_dir.glob(pattern):
                if candidate.is_file() and candidate not in seen:
                    seen.add(candidate)
                    videos.append(candidate)
        videos.sort(key=lambda path: path.name)
        return videos

    def _red_mask(self, frame: np.ndarray) -> np.ndarray:
        red = frame[:, :, 0].astype(np.float32)
        green = frame[:, :, 1].astype(np.float32)
        blue = frame[:, :, 2].astype(np.float32)
        dominance = red - np.maximum(green, blue)
        mask = (
            (red >= 140.0) &
            (dominance >= 40.0) &
            (green <= 130.0) &
            (blue <= 130.0)
        )
        return mask.astype(np.float32)

    def evaluate(
        self,
        puzzle_id: str,
        candidate_image: PathLike,
        *,
        video_sample_stride: int = 5,
    ) -> AbstractPuzzleEvaluator.OptionEvaluationResult:
        record = self.get_record(puzzle_id)
        correct = str(record.get("correct_option", "")).strip().upper()
        if not correct or len(correct) != 1:
            raise ValueError("Puzzle record missing valid 'correct_option' (single letter)")

        candidate_path = Path(candidate_image)
        attempt_dir = candidate_path.parent

        transcript_option = self.transcript_option_from_attempt(attempt_dir)
        text_option = self.text_option_from_attempt(attempt_dir)
        video_option = self.video_option_from_attempt(attempt_dir, record, video_sample_stride)
        image_option, red_pixel_count, red_centroid = self.image_option_from_path(candidate_path, record)

        result = AbstractPuzzleEvaluator.OptionEvaluationResult(
            puzzle_id=puzzle_id,
            correct_option=correct,
            transcribe_option=transcript_option,
            video_option=video_option,
            image_option=image_option,
            text_option=text_option,
            attempt_dir=attempt_dir.as_posix(),
        )
        result.red_pixel_count = red_pixel_count
        result.red_centroid = red_centroid
        return result
    
    @staticmethod
    def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
        parser = argparse.ArgumentParser(description="Evaluate point target puzzles")
        parser.add_argument("metadata", type=Path)
        parser.add_argument("puzzle_id", type=str)
        parser.add_argument("candidate", type=Path)
        parser.add_argument("--base-dir", type=Path, default=None)
        parser.add_argument("--video-stride", dest="video_sample_stride", type=int, default=5)
        return parser.parse_args(argv)


    @staticmethod
    def main(argv: Optional[list[str]] = None) -> None:
        args = PointTargetPuzzleEvaluator._parse_args(argv)
        evaluator = PointTargetPuzzleEvaluator(args.metadata, base_dir=args.base_dir)
        result = evaluator.evaluate(
            args.puzzle_id,
            args.candidate,
            video_sample_stride=args.video_sample_stride,
        )
        print(json.dumps(result.to_dict(), indent=2))

__all__ = [
    "PointCandidate",
    "PointTargetPuzzleGenerator",
    "PointTargetPuzzleEvaluator",
]
