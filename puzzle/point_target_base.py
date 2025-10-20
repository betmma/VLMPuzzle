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
        spread = max(18.0, 0.9 * radius)
        if angle is None:
            angle = self._rng.uniform(0.0, math.tau)
        dx,dy=math.cos(angle)*spread, math.sin(angle)*spread
        for i in range(target_count):
            cx = base_x + dx*(i-correct_index)
            cy = base_y + dy*(i-correct_index)
            label = labels[i]
            candidates.append(PointCandidate(x=cx, y=cy, label=label))
        self.candidates, self.correct_label= candidates, correct_label

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
        spread = max(18.0, 0.9 * radius)
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
        draw.line(
            [[round(p.x), round(p.y)] for p in points],
            fill=self.CANDIDATE_OUTLINE_COLOR,
            width=round(self.LINE_WIDTH*width_factor),
        )
        
    def draw_circle(self,draw,center:Point,radius:int)->None:
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
        )
        records = [generator.create_random_puzzle() for _ in range(max(1, args.count))]
        generator.write_metadata(records, generator.output_dir / "data.json")


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
        if height <= 0 or width <= 0:
            return None, 0, None
        canvas_dims_obj = record.get("canvas_dimensions")
        scale = self._extract_scale(canvas_dims_obj, width, height)
        if scale is None:
            return None, 0, None
        scale_x, scale_y = scale
        red_mask = self._red_mask(frame)
        red_pixels = np.column_stack(np.nonzero(red_mask > 0.5))
        red_count = int(red_pixels.shape[0])
        if red_count < 20:
            return None, red_count, None
        mean_y = float(red_pixels[:, 0].mean())
        mean_x = float(red_pixels[:, 1].mean())
        red_point = (mean_x, mean_y)
        candidates_raw = record.get("candidates")
        best_label: Optional[str] = None
        best_distance: Optional[float] = None
        if isinstance(candidates_raw, Iterable):
            for entry in candidates_raw:
                if isinstance(entry, dict):
                    label_obj = entry.get("label")
                    if isinstance(label_obj, str) and len(label_obj) == 1:
                        cx_obj = entry.get("x")
                        cy_obj = entry.get("y")
                        if isinstance(cx_obj, (int, float)) and isinstance(cy_obj, (int, float)):
                            cx = float(cx_obj) * scale_x
                            cy = float(cy_obj) * scale_y
                            distance = math.hypot(cx - mean_x, cy - mean_y)
                            if best_distance is None or distance < best_distance:
                                best_distance = distance
                                best_label = label_obj
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
