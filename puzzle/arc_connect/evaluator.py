"""Evaluator for arc connection puzzles.

Transcribes the attempt video in the output folder to detect the spoken option
letter (A–E) using ``scripts/transcribe_video.py`` and analyses the candidate
image to determine which right-hand arc, if any, connects to the left arc via a
black line once the masked band is removed.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from collections import deque
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import re

import shutil
import tempfile
from collections import Counter

import numpy as np
from PIL import Image

from ..base import AbstractPuzzleEvaluator, PathLike


class ArcConnectEvaluator(AbstractPuzzleEvaluator):
    """Transcribe the attempt's video and check the spoken option."""

    VIDEO_GLOBS = ("video_*.mp4", "video_*.webm", "video_*.mov", "*.mp4", "*.webm", "*.mov")
    BLACK_THRESHOLD = 92
    START_SEARCH_RADIUS = 8
    TARGET_SEARCH_RADIUS = 6
    CONNECT_RADIUS = 2

    def evaluate(
        self,
        puzzle_id: str,
        candidate_image: PathLike,
        *,
        engine: str = "local",
        model: str = "whisper-1",
        base_url: Optional[str] = None,
    ) -> AbstractPuzzleEvaluator.OptionEvaluationResult:
        record = self.get_record(puzzle_id)
        correct = str(record.get("correct_option", "")).strip().upper() or ""
        if correct not in ("A", "B", "C", "D", "E"):
            raise ValueError("Puzzle record missing valid 'correct_option' (A–E)")

        candidate_path = Path(candidate_image)
        attempt_dir = candidate_path.parent
        
        text_path = attempt_dir / "content.txt"
        if not text_path.exists() or not text_path.is_file():
            raise FileNotFoundError(f"Text not found: {text_path}")
        text_response = text_path.read_text(encoding="utf-8")
        transcript_option, transcript_info, video_path_str = self._transcribe_attempt(
            attempt_dir,
            engine=engine,
            model=model,
            base_url=base_url,
        )
        text_option = self._text_option_from_response(text_response)
        image_option, connected_labels, image_debug = self._image_option_from_path(
            candidate_path,
            record,
            save_debug_image=True,
        )
        # Determine video option by sampling frames and voting.
        video_option = None
        if video_path_str is not None:
            try:
                video_option = self._video_option_from_video_path(Path(video_path_str), record)
            except Exception:
                # If frame extraction fails (ffmpeg not present or other), leave as None
                video_option = None

        result = AbstractPuzzleEvaluator.OptionEvaluationResult(
            puzzle_id=puzzle_id,
            correct_option=correct,
            transcribe_option=transcript_option,
            video_option=None,
            image_option=image_option,
            text_option=text_option,
            attempt_dir=attempt_dir.as_posix(),
        )
        result.connected_labels = connected_labels
        result.image_connections = image_debug
        result.image_is_correct = connected_labels == [correct]
        result.transcription_json_path = transcript_info
        result.video_path = video_path_str
        result.video_option = video_option
        return result

    def _video_option_from_video_path(self, video_path: Path, record: Dict[str, object]) -> Optional[str]:
        """Extract every 5th frame from video_path, evaluate each frame with image evaluator
        and return the voted option (most common). Returns None if no frames or extraction failed.
        """
        if not video_path.exists() or not video_path.is_file():
            return None
        tmpdir = Path(tempfile.mkdtemp(prefix="arc_connect_frames_"))
        try:
            # Use ffmpeg to extract every 5th frame
            # -vsync vfr ensures variable frame rate output, select keeps frames where (n % 5) == 0
            out_pattern = (tmpdir / "frame_%06d.png").as_posix()
            cmd = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                video_path.as_posix(),
                "-vf",
                "select='not(mod(n,5))',setpts=N/FRAME_RATE/TB",
                "-vsync",
                "vfr",
                out_pattern,
            ]
            completed = subprocess.run(cmd, capture_output=True, text=True)
            if completed.returncode != 0:
                # ffmpeg failed; give up
                return None
            frames = sorted(tmpdir.glob("frame_*.png"))
            if not frames:
                return None
            votes: List[str] = []
            for frame in frames:
                try:
                    opt, _, _ = self._image_option_from_path(frame, record)
                except Exception:
                    opt = None
                if opt is None or opt == "none" or opt == "multiple": 
                    continue
                votes.append(opt)
            if not votes:
                return None
            counts = Counter(votes)
            # Determine winner; if tie between top keys return "multiple"
            most_common = counts.most_common()
            if len(most_common) == 0:
                return None
            if len(most_common) == 1:
                return most_common[0][0]
            top_count = most_common[0][1]
            top_items = [item for item, cnt in most_common if cnt == top_count]
            if len(top_items) == 1:
                return top_items[0]
            return "multiple"
        finally:
            try:
                shutil.rmtree(tmpdir)
            except Exception:
                pass

    def _transcribe_attempt(
        self,
        attempt_dir: Path,
        *,
        engine: str,
        model: str,
        base_url: Optional[str],
    ) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        video_path = self._first_video(attempt_dir)
        if video_path is None:
            return None, None, None
        json_out = attempt_dir / "transcription.json"
        cmd: List[str] = [
            str(Path.cwd() / "scripts" / "transcribe_video.py"),
            video_path.as_posix(),
            "--output-json",
            json_out.as_posix(),
        ]
        if engine == "api":
            cmd.extend(["--engine", "api", "--model", model])
            if base_url:
                cmd.extend(["--base-url", base_url])
        else:
            cmd.extend(["--engine", "local"])

        import sys as _sys
        py_cmd = [_sys.executable, cmd[0]] + cmd[1:]
        completed = subprocess.run(py_cmd, capture_output=True, text=True)
        transcript_path: Optional[Path] = None
        stdout_lines = completed.stdout.strip().splitlines()
        if stdout_lines:
            candidate = Path(stdout_lines[-1].strip())
            if candidate.exists() and candidate.is_file():
                transcript_path = candidate
        if transcript_path is None and json_out.exists() and json_out.is_file():
            transcript_path = json_out
        if transcript_path is None:
            return None, None, video_path.as_posix()
        payload = json.loads(transcript_path.read_text(encoding="utf-8"))
        nato_word = payload.get("first_nato_word")
        option: Optional[str] = None
        if isinstance(nato_word, str) and nato_word.strip():
            option = nato_word.strip().upper()[:1]
        if option is None:
            transcript_text = payload.get("transcript")
            if isinstance(transcript_text, str) and transcript_text.strip():
                option = AbstractPuzzleEvaluator.extract_first_nato_word(transcript_text.strip())
        return option, transcript_path.as_posix(), video_path.as_posix()

    def _text_option_from_response(self, response: str) -> Optional[str]:
        if not response:
            return None
        matches = re.findall(r"\b([A-E])\b", response.upper())
        if not matches:
            return None
        return matches[-1]

    def _image_option_from_path(
        self,
        image_path: Path,
        record: Dict[str, object],
        *,
        save_debug_image: bool = False,
    ) -> Tuple[str, List[str], Dict[str, object]]:
        target_width, target_height = self._record_canvas_dimensions(record)
        with Image.open(image_path) as pil_image:
            rgb_image = pil_image.convert("RGB")
            if rgb_image.size != (target_width, target_height):
                rgb_image = rgb_image.resize((target_width, target_height), Image.NEAREST)
            data = np.array(rgb_image)
        black_mask = np.max(data, axis=2) <= self.BLACK_THRESHOLD
        branch_upper = True
        mask_rect = self._mask_rect(record)
        start_point = self._find_start_point(record, black_mask, mask_rect, branch_upper)
        if start_point is None:
            branch_upper = False
            start_point = self._find_start_point(record, black_mask, mask_rect, branch_upper)
        start_hint = start_point
        if start_hint is None:
            start_hint = self._start_point_hint(record, black_mask, mask_rect, branch_upper)
        debug_info: Dict[str, object] = {
            "branch_upper": branch_upper,
            "start_point": start_point,
        }
        if start_hint is not None:
            debug_info["start_hint"] = start_hint
        if start_point is None:
            debug_info["visited_sum"] = 0
            if save_debug_image:
                start_debug_path = self._write_start_debug(image_path, data, start_hint)
                if start_debug_path is not None:
                    debug_info["start_debug_image"] = start_debug_path.as_posix()
            return "none", [], debug_info

        visited = self._flood_from(black_mask, start_point)
        debug_image_path: Optional[Path] = None
        if save_debug_image:
            debug_image_path = self._write_flood_debug(image_path, data, visited)
        connections: List[str] = []
        candidate_points: Dict[str, Optional[Tuple[int, int]]] = {}
        candidates = record.get("candidates", [])
        for candidate in candidates:  # type: ignore[assignment]
            if not isinstance(candidate, dict):
                continue
            label = candidate.get("label")
            if not isinstance(label, str):
                continue
            label = label.strip().upper()[:1]
            if label not in ("A", "B", "C", "D", "E"):
                continue
            point = self._find_candidate_point(candidate, mask_rect[2], black_mask, branch_upper)
            candidate_points[label] = point
            if point is None:
                continue
            if self._visited_near(visited, point, self.CONNECT_RADIUS):
                connections.append(label)
        debug_info["candidate_points"] = candidate_points
        debug_info["visited_sum"] = int(np.sum(visited))
        if debug_image_path is not None:
            debug_info["flood_debug_image"] = debug_image_path.as_posix()
        if not connections:
            return "none", connections, debug_info
        if len(connections) == 1:
            return connections[0], connections, debug_info
        return "multiple", connections, debug_info

    def _write_flood_debug(
        self,
        source_image_path: Path,
        rgb_data: np.ndarray,
        visited: np.ndarray,
    ) -> Optional[Path]:
        if rgb_data.ndim != 3 or rgb_data.shape[2] != 3:
            return None
        overlay = np.array(rgb_data, copy=True)
        overlay[visited] = np.array([255, 0, 0], dtype=np.uint8)
        debug_dir = Path.cwd() / "debug" / "arc_connect"
        debug_dir.mkdir(parents=True, exist_ok=True)
        parent_part = source_image_path.parent.name.strip().replace(" ", "_")
        if parent_part:
            filename = f"{parent_part}_{source_image_path.stem}_flood.png"
        else:
            filename = f"{source_image_path.stem}_flood.png"
        debug_path = debug_dir / filename
        Image.fromarray(overlay).save(debug_path)
        return debug_path

    def _start_point_hint(
        self,
        record: Dict[str, object],
        black_mask: np.ndarray,
        mask_rect: Tuple[int, int, int, int],
        branch_upper: bool,
    ) -> Optional[Tuple[int, int]]:
        left_arc = record.get("left_arc")
        if not isinstance(left_arc, dict):
            return None
        ys = self._circle_crossing(left_arc, float(mask_rect[0]))
        if ys is None:
            return None
        target_y = ys[0] if branch_upper else ys[1]
        height, width = black_mask.shape
        x_hint = self._clamp_round(mask_rect[0] - 1, 0, width - 1)
        y_hint = self._clamp_round(target_y, 0, height - 1)
        return (x_hint, y_hint)

    def _write_start_debug(
        self,
        source_image_path: Path,
        rgb_data: np.ndarray,
        start_hint: Optional[Tuple[int, int]],
    ) -> Optional[Path]:
        if rgb_data.ndim != 3 or rgb_data.shape[2] != 3:
            return None
        overlay = np.array(rgb_data, copy=True)
        if start_hint is not None:
            x_hint, y_hint = start_hint
            height, width = overlay.shape[:2]
            x_min = max(0, x_hint - 2)
            x_max = min(width - 1, x_hint + 2)
            y_min = max(0, y_hint - 2)
            y_max = min(height - 1, y_hint + 2)
            overlay[y_min:y_max + 1, x_min:x_max + 1] = np.array([255, 0, 0], dtype=np.uint8)
        debug_dir = Path.cwd() / "debug" / "arc_connect"
        debug_dir.mkdir(parents=True, exist_ok=True)
        parent_part = source_image_path.parent.name.strip().replace(" ", "_")
        if parent_part:
            filename = f"{parent_part}_{source_image_path.stem}_start.png"
        else:
            filename = f"{source_image_path.stem}_start.png"
        debug_path = debug_dir / filename
        Image.fromarray(overlay).save(debug_path)
        return debug_path

    def _mask_rect(self, record: Dict[str, object]) -> Tuple[int, int, int, int]:
        mask = record.get("mask_rect")
        if isinstance(mask, Sequence) and len(mask) >= 4:
            left = int(round(float(mask[0])))
            top = int(round(float(mask[1])))
            right = int(round(float(mask[2])))
            bottom = int(round(float(mask[3])))
            return (left, top, right, bottom)
        raise ValueError("Puzzle record missing mask_rect")

    def _sorted_labels_by_crossing(
        self,
        candidates: Iterable[object],
        mask_right: float,
        branch_index: int,
    ) -> List[str]:
        pairs: List[Tuple[str, float]] = []
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            label = candidate.get("label")
            if not isinstance(label, str):
                continue
            ys = self._circle_crossing(candidate, mask_right)
            if ys is None:
                continue
            y_val = ys[branch_index]
            pairs.append((label.strip().upper()[:1], y_val))
        pairs.sort(key=lambda item: item[1])
        return [label for label, _ in pairs]

    def _order_mismatch(self, proposal: List[str], actual: List[str]) -> int:
        mismatch = 0
        for idx, label in enumerate(proposal):
            if idx >= len(actual):
                mismatch += len(proposal) - idx
                break
            if actual[idx] != label:
                mismatch += 1
        if len(actual) > len(proposal):
            mismatch += len(actual) - len(proposal)
        return mismatch

    def _circle_crossing(
        self,
        circle_obj: Dict[str, object],
        x_line: float,
    ) -> Optional[Tuple[float, float]]:
        cx_raw = circle_obj.get("cx")
        cy_raw = circle_obj.get("cy")
        r_raw = circle_obj.get("r")
        if not isinstance(cx_raw, (int, float)):
            return None
        if not isinstance(cy_raw, (int, float)):
            return None
        if not isinstance(r_raw, (int, float)):
            return None
        cx = float(cx_raw)
        cy = float(cy_raw)
        radius = float(r_raw)
        dx = x_line - cx
        discriminant = radius * radius - dx * dx
        if discriminant <= 1e-6:
            return None
        root = math.sqrt(discriminant)
        return (cy - root, cy + root)

    def _find_start_point(
        self,
        record: Dict[str, object],
        black_mask: np.ndarray,
        mask_rect: Tuple[int, int, int, int],
        branch_upper: bool,
    ) -> Optional[Tuple[int, int]]:
        left_arc = record.get("left_arc")
        if not isinstance(left_arc, dict):
            return None
        ys = self._circle_crossing(left_arc, float(mask_rect[0]))
        if ys is None:
            return None
        target_y = ys[0] if branch_upper else ys[1]
        x_candidates = [mask_rect[0] - 2, mask_rect[0] - 1, mask_rect[0] + 1]
        return self._seek_black_pixel(black_mask, x_candidates, target_y, self.START_SEARCH_RADIUS)

    def _find_candidate_point(
        self,
        candidate: Dict[str, object],
        mask_right: int,
        black_mask: np.ndarray,
        branch_upper: bool,
    ) -> Optional[Tuple[int, int]]:
        ys = self._circle_crossing(candidate, float(mask_right))
        if ys is None:
            return None
        target_y = ys[0] if branch_upper else ys[1]
        x_candidates = [mask_right - 1, mask_right - 2, mask_right - 3]
        return self._seek_black_pixel(black_mask, x_candidates, target_y, self.TARGET_SEARCH_RADIUS)

    def _seek_black_pixel(
        self,
        black_mask: np.ndarray,
        x_candidates: List[int],
        target_y: float,
        radius: int,
    ) -> Optional[Tuple[int, int]]:
        height, width = black_mask.shape
        y_base = self._clamp_round(target_y, 0, height - 1)
        for base_x in x_candidates:
            x_base = self._clamp_round(base_x, 0, width - 1)
            found = self._search_radius(black_mask, x_base, y_base, radius)
            if found is not None:
                return found
        return None

    def _search_radius(
        self,
        black_mask: np.ndarray,
        base_x: int,
        base_y: int,
        radius: int,
    ) -> Optional[Tuple[int, int]]:
        height, width = black_mask.shape
        for current_radius in range(radius + 1):
            x_min = max(0, base_x - current_radius)
            x_max = min(width - 1, base_x + current_radius)
            y_min = max(0, base_y - current_radius)
            y_max = min(height - 1, base_y + current_radius)
            for y in range(y_min, y_max + 1):
                for x in range(x_min, x_max + 1):
                    dx = x - base_x
                    dy = y - base_y
                    if dx * dx + dy * dy > current_radius * current_radius:
                        continue
                    if black_mask[y, x]:
                        return (x, y)
        return None

    def _flood_from(
        self,
        black_mask: np.ndarray,
        start: Tuple[int, int],
    ) -> np.ndarray:
        height, width = black_mask.shape
        visited = np.zeros((height, width), dtype=bool)
        queue: deque[Tuple[int, int]] = deque()
        sx, sy = start
        if not black_mask[sy, sx]:
            return visited
        visited[sy, sx] = True
        queue.append((sx, sy))
        neighbors = [
            (-1, -1), (0, -1), (1, -1),
            (-1, 0), (1, 0),
            (-1, 1), (0, 1), (1, 1),
        ]
        while queue:
            cx, cy = queue.popleft()
            for dx, dy in neighbors:
                nx = cx + dx
                ny = cy + dy
                if nx < 0 or ny < 0 or nx >= width or ny >= height:
                    continue
                if not black_mask[ny, nx]:
                    continue
                if visited[ny, nx]:
                    continue
                visited[ny, nx] = True
                queue.append((nx, ny))
        return visited

    def _visited_near(
        self,
        visited: np.ndarray,
        point: Tuple[int, int],
        radius: int,
    ) -> bool:
        height, width = visited.shape
        px, py = point
        x_min = max(0, px - radius)
        x_max = min(width - 1, px + radius)
        y_min = max(0, py - radius)
        y_max = min(height - 1, py + radius)
        for y in range(y_min, y_max + 1):
            for x in range(x_min, x_max + 1):
                if visited[y, x]:
                    return True
        return False

    def _clamp_round(self, value: float, lower: int, upper: int) -> int:
        rounded = int(round(float(value)))
        if rounded < lower:
            return lower
        if rounded > upper:
            return upper
        return rounded

    def _first_video(self, attempt_dir: Path) -> Optional[Path]:
        for pattern in self.VIDEO_GLOBS:
            for candidate in attempt_dir.glob(pattern):
                if candidate.is_file():
                    return candidate
        return None


__all__ = ["ArcConnectEvaluator"]


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate arc connection puzzles via video transcription")
    parser.add_argument("metadata", type=Path)
    parser.add_argument("puzzle_id", type=str)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--base-dir", type=Path, default=None)
    parser.add_argument("--engine", choices=["local", "api"], default="local")
    parser.add_argument("--model", type=str, default="whisper-1")
    parser.add_argument("--base-url", dest="base_url", type=str, default=None)
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)
    evaluator = ArcConnectEvaluator(args.metadata, base_dir=args.base_dir)
    result = evaluator.evaluate(
        args.puzzle_id,
        args.candidate,
        engine=args.engine,
        model=args.model,
        base_url=args.base_url,
    )
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
