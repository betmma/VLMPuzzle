import json
import math
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from puzzle.arc_connect.evaluator import ArcConnectEvaluator


class ArcConnectEvaluatorTests(unittest.TestCase):
    def setUp(self) -> None:
        debug_root = Path("debug")
        debug_root.mkdir(parents=True, exist_ok=True)
        self._tmpdir = tempfile.TemporaryDirectory(dir=debug_root)
        base_path = Path(self._tmpdir.name)
        self.base_path = base_path
        self.metadata_path = base_path / "metadata.json"
        self.record_id = "arc-test-001"
        self.canvas_width = 200
        self.canvas_height = 120
        self.mask_left = 90
        self.mask_right = 110
        self.left_arc = {"cx": 70.0, "cy": 60.0, "r": 60.0}
        candidate_cys = [40.0, 50.0, 60.0, 70.0, 80.0]
        labels = list("ABCDE")
        candidates = []
        for label, cy in zip(labels, candidate_cys):
            candidates.append({"cx": 70.0, "cy": cy, "r": 60.0, "label": label})
        record = {
            "id": self.record_id,
            "correct_option": "C",
            "canvas_dimensions": [self.canvas_width, self.canvas_height],
            "mask_rect": [self.mask_left, 5, self.mask_right, self.canvas_height - 5],
            "left_arc": self.left_arc,
            "candidates": candidates,
        }
        self.metadata_path.write_text(json.dumps([record]), encoding="utf-8")
        self.evaluator = ArcConnectEvaluator(self.metadata_path, base_dir=self.base_path)
        self.correct_option = "C"
        self.branch_upper = True
        self.start_y = self._circle_upper_y(self.left_arc, self.mask_left)
        self.target_points = {
            label: self._circle_upper_y(candidate, self.mask_right)
            for label, candidate in zip(labels, candidates)
        }

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def _circle_upper_y(self, circle: dict, x_line: float) -> int:
        cx = float(circle["cx"])
        cy = float(circle["cy"])
        radius = float(circle["r"])
        dx = x_line - cx
        root = math.sqrt(max(radius * radius - dx * dx, 0.0))
        return int(round(cy - root))

    def _make_candidate_image(self, destination: Path, labels_to_connect: list[str]) -> None:
        image = Image.new("RGB", (self.canvas_width, self.canvas_height), "white")
        draw = ImageDraw.Draw(image)
        start_x = self.mask_left - 2
        start_y = self.start_y
        draw.line([(start_x - 3, start_y), (start_x, start_y)], fill=(0, 0, 0), width=3)
        for label in labels_to_connect:
            end_y = self.target_points[label]
            draw.line([(start_x, start_y), (self.mask_right - 1, end_y)], fill=(0, 0, 0), width=3)
            draw.line([(self.mask_right - 1, end_y), (self.mask_right + 6, end_y)], fill=(0, 0, 0), width=3)
        image.save(destination)

    def _write_attempt(self, attempt_dir: Path, connected_labels: list[str], text_content: str) -> Path:
        attempt_dir.mkdir(parents=True, exist_ok=True)
        candidate_path = attempt_dir / "result.png"
        self._make_candidate_image(candidate_path, connected_labels)
        (attempt_dir / "content.txt").write_text(text_content, encoding="utf-8")
        return candidate_path

    def test_detects_single_connection(self) -> None:
        attempt_dir = self.base_path / "attempt_single"
        candidate_path = self._write_attempt(attempt_dir, [self.correct_option], "Answer is C")

        result = self.evaluator.evaluate(self.record_id, candidate_path)

        self.assertEqual(result.image_option, self.correct_option)
        self.assertEqual(result.connected_labels, [self.correct_option])
        self.assertTrue(result.image_is_correct)
        self.assertEqual(result.text_option, "C")
        self.assertIsNone(result.transcribe_option)
        self.assertIsNone(result.video_path)

    def test_detects_multiple_connections(self) -> None:
        attempt_dir = self.base_path / "attempt_multi"
        candidate_path = self._write_attempt(attempt_dir, ["C", "D"], "Answer is D")

        result = self.evaluator.evaluate(self.record_id, candidate_path)

        self.assertEqual(result.image_option, "multiple")
        self.assertEqual(result.connected_labels, ["C", "D"])
        self.assertFalse(result.image_is_correct)


if __name__ == "__main__":
    unittest.main()
