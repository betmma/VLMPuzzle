import json
import tempfile
import unittest
from pathlib import Path

from puzzle.rects import RectsEvaluator


class RectsEvaluatorTextTests(unittest.TestCase):
    def setUp(self) -> None:
        debug_root = Path(__file__).resolve().parent / "debug"
        debug_root.mkdir(exist_ok=True)
        self._tmpdir = tempfile.TemporaryDirectory(dir=debug_root)
        self.tmp_path = Path(self._tmpdir.name)
        self.metadata_path = self.tmp_path / "rects_metadata.json"
        record = {
            "id": "rect-test",
            "rectangles": [
                {"z": 3, "color": [68, 229, 229]},
                {"z": 2, "color": [229, 68, 68]},
                {"z": 1, "color": [149, 68, 229]},
                {"z": 0, "color": [149, 229, 68]},
            ],
            "image": "puzzles/rect-test_puzzle.png",
            "solution_image_path": "solutions/rect-test_solution.png",
            "canvas_dimensions": [384, 384],
        }
        self.metadata_path.write_text(json.dumps([record]), encoding="utf-8")
        self.attempt_dir = self.tmp_path / "attempt"
        self.attempt_dir.mkdir()
        self.evaluator = RectsEvaluator(self.metadata_path, base_dir=self.tmp_path)

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_text_response_matches_expected_order(self) -> None:
        text_payload = (
            "The stacked rectangles from highest to lowest are turquoise, crimson, violet, lime."
        )
        content_path = self.attempt_dir / "content.txt"
        content_path.write_text(text_payload, encoding="utf-8")
        candidate_path = self.attempt_dir / "result.png"

        result = self.evaluator.evaluate("rect-test", candidate_path)

        self.assertEqual(result.correct, result.total)
        self.assertEqual(result.expected_order, result.predicted_order[: len(result.expected_order)])
        self.assertEqual(result.expected_names, result.predicted_names[: len(result.expected_names)])


if __name__ == "__main__":
    unittest.main()
