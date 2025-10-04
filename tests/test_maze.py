import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from puzzle.maze import MazeGenerator, MazeEvaluator


class MazeEvaluatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.tmp.name) / "maze"
        self.generator = MazeGenerator(output_dir=self.output_dir, rows=11, cols=11, cell_size=24, seed=42)
        self.record = self.generator.create_puzzle(puzzle_id="maze-test")

        metadata_path = self.output_dir / "puzzles.json"
        metadata_path.write_text(json.dumps([self.record.to_dict()]), encoding="utf-8")
        self.metadata_path = metadata_path
        self.evaluator = MazeEvaluator(metadata_path)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _solution_image_path(self) -> Path:
        return self.output_dir / self.record.solution_image_path

    def test_solution_connects_start_to_goal(self) -> None:
        candidate_path = self._solution_image_path()
        result = self.evaluator.evaluate(self.record.id, candidate_path)
        self.assertTrue(result.connected)
        self.assertTrue(result.touches_goal)
        self.assertFalse(result.stray_in_walls)

    def test_wall_overlap_is_flagged(self) -> None:
        candidate_path = self.output_dir / "wall_overlap.png"
        with Image.open(self._solution_image_path()) as image:
            candidate = image.copy()
        # Paint red on a wall cell (0,0)
        draw_cell = self.record.cell_bboxes[0][0]
        wall_patch = candidate.crop(draw_cell)
        wall_patch.paste((255, 0, 0), (0, 0, wall_patch.width, wall_patch.height))
        candidate.paste(wall_patch, draw_cell)
        candidate.save(candidate_path)

        result = self.evaluator.evaluate(self.record.id, candidate_path)
        self.assertTrue(result.stray_in_walls)
        self.assertIn("overlaps walls", result.message)

    def test_missing_path_fails(self) -> None:
        puzzle_image_path = self.output_dir / self.record.puzzle_image_path
        result = self.evaluator.evaluate(self.record.id, puzzle_image_path)
        self.assertFalse(result.connected)
        self.assertFalse(result.touches_goal)


if __name__ == "__main__":
    unittest.main()
