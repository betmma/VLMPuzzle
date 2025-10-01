import json
import tempfile
import unittest
from pathlib import Path

from puzzle.arcagi import ArcPuzzleGenerator, ArcPuzzleEvaluator


class ArcPuzzleIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.dataset_dir = root / "training"
        self.dataset_dir.mkdir()
        self.task_path = self.dataset_dir / "sample_task.json"
        sample_task = {
            "train": [
                {
                    "input": [
                        [0, 1, 0],
                        [1, 0, 1],
                        [0, 1, 0],
                    ],
                    "output": [
                        [1, 1, 1],
                        [1, 0, 1],
                        [1, 1, 1],
                    ],
                },
                {
                    "input": [
                        [0, 0, 0],
                        [0, 1, 0],
                        [0, 0, 0],
                    ],
                    "output": [
                        [2, 2, 2],
                        [2, 0, 2],
                        [2, 2, 2],
                    ],
                },
            ],
            "test": [
                {
                    "input": [
                        [0, 1, 0],
                        [1, 0, 1],
                        [0, 1, 0],
                    ],
                    "output": [
                        [2, 2, 2],
                        [2, 2, 2],
                        [2, 2, 2],
                    ],
                },
            ],
        }
        self.task_path.write_text(json.dumps(sample_task), encoding="utf-8")
        self.output_dir = root / "arc_puzzles"
        self.generator = ArcPuzzleGenerator(dataset_dir=self.dataset_dir, output_dir=self.output_dir, cell_size=16, seed=123)
        self.record = self.generator.create_puzzle(task_path=self.task_path, puzzle_id="arc-test")
        metadata_path = self.output_dir / "puzzles.json"
        metadata_path.write_text(json.dumps([self.record.to_dict()]), encoding="utf-8")
        self.metadata_path = metadata_path
        self.evaluator = ArcPuzzleEvaluator(self.metadata_path)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_metadata_contains_test_output(self) -> None:
        placement_kinds = {placement.kind for placement in self.record.placements}
        self.assertIn("test_output", placement_kinds)
        self.assertEqual(self.record.test_rows, 3)
        self.assertEqual(self.record.test_cols, 3)

    def test_solution_image_scores_full_accuracy(self) -> None:
        solution_path = (self.output_dir / self.record.solution_image_path).resolve()
        result = self.evaluator.evaluate(self.record.id, solution_path)
        self.assertEqual(result.correct_cells, result.total_cells)
        self.assertEqual(result.predicted_grid, self.record.test_output)

    def test_blank_puzzle_image_scores_lower_accuracy(self) -> None:
        puzzle_path = (self.output_dir / self.record.puzzle_image_path).resolve()
        result = self.evaluator.evaluate(self.record.id, puzzle_path)
        self.assertLess(result.accuracy, 1.0)


if __name__ == "__main__":
    unittest.main()
