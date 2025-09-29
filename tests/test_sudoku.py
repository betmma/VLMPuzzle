import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from puzzle.sudoku import SudokuEvaluator, SudokuGenerator


class SudokuEvaluatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.tmp.name) / "sudoku"
        self.generator = SudokuGenerator(output_dir=self.output_dir, clue_target=30, seed=123, canvas_size=360)
        self.record = self.generator.create_puzzle(puzzle_id="sudoku-test")

        metadata_path = self.output_dir / "puzzles.json"
        metadata_path.write_text(json.dumps([self.record.to_dict()]), encoding="utf-8")
        self.metadata_path = metadata_path
        self.evaluator = SudokuEvaluator(metadata_path)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _solution_image_path(self) -> Path:
        return self.output_dir / self.record.solution_image_path

    def test_perfect_solution_scores_full_accuracy(self) -> None:
        candidate_path = self._solution_image_path()
        result = self.evaluator.evaluate(self.record.id, candidate_path)
        self.assertEqual(result.correct_cells, result.total_cells)
        self.assertAlmostEqual(result.accuracy, 1.0)
        self.assertTrue(result.is_valid_solution)
        self.assertTrue(all(cell.predicted == cell.expected for cell in result.cell_breakdown))

    def test_incorrect_solution_reports_mistakes(self) -> None:
        solution_path = self._solution_image_path()
        with Image.open(solution_path) as image:
            candidate = image.copy()
        draw = ImageDraw.Draw(candidate)
        bbox = self.record.cell_bboxes[0][0]
        draw.rectangle(bbox, fill=(255, 255, 255))
        candidate_path = Path(self.tmp.name) / "incorrect.png"
        candidate.save(candidate_path)

        result = self.evaluator.evaluate(self.record.id, candidate_path)

        self.assertLess(result.correct_cells, result.total_cells)
        incorrect_cells = [cell for cell in result.cell_breakdown if not cell.is_correct]
        self.assertTrue(any(cell.row == 0 and cell.col == 0 for cell in incorrect_cells))
        self.assertFalse(result.is_valid_solution)
        self.assertIsNone(next(cell.predicted for cell in incorrect_cells if cell.row == 0 and cell.col == 0))

    def test_padding_is_trimmed_before_scoring(self) -> None:
        solution_path = self._solution_image_path()
        with Image.open(solution_path) as image:
            padded = Image.new("RGB", (image.width + 40, image.height + 40), (200, 200, 200))
            padded.paste(image, (20, 20))
        padded_path = Path(self.tmp.name) / "padded.png"
        padded.save(padded_path)

        result = self.evaluator.evaluate(self.record.id, padded_path, trim_tolerance=20)

        self.assertAlmostEqual(result.accuracy, 1.0)
        self.assertTrue(result.is_valid_solution)


if __name__ == "__main__":
    unittest.main()
