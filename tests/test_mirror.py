import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image
import numpy as np

from puzzle.mirror import MirrorGenerator, MirrorEvaluator


class MirrorEvaluatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.tmp.name) / "mirror"
        self.generator = MirrorGenerator(output_dir=self.output_dir, rows=4, cols=6, cell_size=32, seed=42)
        self.record = self.generator.create_puzzle(puzzle_id="mirror-test")
        metadata_path = self.output_dir / "puzzles.json"
        metadata_path.write_text(json.dumps([self.record.to_dict()]), encoding="utf-8")
        self.metadata_path = metadata_path
        self.evaluator = MirrorEvaluator(metadata_path)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_monochrome_generation(self) -> None:
        mono_generator = MirrorGenerator(output_dir=self.output_dir, rows=4, cols=6, cell_size=32, monochrome=True, seed=99)
        mono_record = mono_generator.create_puzzle(puzzle_id="mirror-mono")
        left_colors = {tuple(cell.color) for cell in mono_record.colored_cells}
        self.assertLessEqual(len(left_colors), 1)

    def test_cell_aspect_ratio_uses_outer_padding(self) -> None:
        ratio_generator = MirrorGenerator(
            output_dir=self.output_dir,
            rows=4,
            cols=6,
            cell_size=40,
            cell_aspect_ratio=2.0,
            seed=1,
        )
        ratio_record = ratio_generator.create_puzzle(puzzle_id="mirror-ratio")
        pad_left, pad_top, pad_right, pad_bottom = ratio_record.cell_padding
        self.assertEqual(ratio_record.cell_width, ratio_record.cell_size)
        self.assertEqual(ratio_record.cell_height, ratio_record.cell_size)
        self.assertGreater(pad_left + pad_right, 0)
        self.assertEqual(pad_top + pad_bottom, 0)
        inner_left, inner_top, inner_right, inner_bottom = ratio_record.cell_inner_bounds
        inner_width = inner_right - inner_left
        inner_height = inner_bottom - inner_top
        self.assertEqual(inner_width, inner_height)
        self.assertEqual(inner_width, ratio_record.cell_inner_size)

    def _solution_image_path(self) -> Path:
        return self.output_dir / self.record.solution_image_path

    def test_perfect_mirror_scores_full_accuracy(self) -> None:
        candidate_path = self._solution_image_path()
        result = self.evaluator.evaluate(self.record.id, candidate_path)
        self.assertEqual(result.correct_cells, result.total_cells)
        self.assertAlmostEqual(result.accuracy, 1.0)

    def test_random_image_scores_low_accuracy(self) -> None:
        candidate_path = Path(self.tmp.name) / "random.png"
        pad_left, pad_top, pad_right, pad_bottom = self.record.cell_padding
        height = self.record.grid_size[0] * self.record.cell_size + pad_top + pad_bottom
        width = self.record.grid_size[1] * self.record.cell_size + pad_left + pad_right
        random_image = Image.fromarray(
            np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        )
        random_image.save(candidate_path)

        result = self.evaluator.evaluate(self.record.id, candidate_path)
        self.assertLess(result.accuracy, 0.5)


if __name__ == "__main__":
    unittest.main()
