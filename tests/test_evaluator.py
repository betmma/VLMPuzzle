import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from puzzle.jigsaw import JigsawGenerator
from puzzle.jigsaw import JigsawEvaluator


class JigsawEvaluatorTests(unittest.TestCase):
    """Regression tests ensuring the evaluator scores puzzles correctly."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        base_path = Path(self._tmpdir.name)
        self.output_dir = base_path / "dataset"
        self.generator = JigsawGenerator(
            output_dir=self.output_dir,
            rows=2,
            cols=2,
            image_size=(120, 120),
            allow_rotation=False,
            scatter_scale=1.4,
            seed=42,
        )
        self.base_image = self._make_test_image((160, 160))
        self.record = self.generator.create_puzzle(
            image=self.base_image,
            image_source="synthetic",
            puzzle_id="test-puzzle",
        )
        self.metadata_path = self.output_dir / "puzzles.json"
        self.metadata_path.write_text(json.dumps([self.record.to_dict()]), encoding="utf-8")
        self.original_image_path = self.output_dir / self.record.original_image_path
        self.evaluator = JigsawEvaluator(self.metadata_path)

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def _make_test_image(self, size: tuple[int, int]) -> Image.Image:
        width, height = size
        half_w, half_h = width // 2, height // 2
        image = Image.new("RGB", size, (0, 0, 0))
        quadrants = [
            ((0, 0), (half_w, half_h), (255, 0, 0)),
            ((half_w, 0), (width, half_h), (0, 255, 0)),
            ((0, half_h), (half_w, height), (0, 0, 255)),
            ((half_w, half_h), (width, height), (255, 255, 0)),
        ]
        for (x0, y0), (x1, y1), color in quadrants:
            tile = Image.new("RGB", (x1 - x0, y1 - y0), color)
            image.paste(tile, (x0, y0))
        return image

    def test_perfect_reconstruction_scores_all_pieces(self) -> None:
        candidate_path = Path(self._tmpdir.name) / "perfect.png"
        candidate_path.write_bytes(self.original_image_path.read_bytes())

        result = self.evaluator.evaluate(self.record.id, candidate_path)

        self.assertEqual(result.correct_pieces, result.total_pieces)
        self.assertAlmostEqual(result.accuracy, 1.0)
        self.assertTrue(all(piece.is_correct for piece in result.per_piece))

    def test_evaluator_trims_padding_before_comparing(self) -> None:
        with Image.open(self.original_image_path) as original:
            padded = Image.new("RGB", (original.width + 20, original.height + 20), (10, 10, 10))
            padded.paste(original, (10, 10))
        candidate_path = Path(self._tmpdir.name) / "padded.png"
        padded.save(candidate_path)

        result = self.evaluator.evaluate(self.record.id, candidate_path)

        self.assertAlmostEqual(result.accuracy, 1.0)

    def test_incorrect_tile_reduces_accuracy(self) -> None:
        with Image.open(self.original_image_path) as original:
            candidate = original.copy()
        draw = ImageDraw.Draw(candidate)
        spec = self.record.pieces[0]
        left, top, right, bottom = map(int, spec.bbox)
        draw.rectangle((left, top, right - 1, bottom - 1), fill=(0, 0, 0))
        candidate_path = Path(self._tmpdir.name) / "incorrect.png"
        candidate.save(candidate_path)

        result = self.evaluator.evaluate(self.record.id, candidate_path)

        self.assertLess(result.correct_pieces, result.total_pieces)
        incorrect_ids = {piece.piece_id for piece in result.per_piece if not piece.is_correct}
        self.assertIn(str(spec.id), incorrect_ids)


if __name__ == "__main__":
    unittest.main()
