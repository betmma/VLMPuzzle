import json
from pathlib import Path

from puzzle.circle_count import (
    CircleCountEvaluator,
    CircleCountGenerator,
)


def test_circle_count_generator_creates_metadata(tmp_path):
    output_dir = tmp_path / "debug" / "circle_out"
    generator = CircleCountGenerator(output_dir, seed=42, canvas_width=200, aspect=0.75)
    record = generator.create_puzzle(puzzle_id="sample")

    assert record.circle_count >= 6
    assert record.circle_count <= 10
    assert record.prompt == "Speak out how many circles are in the image"

    puzzle_path = Path(output_dir) / record.image
    solution_path = Path(output_dir) / record.solution_image_path
    assert puzzle_path.exists()
    assert solution_path.exists()

    assert len(record.circles) == record.circle_count


def test_circle_count_evaluator_reads_text(tmp_path):
    base_dir = tmp_path / "debug"
    output_meta = base_dir / "data.json"
    record = {
        "id": "test",
        "prompt": "Speak out how many circles are in the image",
        "canvas_dimensions": [320, 480],
        "circle_count": 7,
        "circles": [],
        "image": "puzzles/test.png",
        "solution_image_path": "solutions/test_solution.png",
        "type": "circle_count",
    }
    output_meta.parent.mkdir(parents=True, exist_ok=True)
    output_meta.write_text(json.dumps([record]), encoding="utf-8")

    evaluator = CircleCountEvaluator(output_meta)

    attempt_dir = base_dir / "attempt1"
    attempt_dir.mkdir(parents=True, exist_ok=True)
    candidate_image = attempt_dir / "candidate.png"
    candidate_image.write_bytes(b"")
    (attempt_dir / "content.txt").write_text("There are seven circles here.", encoding="utf-8")

    result = evaluator.evaluate("test", candidate_image)

    assert result.predicted_count == 7
    assert result.circle_count == 7
    assert result.is_correct
    assert result.video_path is None
