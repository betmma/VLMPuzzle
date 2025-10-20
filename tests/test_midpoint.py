import json

import cv2
import numpy as np

from puzzle.midpoint import MidpointEvaluator, MidpointGenerator


def test_midpoint_generator_outputs(tmp_path):
    output_root = tmp_path / "midpoint_out"
    generator = MidpointGenerator(output_root, seed=123)
    record = generator.create_puzzle(puzzle_id="unit-test")

    labels = [candidate.label for candidate in record.candidates]
    assert sorted(labels) == list("ABCDE")
    assert record.correct_option in labels
    assert record.point_radius > 0

    canvas_width, canvas_height = record.canvas_dimensions
    for candidate in record.candidates:
        assert 0 <= candidate.x <= canvas_width
        assert 0 <= candidate.y <= canvas_height

    start = record.segment.start
    end = record.segment.end
    midpoint = record.midpoint
    mid_x = (start[0] + end[0]) / 2.0
    mid_y = (start[1] + end[1]) / 2.0
    assert abs(mid_x - midpoint[0]) < 1e-3
    assert abs(mid_y - midpoint[1]) < 1e-3

    puzzle_path = output_root / record.image
    solution_path = output_root / record.solution_image_path
    assert puzzle_path.exists()
    assert solution_path.exists()


def test_midpoint_evaluator_end_to_end(tmp_path):
    meta_dir = tmp_path / "meta"
    meta_dir.mkdir()
    metadata_path = meta_dir / "metadata.json"
    record = {
        "id": "demo",
        "correct_option": "C",
        "canvas_dimensions": [240, 240],
        "point_radius": 12,
        "midpoint": [120.0, 120.0],
        "segment": {
            "start": [60.0, 120.0],
            "end": [180.0, 120.0],
        },
        "candidates": [
            {"label": "A", "x": 100.0, "y": 120.0},
            {"label": "B", "x": 120.0, "y": 100.0},
            {"label": "C", "x": 120.0, "y": 120.0},
            {"label": "D", "x": 140.0, "y": 120.0},
            {"label": "E", "x": 120.0, "y": 140.0},
        ],
        "image": "puzzles/demo.png",
        "solution_image_path": "solutions/demo.png",
        "prompt": "dummy",
        "margin": 12,
        "type": "midpoint",
    }
    metadata_path.write_text(json.dumps([record]), encoding="utf-8")
    evaluator = MidpointEvaluator(metadata_path)

    attempt_dir = tmp_path / "attempt"
    attempt_dir.mkdir()
    candidate_path = attempt_dir / "candidate.png"

    frame = np.zeros((240, 240, 3), dtype=np.uint8)
    cv2.circle(frame, (120, 120), 14, (0, 0, 255), -1)
    cv2.imwrite(candidate_path.as_posix(), frame)

    (attempt_dir / "content.txt").write_text("Answer is Charlie", encoding="utf-8")

    result = evaluator.evaluate("demo", candidate_path)

    assert result.correct_option == "C"
    assert result.image_option == "C"
    assert result.video_option is None
    assert result.transcribe_option is None
    assert result.text_option == "C"
    assert getattr(result, "red_pixel_count", 0) >= 20
    assert getattr(result, "red_centroid", None) is not None