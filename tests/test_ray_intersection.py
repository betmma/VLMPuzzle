import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from puzzle.ray_intersection import (
    RayIntersectionEvaluator,
    RayIntersectionGenerator,
)


def test_ray_intersection_generator_outputs():
    output_root = Path("debug/test_ray_intersection_generator")
    if output_root.exists():
        shutil.rmtree(output_root)
    generator = RayIntersectionGenerator(output_root, seed=123)
    record = generator.create_puzzle(puzzle_id="unit-test")

    labels = [candidate.label for candidate in record.candidates]
    assert sorted(labels) == list("ABCDE")
    assert record.correct_option in labels
    assert record.point_radius > 0
    canvas_width, canvas_height = record.canvas_dimensions
    for candidate in record.candidates:
        assert 0 <= candidate.x <= canvas_width
        assert 0 <= candidate.y <= canvas_height
    puzzle_path = output_root / record.puzzle_image_path
    solution_path = output_root / record.solution_image_path
    assert puzzle_path.exists()
    assert solution_path.exists()


def test_ray_intersection_evaluator_red_detection(tmp_path):
    meta_dir = Path("debug/test_ray_intersection_meta")
    if meta_dir.exists():
        shutil.rmtree(meta_dir)
    meta_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = meta_dir / "metadata.json"
    record = {
        "id": "demo",
        "correct_option": "C",
        "canvas_dimensions": [240, 240],
        "point_radius": 20,
        "candidates": [
            {"label": "A", "x": 80.0, "y": 120.0},
            {"label": "B", "x": 120.0, "y": 80.0},
            {"label": "C", "x": 120.0, "y": 120.0},
            {"label": "D", "x": 160.0, "y": 120.0},
            {"label": "E", "x": 120.0, "y": 160.0},
        ],
        "rays": [],
        "intersection": [120.0, 120.0],
        "margin": 12,
        "prompt": "dummy",
        "puzzle_image_path": "puzzles/x.png",
        "solution_image_path": "solutions/x.png",
    }
    metadata_path.write_text(json.dumps([record]), encoding="utf-8")
    evaluator = RayIntersectionEvaluator(metadata_path)

    frame = np.zeros((240, 240, 3), dtype=np.uint8)
    cv2.circle(frame, (120, 120), 16, (255, 10, 10), -1)
    predicted, scores = evaluator._score_red_hits(frame, record)
    assert predicted == "C"
    assert scores.get("C", 0.0) > 0.5


def test_ray_intersection_evaluate_without_video(tmp_path):
    meta_dir = Path("debug/test_ray_intersection_meta")
    metadata_path = meta_dir / "metadata.json"
    evaluator = RayIntersectionEvaluator(metadata_path)

    attempt_dir = Path("debug/test_ray_intersection_attempt")
    if attempt_dir.exists():
        shutil.rmtree(attempt_dir)
    attempt_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = attempt_dir / "candidate.png"
    candidate_path.write_bytes(b"")
    (attempt_dir / "content.txt").write_text("Answer is Charlie", encoding="utf-8")

    result = evaluator.evaluate("demo", candidate_path)
    assert result.transcript_option == "C"
    assert result.transcript_is_correct is True
    assert result.red_scores == {}
    assert result.red_is_correct is False
    assert result.combined_is_correct is False
