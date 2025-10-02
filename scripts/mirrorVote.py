import json
import os
import subprocess
import sys
from typing import Any, Dict, List

from veo3 import generate_video_output

PUZZLE_TYPE = 'mirror'
def _load_puzzle(puzzles_path: str, puzzle_id: str) -> Dict[str, Any]:
    with open(puzzles_path, "r", encoding="utf-8") as handle:
        puzzles = json.load(handle)
    for puzzle in puzzles:
        if puzzle.get("id") == puzzle_id:
            return puzzle
    raise ValueError(f"Puzzle {puzzle_id} not found in {puzzles_path}")


def _resolve_image_path(puzzles_path: str, puzzle_image_path: str) -> str:
    puzzles_dir = os.path.dirname(puzzles_path)
    full_path = os.path.join(puzzles_dir, puzzle_image_path)
    return os.path.abspath(full_path)


def _write_evaluation(output_dir: str, evaluation_record: Dict[str, Any]) -> None:
    destination = os.path.join(output_dir, "evaluation.json")
    with open(destination, "w", encoding="utf-8") as report:
        json.dump(evaluation_record, report, ensure_ascii=False, indent=2)


def run_generations_for_puzzle(
    puzzle_id: str,
    attempts: int = 3,
    puzzles_path: str = "data/mirror/puzzles.json",
) -> List[Dict[str, Any]]:
    """Generate multiple video attempts for a mirror puzzle and evaluate each result."""
    puzzle = _load_puzzle(puzzles_path, puzzle_id)
    image_path = _resolve_image_path(puzzles_path, puzzle["puzzle_image_path"])
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Puzzle image not found at {image_path}")

    prompt = puzzle.get("prompt", "").strip()
    if not prompt:
        raise ValueError(f"Puzzle {puzzle_id} has no prompt text")

    results: List[Dict[str, Any]] = []
    for attempt in range(1, attempts + 1):
        output_dir = generate_video_output(image_path, prompt)
        result_png = os.path.join(output_dir, "result.png")
        if not os.path.exists(result_png):
            raise FileNotFoundError(f"Expected result frame not found at {result_png}")

        command = [
            sys.executable,
            "-m",
            f"puzzle.{PUZZLE_TYPE}.evaluator",
            puzzles_path,
            puzzle_id,
            result_png,
        ]
        completed = subprocess.run(command, capture_output=True, text=True)

        evaluation_record: Dict[str, Any] = {
            "attempt": attempt,
            "output_directory": output_dir,
            "result_png": result_png,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
        _write_evaluation(output_dir, evaluation_record)
        results.append(evaluation_record)

    return results

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/mirrorVote.py <puzzle_id> [attempts] [puzzles_path]")
        sys.exit(1)

    puzzle_id_arg = sys.argv[1]
    attempts_arg = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    puzzles_path_arg = sys.argv[3] if len(sys.argv) > 3 else f"data/{PUZZLE_TYPE}/puzzles.json"

    all_results = run_generations_for_puzzle(
        puzzle_id=puzzle_id_arg,
        attempts=attempts_arg,
        puzzles_path=puzzles_path_arg,
    )

    for result in all_results:
        print(f"Attempt {result['attempt']} - Return code: {result['returncode']}")
        print(f"Stdout: {result['stdout']}")
        print(f"Stderr: {result['stderr']}")
        print(f"Result PNG: {result['result_png']}")
        print(f"Output Directory: {result['output_directory']}")
        print("-" * 40)