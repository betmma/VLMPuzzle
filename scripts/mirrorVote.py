import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

from veo3 import generate_video_output_multiple_tries
from gpt5 import generate_multiple_tries

def _load_puzzle(puzzles_path: str, puzzle_id: str) -> Dict[str, Any]:
    with open(puzzles_path, "r", encoding="utf-8") as handle:
        puzzles = json.load(handle)
    for puzzle in puzzles:
        if puzzle.get("id") == puzzle_id:
            return puzzle
    raise ValueError(f"Puzzle {puzzle_id} not found in {puzzles_path}")


def _resolve_image_path(puzzles_path: str, image: str) -> str:
    puzzles_dir = os.path.dirname(puzzles_path)
    full_path = os.path.join(puzzles_dir, image)
    return os.path.abspath(full_path)


def _write_evaluation(output_dir: str, evaluation_record: Dict[str, Any]) -> None:
    destination = os.path.join(output_dir, "evaluation.json")
    with open(destination, "w", encoding="utf-8") as report:
        json.dump(evaluation_record, report, ensure_ascii=False, indent=2)


def _sanitize_path_component(component: str) -> str:
    safe = "".join(
        char if char.isalnum() or char in ("-", "_") else "_" for char in component.strip()
    )
    return safe or "value"


def _derive_puzzle_type(puzzle: Dict[str, Any], puzzles_path: str) -> str:
    puzzle_type = puzzle.get("type")
    if isinstance(puzzle_type, str) and puzzle_type.strip():
        return puzzle_type.strip()

    global_type = globals().get("PUZZLE_TYPE")
    if isinstance(global_type, str) and global_type.strip():
        return global_type.strip()

    parent_dir = os.path.basename(os.path.dirname(os.path.abspath(puzzles_path)))
    return parent_dir or "puzzle"


def _prepare_vote_run_dir(puzzle_type: str, puzzle_id: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.join("data", "voteOutput")
    os.makedirs(base_dir, exist_ok=True)

    type_component = _sanitize_path_component(puzzle_type)
    id_component = _sanitize_path_component(puzzle_id)
    run_dir_name = f"{type_component}_{id_component}_{timestamp}"
    vote_run_dir = os.path.join(base_dir, run_dir_name)
    os.makedirs(vote_run_dir, exist_ok=True)
    return vote_run_dir


def run_generations_for_puzzle(
    puzzle_id: str,
    attempts: int = 3,
    puzzles_path: str = "data/mirror/data.json",
    use_gpt_5: bool = False,
    puzzle_type_override: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Generate multiple video attempts for a mirror puzzle and evaluate each result."""
    puzzle = _load_puzzle(puzzles_path, puzzle_id)
    image_path = _resolve_image_path(puzzles_path, puzzle["image"])
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Puzzle image not found at {image_path}")

    key='prompt'
    if use_gpt_5 and 'gpt5_prompt' in puzzle:
        key='gpt5_prompt'
    prompt = puzzle.get(key, "").strip()
    if not prompt:
        raise ValueError(f"Puzzle {puzzle_id} has no prompt text")

    fallback_type = _derive_puzzle_type(puzzle, puzzles_path)
    effective_type = puzzle_type_override or globals().get("PUZZLE_TYPE") or fallback_type

    vote_run_dir = _prepare_vote_run_dir(effective_type, puzzle_id)

    results: List[Dict[str, Any]] = []
    for attempt in range(1, attempts + 1):
        if use_gpt_5:
            output_dir = generate_multiple_tries(image_path, prompt)
            result_png = os.path.join(output_dir, "result.png")
        else:
            output_dir = generate_video_output_multiple_tries(image_path, prompt)
            result_png = os.path.join(output_dir, "result.png")
            if not os.path.exists(result_png):
                raise FileNotFoundError(f"Expected result frame not found at {result_png}")

        command = [
            sys.executable,
            "-m",
            f"puzzle.{effective_type}.evaluator",
            puzzles_path,
            puzzle_id,
            result_png,
        ]
        completed = subprocess.run(command, capture_output=True, text=True)

        attempt_dir = os.path.join(vote_run_dir, f"attempt_{attempt:02d}")
        os.makedirs(attempt_dir, exist_ok=True)
        if use_gpt_5:
            vote_result_png = os.path.join(attempt_dir, "result.png")
        else:
            vote_result_png = os.path.join(attempt_dir, "result.png")
            shutil.copy2(result_png, vote_result_png)

        evaluation_record: Dict[str, Any] = {
            "attempt": attempt,
            "output_directory": output_dir,
            "result_png": result_png,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "vote_run_directory": vote_run_dir,
            "vote_output_directory": attempt_dir,
            "vote_result_png": vote_result_png,
        }
        _write_evaluation(output_dir, evaluation_record)
        _write_evaluation(attempt_dir, evaluation_record)
        results.append(evaluation_record)

    return results


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python scripts/mirrorVote.py <puzzle_type> <puzzle_id> [attempts] [puzzles_path] [use_gpt_5]")
        sys.exit(1)

    PUZZLE_TYPE = sys.argv[1]
    puzzle_id_arg = sys.argv[2]
    attempts_arg = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    puzzles_path_arg = sys.argv[4] if len(sys.argv) > 4 else f"data/{PUZZLE_TYPE}/data.json"
    use_gpt_5 = sys.argv[5].lower() == "true" if len(sys.argv) > 5 else False

    all_results = run_generations_for_puzzle(
        puzzle_id=puzzle_id_arg,
        attempts=attempts_arg,
        puzzles_path=puzzles_path_arg,
        use_gpt_5=use_gpt_5,
        puzzle_type_override=PUZZLE_TYPE,
    )

    for result in all_results:
        print(f"Attempt {result['attempt']} - Return code: {result['returncode']}")
        print(f"Stdout: {result['stdout']}")
        print(f"Stderr: {result['stderr']}")
        print(f"Result PNG: {result['result_png']}")
        print(f"Vote Result PNG: {result['vote_result_png']}")
        print(f"Output Directory: {result['output_directory']}")
        print(f"Vote Output Directory: {result['vote_output_directory']}")
        print(f"Vote Run Directory: {result['vote_run_directory']}")
        print("-" * 40)
