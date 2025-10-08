import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence

# Ensure repository root is on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from puzzle.arcagi import ArcPuzzleEvaluator
from veo3 import generate_video_output_multiple_tries


def read_metadata(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Metadata not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError("Metadata must be a list of puzzle records")
    return payload


def sanitize(component: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in component.strip())
    return safe or "value"


def prepare_run_dir(puzzle_id: str, base_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_dir / f"arcagi_{sanitize(puzzle_id)}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def resolve_image_path(metadata_path: Path, relative_path: str) -> Path:
    return (metadata_path.parent / relative_path).resolve()


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def process_puzzle(
    evaluator: ArcPuzzleEvaluator,
    record: Dict[str, Any],
    metadata_path: Path,
    attempts: int,
    vote_root: Path,
) -> List[Dict[str, Any]]:
    puzzle_id = record.get("id") or ""
    prompt = (record.get("prompt") or "").strip()
    if not puzzle_id:
        raise ValueError("Puzzle record missing 'id'")
    if not prompt:
        raise ValueError(f"Puzzle {puzzle_id} has no prompt text")

    puzzle_img_rel = record.get("puzzle_image_path")
    if not isinstance(puzzle_img_rel, str) or not puzzle_img_rel:
        raise ValueError(f"Puzzle {puzzle_id} missing puzzle_image_path")
    puzzle_img = resolve_image_path(metadata_path, puzzle_img_rel)
    if not puzzle_img.exists():
        raise FileNotFoundError(f"Puzzle image not found: {puzzle_img}")

    run_dir = prepare_run_dir(puzzle_id, vote_root)

    results: List[Dict[str, Any]] = []
    for attempt in range(1, attempts + 1):
        output_dir = Path(generate_video_output_multiple_tries(puzzle_img.as_posix(), prompt)).resolve()
        result_png = output_dir / "result.png"
        if not result_png.exists():
            raise FileNotFoundError(f"Expected result frame not found at {result_png}")

        eval_result = evaluator.evaluate(puzzle_id, result_png)
        eval_dict = eval_result.to_dict()

        attempt_dir = run_dir / f"attempt_{attempt:02d}"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        vote_result_png = attempt_dir / "result.png"
        shutil.copy2(result_png, vote_result_png)

        evaluation_record: Dict[str, Any] = {
            "attempt": attempt,
            "puzzle_id": puzzle_id,
            "output_directory": output_dir.as_posix(),
            "result_png": result_png.as_posix(),
            "vote_run_directory": run_dir.as_posix(),
            "vote_output_directory": attempt_dir.as_posix(),
            "vote_result_png": vote_result_png.as_posix(),
            "evaluation": eval_dict,
        }

        write_json(output_dir / "evaluation.json", evaluation_record)
        write_json(attempt_dir / "evaluation.json", evaluation_record)
        results.append(evaluation_record)

    # also write a run manifest
    write_json(run_dir / "run_manifest.json", {"puzzle_id": puzzle_id, "attempts": attempts, "results": results})
    return results


def parse_args(argv = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run k generations per ARC-AGI puzzle for a range of indices and evaluate each.")
    parser.add_argument("m", type=int, help="1-based start index (inclusive)")
    parser.add_argument("n", type=int, help="1-based end index (inclusive)")
    parser.add_argument("k", type=int, help="Number of responses per puzzle")
    parser.add_argument("--metadata", type=Path, default=Path("data/arcagi/puzzles.json"))
    parser.add_argument("--vote-root", type=Path, default=Path("data/voteOutputArcagi"))
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv = None) -> None:
    args = parse_args(argv)
    if args.m <= 0 or args.n <= 0 or args.k <= 0:
        raise ValueError("m, n, k must be positive integers")
    if args.n < args.m:
        raise ValueError("n must be >= m")

    metadata_path = args.metadata.resolve()
    records = read_metadata(metadata_path)

    start_idx = args.m - 1
    end_idx = args.n
    if start_idx >= len(records):
        raise IndexError(f"Start index {args.m} exceeds number of records {len(records)}")
    slice_records = records[start_idx:end_idx]
    if not slice_records:
        raise IndexError("Empty slice; check m and n range")

    evaluator = ArcPuzzleEvaluator(metadata_path)
    vote_root = args.vote_root.resolve()
    vote_root.mkdir(parents=True, exist_ok=True)

    for idx, record in enumerate(slice_records, start=args.m):
        puzzle_id = record.get("id", "?")
        print(f"[{idx}/{len(records)}] Processing puzzle {puzzle_id} with {args.k} attempt(s)...")
        process_puzzle(evaluator, record, metadata_path, attempts=args.k, vote_root=vote_root)
    print("Done.")


if __name__ == "__main__":
    main()
