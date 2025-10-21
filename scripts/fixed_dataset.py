import argparse
import json
import multiprocessing as mp
import subprocess
import sys
import time, os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple


def _discover_puzzle_types(dataset_root: Path, requested: Sequence[str]) -> List[str]:
    if requested:
        return list(dict.fromkeys(requested))
    return sorted(
        entry.name
        for entry in dataset_root.iterdir()
        if entry.is_dir() and (entry / "data.json").is_file()
    )


def _load_metadata(metadata_path: Path) -> List[Dict[str, object]]:
    with metadata_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"Metadata at {metadata_path} must be a list")
    return payload


def _select_puzzles_path(puzzle_root: Path) -> Path:
    candidates = (
        puzzle_root / "data.json",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No puzzles file found under {puzzle_root}")


def _collect_jobs(
    dataset_root: Path,
    puzzle_types: Sequence[str],
    attempts: int,
    use_gpt_5: bool,
    allowed_ids: Sequence[str],
) -> Tuple[List[Tuple[str, str, int, str, bool]], Dict[str, Path]]:
    allowed = set(allowed_ids)
    jobs: List[Tuple[str, str, int, str, bool]] = []
    puzzle_dirs: Dict[str, Path] = {}
    for puzzle_type in puzzle_types:
        puzzle_dir = dataset_root / puzzle_type
        metadata_path = puzzle_dir / "data.json"
        metadata = _load_metadata(metadata_path)
        puzzles_path = _select_puzzles_path(puzzle_dir)
        for entry in metadata:
            if not isinstance(entry, dict):
                raise ValueError(f"Metadata entry in {metadata_path} must be an object")
            puzzle_id = entry.get("id")
            if not isinstance(puzzle_id, str) or not puzzle_id:
                raise ValueError(f"Puzzle entry in {metadata_path} missing a valid id")
            if allowed and puzzle_id not in allowed:
                continue
            jobs.append((puzzle_type, puzzle_id, attempts, str(puzzles_path), use_gpt_5))
        puzzle_dirs[puzzle_type] = puzzle_dir
    return jobs, puzzle_dirs


def _sanitize_vote_component(text: str) -> str:
    cleaned = "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in text.strip()
    )
    return cleaned or "value"


def _build_vote_key(puzzle_type: str, puzzle_id: str) -> str:
    return _sanitize_vote_component(puzzle_type) + "_" + _sanitize_vote_component(puzzle_id)


def _discover_completed_puzzles(vote_output_root: Path) -> Set[str]:
    completed: Set[str] = set()
    if not vote_output_root.exists() or not vote_output_root.is_dir():
        return completed
    for run_dir in vote_output_root.iterdir():
        if not run_dir.is_dir():
            continue
        parts = run_dir.name.rsplit("_", 2)
        if len(parts) != 3:
            continue
        prefix = parts[0]
        if prefix and os.listdir(run_dir): # not empty
            completed.add(prefix)
    return completed


def _run_evaluation_job(task: Tuple[str, str, int, str, bool], timeout: Optional[float]) -> Dict[str, object]:
    puzzle_type, puzzle_id, attempts, puzzles_path, use_gpt_5 = task
    command = [
        sys.executable,
        "scripts/mirrorVote.py",
        puzzle_type,
        puzzle_id,
        str(attempts),
        puzzles_path,
        "true" if use_gpt_5 else "false",
    ]
    start = time.monotonic()
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    duration = time.monotonic() - start
    return {
        "puzzle_type": puzzle_type,
        "puzzle_id": puzzle_id,
        "attempts": attempts,
        "puzzles_path": puzzles_path,
        "use_gpt_5": use_gpt_5,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "command": command,
        "elapsed_seconds": duration,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate model answers on a fixed puzzle dataset without generating new puzzles."
    )
    parser.add_argument("puzzle_types", nargs="*", help="Subset of puzzle types to evaluate.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("dataset"),
        help="Root directory containing fixed puzzle datasets.",
    )
    parser.add_argument(
        "--attempts",
        type=int,
        default=1,
        help="Number of generations per puzzle.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel worker count for model runs.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Per-puzzle timeout in seconds for mirrorVote (default: no timeout).",
    )
    parser.add_argument(
        "--use-gpt-5",
        action="store_true",
        help="Use gpt-5 for answer generation instead of veo3.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip puzzles that already have vote outputs under data/voteOutput.",
    )
    parser.add_argument(
        "--vote-output-root",
        type=Path,
        default=Path("data") / "voteOutput",
        help="Directory containing vote outputs for previously evaluated puzzles.",
    )
    parser.add_argument(
        "--puzzle-id",
        action="append",
        default=[],
        help="Limit evaluation to specific puzzle id (repeatable).",
    )
    args = parser.parse_args()

    dataset_root = args.dataset_root
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root {dataset_root} does not exist")

    puzzle_types = _discover_puzzle_types(dataset_root, args.puzzle_types)
    if not puzzle_types:
        print("No puzzle types to evaluate.")
        return

    jobs, puzzle_dirs = _collect_jobs(
        dataset_root,
        puzzle_types,
        args.attempts,
        args.use_gpt_5,
        args.puzzle_id,
    )
    if args.resume:
        completed = _discover_completed_puzzles(args.vote_output_root)
        filtered_jobs = []
        skipped = 0
        for job in jobs:
            puzzle_type, puzzle_id = job[0], job[1]
            vote_key = _build_vote_key(puzzle_type, puzzle_id)
            if vote_key in completed:
                skipped += 1
                continue
            filtered_jobs.append(job)
        jobs = filtered_jobs
        if skipped:
            print(f"Skipping {skipped} already evaluated puzzle(s).")
    if not jobs:
        print("No puzzles matched selection.")
        return

    per_type_results: Dict[str, List[Dict[str, object]]] = {key: [] for key in puzzle_dirs}
    if args.workers > 1:
        pool_size = min(args.workers, len(jobs))
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=pool_size, mp_context=ctx) as executor:
            futures = [executor.submit(_run_evaluation_job, job, args.timeout) for job in jobs]
            completed = 0
            total = len(futures)
            for future in as_completed(futures):
                summary = future.result()
                per_type_results[summary["puzzle_type"]].append(summary)
                completed += 1
                status = "ok" if summary["returncode"] == 0 else f"exit {summary['returncode']}"
                print(f"{summary['puzzle_type']} {summary['puzzle_id']} {status}. ({completed}/{total})")
    else:
        for job in jobs:
            summary = _run_evaluation_job(job, args.timeout)
            per_type_results[summary["puzzle_type"]].append(summary)
            status = "ok" if summary["returncode"] == 0 else f"exit {summary['returncode']}"
            print(f"{summary['puzzle_type']} {summary['puzzle_id']} {status}.")

    # for puzzle_type, summaries in per_type_results.items():
    #     if not summaries:
    #         continue
    #     summaries.sort(key=lambda item: item["puzzle_id"])
    #     summary_path = puzzle_dirs[puzzle_type] / "model_evaluation_summary.json"
    #     with summary_path.open("w", encoding="utf-8") as handle:
    #         json.dump(summaries, handle, ensure_ascii=False, indent=2)
    #     print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()