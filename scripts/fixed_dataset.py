import argparse
import json
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import mirrorVote


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


def _run_evaluation_job(task: Tuple[str, str, int, str, bool]) -> Dict[str, object]:
    puzzle_type, puzzle_id, attempts, puzzles_path, use_gpt_5 = task
    mirrorVote.PUZZLE_TYPE = puzzle_type
    results = mirrorVote.run_generations_for_puzzle(
        puzzle_id=puzzle_id,
        attempts=attempts,
        puzzles_path=puzzles_path,
        use_gpt_5=use_gpt_5,
    )
    return {
        "puzzle_type": puzzle_type,
        "puzzle_id": puzzle_id,
        "attempts": attempts,
        "puzzles_path": puzzles_path,
        "use_gpt_5": use_gpt_5,
        "results": results,
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
        "--use-gpt-5",
        action="store_true",
        help="Use gpt-5 for answer generation instead of veo3.",
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
    if not jobs:
        print("No puzzles matched selection.")
        return

    per_type_results: Dict[str, List[Dict[str, object]]] = {key: [] for key in puzzle_dirs}
    if args.workers > 1:
        with mp.Pool(processes=args.workers) as pool:
            for summary in pool.imap_unordered(_run_evaluation_job, jobs):
                per_type_results[summary["puzzle_type"]].append(summary)
                print(f"{summary['puzzle_type']} {summary['puzzle_id']} complete.")
    else:
        for job in jobs:
            summary = _run_evaluation_job(job)
            per_type_results[summary["puzzle_type"]].append(summary)
            print(f"{summary['puzzle_type']} {summary['puzzle_id']} complete.")

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