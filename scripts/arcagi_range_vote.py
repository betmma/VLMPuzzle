import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Counter, Dict, List, Sequence, Optional
import numpy as np
from PIL import Image

# Ensure repository root is on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from puzzle.arcagi import ArcPuzzleEvaluator
from veo3 import generate_video_output_multiple_tries, generate_video_outputs_multiprocess


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

    puzzle_img_rel = record.get("image")
    if not isinstance(puzzle_img_rel, str) or not puzzle_img_rel:
        raise ValueError(f"Puzzle {puzzle_id} missing image")
    puzzle_img = resolve_image_path(metadata_path, puzzle_img_rel)
    if not puzzle_img.exists():
        raise FileNotFoundError(f"Puzzle image not found: {puzzle_img}")

    run_dir = prepare_run_dir(puzzle_id, vote_root)

    results: List[Dict[str, Any]] = []

    # Run k attempts in parallel; each attempt internally retries (default 3)
    image_paths_list = [puzzle_img.as_posix()] * attempts
    prompt_texts = [prompt] * attempts
    output_dirs = generate_video_outputs_multiprocess(
        image_paths_list,
        prompt_texts,
        # keep per-attempt internal retries consistent with sequential path
        attempts=3,
    )

    for attempt_idx, out_dir_str in enumerate(output_dirs, start=1):
        output_dir = Path(out_dir_str).resolve()
        result_png = output_dir / "result.png"
        if not result_png.exists():
            raise FileNotFoundError(f"Expected result frame not found at {result_png}")

        eval_result = evaluator.evaluate(puzzle_id, result_png)
        eval_dict = eval_result.to_dict()

        attempt_dir = run_dir / f"attempt_{attempt_idx:02d}"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        vote_result_png = attempt_dir / "result.png"
        shutil.copy2(result_png, vote_result_png)

        evaluation_record: Dict[str, Any] = {
            "attempt": attempt_idx,
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


def _iter_attempt_evaluations(vote_root: Path, allowed_ids: Optional[set] = None) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if not vote_root.exists():
        return records
    for attempt_eval in vote_root.rglob("attempt_*/evaluation.json"):
        try:
            with attempt_eval.open("r", encoding="utf-8") as handle:
                rec = json.load(handle)
            if isinstance(rec, dict):
                pid = str(rec.get("puzzle_id") or "")
                if allowed_ids and pid not in allowed_ids:
                    continue
                records.append(rec)
        except Exception:
            continue
    return records


def summarize_vote_root(
    vote_root: Path,
    metadata_path: Path,
    *,
    no_change_threshold: float = 5.0,
    allowed_ids: Optional[set] = None,
) -> Dict[str, Any]:
    """Re-evaluate all attempts under vote_root and compute summary.

    - Re-runs ArcPuzzleEvaluator against each attempt's result.png.
    - Counts an attempt as correct only if all cells are correct.
    - Reports attempt-level perfect ratio and per-puzzle success rate.
    - Updates each attempt's evaluation.json with the recomputed evaluation.
    """
    evals = _iter_attempt_evaluations(vote_root, allowed_ids)
    evaluator = ArcPuzzleEvaluator(metadata_path)

    total_attempts = 0
    perfect_attempts = 0
    no_change_attempts = 0  # attempts where output area equals original puzzle input area
    by_puzzle: Dict[str, Dict[str, Any]] = {}
    acc_counts = Counter()

    for rec in evals:
        puzzle_id = str(rec.get("puzzle_id", ""))
        result_png = rec.get("result_png") or ""
        result_path = Path(result_png)

        result=rec['evaluation']

        total_attempts += 1
        correct = int(result['correct_cells'])
        total = int(result['total_cells'])
        is_perfect = total > 0 and (correct == total)
        if is_perfect:
            perfect_attempts += 1
        
        acc = round(result['accuracy'],1)
        acc_counts[acc] += 1
        # if acc>0.9:
        #     print(f"High accuracy {acc} for puzzle {puzzle_id} at {result_path}")
        #     result_dir = result_path.parent
        #     shutil.copytree(result_dir, vote_root.parent / "arcagi2_high_accuracy" / puzzle_id / result_dir.name)

        # Update per-puzzle aggregation
        if puzzle_id not in by_puzzle:
            by_puzzle[puzzle_id] = {"attempts": 0, "perfect": False}
        by_puzzle[puzzle_id]["attempts"] += 1
        by_puzzle[puzzle_id]["perfect"] = by_puzzle[puzzle_id]["perfect"] or is_perfect

        # Detect attempts that changed nothing inside the designed test output area
        # no_change=all(all(i==5 for i in j)for j in result['predicted_grid']) # white is predicted as 5, so all white = all 5 means no change
        try:
            record = evaluator.get_record(puzzle_id)
            puzzle_img_path = evaluator.resolve_path(record.get("image"))
            puzzle_img = Image.open(puzzle_img_path).convert("RGB")
            candidate_img = Image.open(result_path).convert("RGB")
            # Align candidate to the puzzle composite size
            candidate_aligned = evaluator._align(candidate_img, puzzle_img.size, trim_tolerance=12)  # type: ignore[attr-defined]
            # Locate designed output area (test_output bbox)
            placements = record.get("placements")
            x0, y0, x1, y1 = evaluator._find_test_bbox(placements)  # type: ignore[attr-defined]
            crop_puzzle = puzzle_img.crop((x0, y0, x1, y1))
            crop_candidate = candidate_aligned.crop((x0, y0, x1, y1))
            a = np.asarray(crop_puzzle)
            b = np.asarray(crop_candidate)
            if a.shape == b.shape and a.size:
                # Average per-pixel RGB Euclidean distance
                diff = a.astype(np.float32) - b.astype(np.float32)
                dist = np.sqrt(np.sum(diff * diff, axis=2))
                mean_dist = float(dist.mean())
                # print(mean_dist)
                no_change = mean_dist <= float(no_change_threshold)
            else:
                no_change = False
        except Exception as e:
            print(e)
            no_change = False
        if no_change:
            no_change_attempts += 1

    total_puzzles = len([p for p in by_puzzle.keys() if p])
    puzzles_with_perfect = sum(1 for p in by_puzzle.values() if p.get("perfect"))
    attempt_accuracy = (perfect_attempts / total_attempts) if total_attempts else 0.0
    puzzle_success_rate = (puzzles_with_perfect / total_puzzles) if total_puzzles else 0.0
    puzzle_average_accuracy = sum(rec['evaluation']['accuracy'] for rec in evals) / total_attempts if total_attempts else 0.0

    acc_counts = dict(sorted(acc_counts.items()))
    return {
        "vote_root": vote_root.as_posix(),
        "attempts_total": total_attempts,
        "attempts_perfect": perfect_attempts,
        "attempts_average_correctness": attempt_accuracy,
        "attempts_no_change_output_area": no_change_attempts,
        "attempts_no_change_ratio": (no_change_attempts / total_attempts) if total_attempts else 0.0,
        "puzzles_total": total_puzzles,
        "puzzles_with_any_perfect_attempt": puzzles_with_perfect,
        "puzzle_success_rate": puzzle_success_rate,
        "accuracy_counts": dict(acc_counts),
        "puzzle_average_accuracy": puzzle_average_accuracy,
    }


def parse_args(argv = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run k generations per ARC-AGI puzzle for a range of indices and evaluate each.")
    parser.add_argument("m", type=int, help="1-based start index (inclusive)")
    parser.add_argument("n", type=int, help="1-based end index (inclusive)")
    parser.add_argument("k", type=int, help="Number of responses per puzzle")
    parser.add_argument("--metadata", type=Path, default=Path("data/arcagi/data.json"))
    parser.add_argument("--vote-root", type=Path, default=Path("data/voteOutputArcagi"))
    parser.add_argument("--summarize", action="store_true", help="Only summarize existing evaluations under --vote-root and exit.")
    parser.add_argument(
        "--no-change-threshold",
        type=float,
        default=50.0,
        help="Average per-pixel RGB Euclidean distance threshold to count an attempt as 'no change' in the designed output area",
    )
    parser.add_argument("--processes", type=int, default=None, help="Worker processes for parallelizing across puzzles")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv = None) -> None:
    args = parse_args(argv)
    # If summarizing, re-evaluate and print summary then exit
    if args.summarize:
        metadata_path = args.metadata.resolve()
        records = read_metadata(metadata_path)
        # Validate and slice range [m..n]
        if args.m <= 0 or args.n <= 0:
            raise ValueError("m and n must be positive integers for summarization range filtering")
        if args.n < args.m:
            raise ValueError("n must be >= m")
        start_idx = args.m - 1
        end_idx = args.n
        if start_idx >= len(records):
            raise IndexError(f"Start index {args.m} exceeds number of records {len(records)}")
        slice_records = records[start_idx:end_idx]
        if not slice_records:
            raise IndexError("Empty slice; check m and n range")
        allowed_ids = {str(r.get("id")) for r in slice_records if r.get("id")}
        summary = summarize_vote_root(
            args.vote_root.resolve(),
            metadata_path,
            no_change_threshold=args.no_change_threshold,
            allowed_ids=allowed_ids,
        )
        print(json.dumps(summary, indent=2))
        return
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

    # Prepare per-puzzle run directories once
    run_dirs: Dict[str, Path] = {}
    for record in slice_records:
        pid = str(record.get("id") or "")
        if not pid:
            raise ValueError("Record missing 'id'")
        run_dirs[pid] = prepare_run_dir(pid, vote_root)

    # Accumulate results per puzzle to write manifests at the end
    per_puzzle_results: Dict[str, List[Dict[str, Any]]] = {str(r.get("id")): [] for r in slice_records}

    # Resolve image paths and prompts once
    imgs: List[str] = []
    prompts: List[str] = []
    ids: List[str] = []
    for record in slice_records:
        pid = str(record.get("id") or "")
        prompt = (record.get("prompt") or "").strip()
        if not prompt:
            raise ValueError(f"Puzzle {pid} has no prompt text")
        img_rel = record.get("image")
        if not isinstance(img_rel, str) or not img_rel:
            raise ValueError(f"Puzzle {pid} missing image")
        img_path = resolve_image_path(metadata_path, img_rel)
        if not img_path.exists():
            raise FileNotFoundError(f"Puzzle image not found: {img_path}")
        imgs.append(img_path.as_posix())
        prompts.append(prompt)
        ids.append(pid)

    total = len(ids)
    for attempt_idx in range(1, args.k + 1):
        print(f"Batch generating attempt {attempt_idx}/{args.k} for {total} puzzles...")
        outs = generate_video_outputs_multiprocess(
            imgs,
            prompts,
            processes=args.processes,
            attempts=3,  # internal retries per item
        )
        for i, out_dir in enumerate(outs):
            pid = ids[i]
            output_dir = Path(out_dir).resolve()
            result_png = output_dir / "result.png"
            if not result_png.exists():
                raise FileNotFoundError(f"Expected result frame not found at {result_png}")
            eval_result = evaluator.evaluate(pid, result_png)
            eval_dict = eval_result.to_dict()

            attempt_dir = run_dirs[pid] / f"attempt_{attempt_idx:02d}"
            attempt_dir.mkdir(parents=True, exist_ok=True)
            vote_result_png = attempt_dir / "result.png"
            shutil.copy2(result_png, vote_result_png)

            evaluation_record: Dict[str, Any] = {
                "attempt": attempt_idx,
                "puzzle_id": pid,
                "output_directory": output_dir.as_posix(),
                "result_png": result_png.as_posix(),
                "vote_run_directory": run_dirs[pid].as_posix(),
                "vote_output_directory": attempt_dir.as_posix(),
                "vote_result_png": vote_result_png.as_posix(),
                "evaluation": eval_dict,
            }
            write_json(output_dir / "evaluation.json", evaluation_record)
            write_json(attempt_dir / "evaluation.json", evaluation_record)
            per_puzzle_results[pid].append(evaluation_record)

    # Write per-puzzle run manifests
    for pid, results in per_puzzle_results.items():
        write_json(run_dirs[pid] / "run_manifest.json", {"puzzle_id": pid, "attempts": args.k, "results": results})
    print("Done.")


if __name__ == "__main__":
    main()
