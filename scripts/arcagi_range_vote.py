import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Counter, Dict, List, Optional

import numpy as np
import re
from PIL import Image, ImageDraw

# Ensure repository root is on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from puzzle.arcagi import ArcPuzzleEvaluator
from puzzle.arcagi.evaluator import ARC_PALETTE
from veo3 import generate_video_output_multiple_tries, generate_video_outputs_multiprocess
from gpt5 import generate_multiple_tries as generate_gpt5_multiple_tries, generate_outputs_multiprocess

GPT5_PROMPT = (
    "Each row contains input and output grids. Learn the pattern and generate the output grid for the last input. "
    "Color palette: 0: black\n1: blue\n2: red\n3: green\n4: yellow\n5: gray\n6: magenta\n7: orange\n8: cyan\n9: brown. "
    "Output a row-major 2d array representing the output grid, with each element an integer from 0 to 9."
)


def _parse_bracket_grid(text: str) -> Optional[List[List[int]]]:
    # Use regex to extract the last [[...]] block (dot matches newlines)
    compact = text.replace(" ", "").replace("\n", "").replace("\r", "")
    matches = re.findall(r"\[\[.+?\]\]", compact, flags=re.DOTALL)
    if not matches:
        return None
    expr = matches[-1]
    try:
        grid = eval(expr)
    except Exception:
        return None

    return grid

def _copy_input_as_result(source_image: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_image, destination)
    return destination

def _render_grid_to_image(
    grid: List[List[int]],
    evaluator: ArcPuzzleEvaluator,
    record: Dict[str, Any],
    input_image: Path,
    destination: Path,
) -> Path:
    base_image = Image.open(input_image).convert("RGB")
    result_image = base_image.copy()
    placements = record["placements"]
    x0, y0, x1, y1 = evaluator._find_test_bbox(placements) 
    cell_size = record["cell_size"]
    draw = ImageDraw.Draw(result_image)
    for row_index, row in enumerate(grid):
        for col_index, value in enumerate(row):
            color = ARC_PALETTE.get(int(value), ARC_PALETTE[0])
            left = x0 + col_index * cell_size
            top = y0 + row_index * cell_size
            right = left + cell_size
            bottom = top + cell_size
            draw.rectangle(((left, top), (right - 1, bottom - 1)), fill=color)
    destination.parent.mkdir(parents=True, exist_ok=True)
    result_image.save(destination)
    return destination


def ensure_gpt5_result_image(
    evaluator: ArcPuzzleEvaluator,
    puzzle_id: str,
    output_dir: Path,
    input_image_path: Path,
) -> Path:
    output_dir = Path(output_dir)
    result_path = output_dir / "result.png"
    if result_path.exists():
        return result_path
    text_sources = [
        output_dir / "content.txt",
        output_dir / "original_content.txt",
    ]
    text_payload = ""
    for candidate in text_sources:
        if candidate.exists():
            payload = candidate.read_text(encoding="utf-8")
            if payload:
                text_payload = payload
                break
    record = evaluator.get_record(puzzle_id)
    input_image = Path(input_image_path)
    grid = _parse_bracket_grid(text_payload) 
    if grid is None:
        return _copy_input_as_result(input_image, result_path)
    return _render_grid_to_image(grid, evaluator, record, input_image, result_path)


def _extract_puzzle_id_from_input_file(input_file: Path) -> Optional[str]:
    if not input_file.exists():
        return None
    text = input_file.read_text(encoding="utf-8")
    marker = "Input image path:"
    marker_index = text.find(marker)
    if marker_index == -1:
        return None
    remaining = text[marker_index + len(marker):].lstrip()
    if not remaining:
        return None
    lines = remaining.splitlines()
    if not lines:
        return None
    path_text = lines[0].strip()
    if not path_text:
        return None
    puzzle_stem = Path(path_text).stem
    suffix = "_puzzle"
    if puzzle_stem.endswith(suffix):
        return puzzle_stem[: -len(suffix)]
    return puzzle_stem


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


def _gather_existing_output_dirs(existing_root: Path, allowed_ids: List[str]) -> Dict[str, List[Path]]:
    if not existing_root.exists():
        raise FileNotFoundError(f"Existing output root not found: {existing_root}")
    mapping: Dict[str, List[Path]] = {pid: [] for pid in allowed_ids}
    for entry in sorted(existing_root.iterdir()):
        if not entry.is_dir():
            continue
        input_file = entry / "input.txt"
        puzzle_id = _extract_puzzle_id_from_input_file(input_file)
        if puzzle_id is None:
            continue
        if puzzle_id in mapping:
            mapping[puzzle_id].append(entry)
    for pid in mapping:
        mapping[pid].sort()
    return mapping


def _write_evaluation_artifacts(
    evaluator: ArcPuzzleEvaluator,
    puzzle_id: str,
    result_png: Path,
    output_dir: Path,
    run_dir: Path,
    attempt_index: int,
) -> Dict[str, Any]:
    eval_result = evaluator.evaluate(puzzle_id, result_png)
    eval_dict = eval_result.to_dict()
    attempt_dir = run_dir / f"attempt_{attempt_index:02d}"
    attempt_dir.mkdir(parents=True, exist_ok=True)
    vote_result_png = attempt_dir / "result.png"
    shutil.copy2(result_png, vote_result_png)
    evaluation_record: Dict[str, Any] = {
        "attempt": attempt_index,
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
    return evaluation_record


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
            result_png = ensure_gpt5_result_image(
                evaluator,
                puzzle_id,
                output_dir,
                puzzle_img,
            )
        if not result_png.exists():
            raise FileNotFoundError(f"Expected result frame not found at {result_png}")

        evaluation_record = _write_evaluation_artifacts(
            evaluator,
            puzzle_id,
            result_png,
            output_dir,
            run_dir,
            attempt_idx,
        )
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
    parser.add_argument(
        "--use-gpt-5",
        action="store_true",
        help="Generate outputs with GPT-5 instead of the default VEO3 backend",
    )
    parser.add_argument(
        "--existing-output-root",
        type=Path,
        default=None,
        help="Evaluate pre-generated outputs under this directory instead of generating new results",
    )
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

    run_dirs: Dict[str, Path] = {}
    per_puzzle_results: Dict[str, List[Dict[str, Any]]] = {}
    image_paths_by_id: Dict[str, Path] = {}
    imgs: List[str] = []
    prompts: List[str] = []
    ids: List[str] = []
    for record in slice_records:
        pid_value = str(record["id"])
        if not pid_value:
            raise ValueError("Record missing 'id'")
        prompt_value = record["prompt"]
        if not isinstance(prompt_value, str):
            raise ValueError(f"Puzzle {pid_value} prompt must be a string")
        prompt_text = prompt_value.strip()
        image_rel = record["image"]
        if not isinstance(image_rel, str) or not image_rel:
            raise ValueError(f"Puzzle {pid_value} missing image")
        image_path = resolve_image_path(metadata_path, image_rel)
        if not image_path.exists():
            raise FileNotFoundError(f"Puzzle image not found: {image_path}")
        run_dirs[pid_value] = prepare_run_dir(pid_value, vote_root)
        per_puzzle_results[pid_value] = []
        image_paths_by_id[pid_value] = image_path
        if not args.use_gpt_5 and args.existing_output_root is None and not prompt_text:
            raise ValueError(f"Puzzle {pid_value} has no prompt text")
        imgs.append(image_path.as_posix())
        prompts.append(GPT5_PROMPT if args.use_gpt_5 else prompt_text)
        ids.append(pid_value)

    if args.existing_output_root is not None:
        existing_root = args.existing_output_root.resolve()
        per_puzzle_dirs = _gather_existing_output_dirs(existing_root, ids)
        for pid in ids:
            directories = per_puzzle_dirs[pid]
            for attempt_idx, output_dir in enumerate(directories, start=1):
                result_png = output_dir / "result.png"
                if not result_png.exists():
                    result_png = ensure_gpt5_result_image(
                        evaluator,
                        pid,
                        output_dir,
                        image_paths_by_id[pid],
                    )
                if not result_png.exists():
                    raise FileNotFoundError(f"Expected result frame not found at {result_png}")
                evaluation_record = _write_evaluation_artifacts(
                    evaluator,
                    pid,
                    result_png,
                    output_dir,
                    run_dirs[pid],
                    attempt_idx,
                )
                per_puzzle_results[pid].append(evaluation_record)
        for pid, results in per_puzzle_results.items():
            write_json(run_dirs[pid] / "run_manifest.json", {"puzzle_id": pid, "attempts": len(results), "results": results})
        print("Done.")
        return

    total = len(ids)
    for attempt_idx in range(1, args.k + 1):
        print(f"Batch generating attempt {attempt_idx}/{args.k} for {total} puzzles...")
        if args.use_gpt_5:
            outs = generate_outputs_multiprocess(
                imgs,
                prompts,
                processes=args.processes,
                attempts=3,
            )
        else:
            outs = generate_video_outputs_multiprocess(
                imgs,
                prompts,
                processes=args.processes,
                attempts=3,  # internal retries per item
            )
        for i, out_dir in enumerate(outs):
            pid = ids[i]
            output_dir = Path(out_dir).resolve()
            if args.use_gpt_5:
                result_png = ensure_gpt5_result_image(
                    evaluator,
                    pid,
                    output_dir,
                    image_paths_by_id[pid],
                )
            else:
                result_png = output_dir / "result.png"
            if not result_png.exists():
                raise FileNotFoundError(f"Expected result frame not found at {result_png}")
            evaluation_record = _write_evaluation_artifacts(
                evaluator,
                pid,
                result_png,
                output_dir,
                run_dirs[pid],
                attempt_idx,
            )
            per_puzzle_results[pid].append(evaluation_record)

    # Write per-puzzle run manifests
    for pid, results in per_puzzle_results.items():
        write_json(run_dirs[pid] / "run_manifest.json", {"puzzle_id": pid, "attempts": args.k, "results": results})
    print("Done.")


if __name__ == "__main__":
    main()
