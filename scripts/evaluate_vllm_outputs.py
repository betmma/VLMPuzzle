#!/usr/bin/env python3
"""Evaluate existing vllmInfer.py output folders for one puzzle type.

The script scans output/output_* style directories, infers the puzzle id from
the vLLM run artifacts, calls the matching puzzle evaluator module, and writes
mirrorVote-compatible evaluation.json files.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


ROOT = Path(__file__).resolve().parents[1]
PREFERRED_CANDIDATE_NAMES = ("result.png", "final.png", "candidate.png")


def sanitize(component: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in component.strip())
    return safe or "value"


def read_json_object(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists() or not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def read_metadata(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Metadata must be a list: {path}")
    records: List[Dict[str, Any]] = []
    for item in payload:
        if isinstance(item, dict):
            records.append(item)
    return records


def index_records_by_input_image(metadata_path: Path) -> Dict[str, str]:
    by_stem: Dict[str, str] = {}
    if not metadata_path.exists():
        return by_stem
    for record in read_metadata(metadata_path):
        puzzle_id = record.get("id")
        image_value = record.get("image")
        if not puzzle_id or not isinstance(image_value, str):
            continue
        image_path = metadata_path.parent / image_value
        by_stem[image_path.stem] = str(puzzle_id)
    return by_stem


def extract_input_image_from_text(input_file: Path) -> Optional[Path]:
    if not input_file.exists() or not input_file.is_file():
        return None
    text = input_file.read_text(encoding="utf-8")
    marker = "Input image path:"
    marker_index = text.find(marker)
    if marker_index == -1:
        return None
    remaining = text[marker_index + len(marker):].lstrip()
    if not remaining:
        return None
    line = remaining.splitlines()[0].strip()
    return Path(line) if line else None


def extract_id_from_input_text(input_file: Path) -> Optional[str]:
    if not input_file.exists() or not input_file.is_file():
        return None
    text = input_file.read_text(encoding="utf-8")
    for marker in ("Record id:", "Puzzle id:"):
        marker_index = text.find(marker)
        if marker_index == -1:
            continue
        remaining = text[marker_index + len(marker):].lstrip()
        if not remaining:
            continue
        line = remaining.splitlines()[0].strip()
        if line:
            return line
    image_path = extract_input_image_from_text(input_file)
    if image_path is None:
        return None
    stem = image_path.stem
    suffix = "_puzzle"
    return stem[: -len(suffix)] if stem.endswith(suffix) else stem


def infer_puzzle_id(output_dir: Path, image_id_index: Dict[str, str]) -> Optional[str]:
    run_record = read_json_object(output_dir / "run_record.json")
    if run_record:
        record = run_record.get("record")
        if isinstance(record, dict):
            for key in ("id", "record_id", "task_id"):
                value = record.get(key)
                if value:
                    return str(value)

    output_metadata = read_json_object(output_dir / "metadata.json")
    if output_metadata:
        for key in ("record_id", "id", "task_id"):
            value = output_metadata.get(key)
            if value:
                return str(value)

    input_file = output_dir / "input.txt"
    input_id = extract_id_from_input_text(input_file)
    if input_id:
        return input_id

    image_path = extract_input_image_from_text(input_file)
    if image_path is not None:
        return image_id_index.get(image_path.stem)
    return None


def iter_output_dirs(output_root: Path, recursive: bool) -> Iterable[Path]:
    if recursive:
        candidates = output_root.rglob("*")
    else:
        candidates = output_root.iterdir()
    for entry in sorted(candidates):
        if not entry.is_dir():
            continue
        if (entry / "input.txt").exists() or (entry / "metadata.json").exists() or (entry / "run_record.json").exists():
            yield entry


def choose_candidate_path(output_dir: Path, candidate_name: Optional[str]) -> Path:
    if candidate_name:
        return output_dir / candidate_name
    for name in PREFERRED_CANDIDATE_NAMES:
        candidate = output_dir / name
        if candidate.exists() and candidate.is_file():
            return candidate
    pngs = sorted(path for path in output_dir.glob("*.png") if path.is_file())
    if pngs:
        return pngs[-1]
    return output_dir / "result.png"


def prepare_vote_run_dir(vote_root: Path, puzzle_type: str, puzzle_id: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = vote_root / f"{sanitize(puzzle_type)}_{sanitize(puzzle_id)}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def copy_candidate_if_present(candidate_path: Path, attempt_dir: Path) -> Optional[Path]:
    if not candidate_path.exists() or not candidate_path.is_file():
        return None
    destination = attempt_dir / "result.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if candidate_path.resolve() != destination.resolve():
        shutil.copy2(candidate_path, destination)
    return destination


def run_evaluator(
    puzzle_type: str,
    metadata_path: Path,
    puzzle_id: str,
    candidate_path: Path,
    *,
    base_dir: Optional[Path],
    evaluator_args: List[str],
) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        "-m",
        f"puzzle.{puzzle_type}.evaluator",
        metadata_path.as_posix(),
        puzzle_id,
        candidate_path.as_posix(),
    ]
    if base_dir is not None:
        command.extend(["--base-dir", base_dir.as_posix()])
    command.extend(evaluator_args)
    return subprocess.run(command, cwd=ROOT, capture_output=True, text=True)


def parse_ids(raw_value: Optional[str]) -> Optional[set[str]]:
    if raw_value is None:
        return None
    values = {part.strip() for part in raw_value.split(",") if part.strip()}
    return values or None


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("puzzle_type", help="Puzzle package name, e.g. mirror, rects, circle_count")
    parser.add_argument("output_root", type=Path, help="Folder containing vllmInfer.py output_* directories")
    parser.add_argument("--metadata", type=Path, default=None, help="Metadata JSON path; default: data/<puzzle_type>/data.json")
    parser.add_argument("--base-dir", type=Path, default=None, help="Optional evaluator base directory")
    parser.add_argument("--vote-root", type=Path, default=Path("data/voteOutput"), help="Where grouped vote output is written")
    parser.add_argument("--candidate-name", default=None, help="Candidate filename inside each output dir; default: result.png/final.png/candidate.png")
    parser.add_argument("--ids", default=None, help="Comma-separated puzzle ids to evaluate")
    parser.add_argument("--recursive", action="store_true", help="Scan output_root recursively")
    parser.add_argument("--no-vote-copy", action="store_true", help="Only write evaluation.json inside original output dirs")
    parser.add_argument(
        "--evaluator-arg",
        action="append",
        default=[],
        help="Extra argument passed to the evaluator module. Repeat for each token.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    output_root = args.output_root.resolve()
    if not output_root.exists() or not output_root.is_dir():
        raise FileNotFoundError(f"Output root not found: {output_root}")

    metadata_path = (args.metadata or (ROOT / "data" / args.puzzle_type / "data.json")).resolve()
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")

    base_dir = args.base_dir.resolve() if args.base_dir is not None else None
    vote_root = args.vote_root.resolve()
    allowed_ids = parse_ids(args.ids)
    image_id_index = index_records_by_input_image(metadata_path)
    run_dirs: Dict[str, Path] = {}
    attempt_counts: Dict[str, int] = {}

    processed = 0
    failed = 0
    skipped = 0

    for output_dir in iter_output_dirs(output_root, args.recursive):
        puzzle_id = infer_puzzle_id(output_dir, image_id_index)
        if not puzzle_id or (allowed_ids is not None and puzzle_id not in allowed_ids):
            skipped += 1
            continue

        attempt_counts[puzzle_id] = attempt_counts.get(puzzle_id, 0) + 1
        attempt_index = attempt_counts[puzzle_id]
        candidate_path = choose_candidate_path(output_dir, args.candidate_name)
        completed = run_evaluator(
            args.puzzle_type,
            metadata_path,
            puzzle_id,
            candidate_path,
            base_dir=base_dir,
            evaluator_args=list(args.evaluator_arg),
        )

        evaluation_record: Dict[str, Any] = {
            "attempt": attempt_index,
            "puzzle_type": args.puzzle_type,
            "puzzle_id": puzzle_id,
            "output_directory": output_dir.as_posix(),
            "result_png": candidate_path.as_posix(),
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }

        vote_attempt_dir: Optional[Path] = None
        if not args.no_vote_copy:
            vote_run_dir = run_dirs.get(puzzle_id)
            if vote_run_dir is None:
                vote_run_dir = prepare_vote_run_dir(vote_root, args.puzzle_type, puzzle_id)
                run_dirs[puzzle_id] = vote_run_dir
            vote_attempt_dir = vote_run_dir / f"attempt_{attempt_index:02d}"
            vote_result_png = copy_candidate_if_present(candidate_path, vote_attempt_dir)
            evaluation_record["vote_run_directory"] = vote_run_dir.as_posix()
            evaluation_record["vote_output_directory"] = vote_attempt_dir.as_posix()
            if vote_result_png is not None:
                evaluation_record["vote_result_png"] = vote_result_png.as_posix()

        write_json(output_dir / "evaluation.json", evaluation_record)
        if vote_attempt_dir is not None:
            write_json(vote_attempt_dir / "evaluation.json", evaluation_record)

        if completed.returncode == 0:
            processed += 1
            print(f"[ok] {puzzle_id} attempt {attempt_index}: {output_dir}")
        else:
            failed += 1
            print(f"[failed] {puzzle_id} attempt {attempt_index}: {completed.stderr.strip()}")

    summary = {
        "output_root": output_root.as_posix(),
        "metadata": metadata_path.as_posix(),
        "vote_root": None if args.no_vote_copy else vote_root.as_posix(),
        "processed": processed,
        "failed": failed,
        "skipped": skipped,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
