"""Re-evaluate existing vote outputs for a specific puzzle type.

This utility walks an existing vote output directory, re-runs the puzzle
Evaluator on each attempt, and stores refreshed evaluation payloads in a new
vote output directory. The original data remains untouched.
"""

from __future__ import annotations

import argparse
import importlib
import json
import shutil
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple, Type

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from puzzle.base import AbstractPuzzleEvaluator  # noqa: E402


def _discover_evaluator_class(puzzle_type: str) -> Type[AbstractPuzzleEvaluator]:
    module_name = f"puzzle.{puzzle_type}.evaluator"
    module = importlib.import_module(module_name)
    classes = []
    for attr_name in dir(module):
        attr_value = getattr(module, attr_name)
        if isinstance(attr_value, type) and issubclass(attr_value, AbstractPuzzleEvaluator) and attr_value is not AbstractPuzzleEvaluator:
            classes.append(attr_value)
    classes.sort(key=lambda cls: cls.__name__)
    if not classes:
        raise ValueError(f"No evaluator subclass found in {module_name}")
    return classes[0]


def _serialize_result(result: object) -> Dict[str, object]:
    if hasattr(result, "to_dict"):
        payload = result.to_dict()  # type: ignore[call-arg]
        if isinstance(payload, dict):
            return payload
    if is_dataclass(result):
        return asdict(result)
    if isinstance(result, dict):
        return dict(result)
    raise TypeError("Evaluator result must provide a to_dict() method or be a dataclass")


def _iter_evaluation_entries(root: Path) -> Iterator[Tuple[Path, Path, Path]]:
    for vote_run in sorted(root.iterdir()):
        if not vote_run.is_dir():
            continue
        nested_attempt = False
        for attempt_dir in sorted(vote_run.iterdir()):
            if not attempt_dir.is_dir():
                continue
            evaluation_path = attempt_dir / "evaluation.json"
            if evaluation_path.exists() and evaluation_path.is_file():
                nested_attempt = True
                yield vote_run, attempt_dir, evaluation_path
        if nested_attempt:
            continue
        evaluation_path = vote_run / "evaluation.json"
        if evaluation_path.exists() and evaluation_path.is_file():
            yield vote_run, vote_run, evaluation_path


def _resolve_candidate_image(payload: Dict[str, object], attempt_dir: Path, repo_root: Path) -> Optional[Path]:
    def _resolve_path(raw_path: str) -> Path:
        candidate = Path(raw_path)
        if candidate.is_absolute():
            return candidate
        return (repo_root / candidate).resolve()

    raw_output_dir = payload.get("output_directory")
    if isinstance(raw_output_dir, str) and raw_output_dir.strip():
        output_dir = _resolve_path(raw_output_dir.strip())
        if output_dir.exists() and output_dir.is_dir():
            preferred_names = ("result.png", "final.png", "candidate.png")
            for name in preferred_names:
                candidate_path = output_dir / name
                if candidate_path.exists() and candidate_path.is_file():
                    return candidate_path
            png_candidates = sorted(output_dir.glob("*.png"))
            if png_candidates:
                return png_candidates[-1]

    for key in ("vote_result_png", "result_png"):
        raw_value = payload.get(key)
        if isinstance(raw_value, str) and raw_value.strip():
            direct_path = _resolve_path(raw_value.strip())
            if direct_path.exists() and direct_path.is_file():
                return direct_path
            candidate_local = attempt_dir / Path(raw_value).name
            if candidate_local.exists() and candidate_local.is_file():
                return candidate_local

    fallback = attempt_dir / "result.png"
    if fallback.exists() and fallback.is_file():
        return fallback
    return None


def _default_output_root(source_root: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return source_root.parent / f"{source_root.name}_reeval_{timestamp}"


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Re-evaluate vote outputs for a puzzle type")
    parser.add_argument("input_vote_root", type=Path, help="Existing vote output directory to re-evaluate")
    parser.add_argument("--metadata", type=Path, default=None, help="Puzzle metadata JSON path (default: data/<puzzle_type>/data.json)")
    parser.add_argument("--base-dir", type=Path, default=None, help="Optional override for evaluator base directory")
    parser.add_argument("--output-root", type=Path, default=None, help="Destination vote output directory (default: create timestamped sibling)")
    parser.add_argument("--no-copy-result-image", action="store_true", help="Skip copying result images into the new vote output")
    return parser


def main(argv: Optional[Iterator[str]] = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    puzzle_type = ''

    input_vote_root = args.input_vote_root.resolve()
    if not input_vote_root.exists() or not input_vote_root.is_dir():
        raise FileNotFoundError(f"Vote output directory not found: {input_vote_root.as_posix()}")

    output_root = args.output_root.resolve() if args.output_root else _default_output_root(input_vote_root)
    output_root.mkdir(parents=True, exist_ok=True)

    metadata_path = args.metadata if args.metadata else (REPO_ROOT / "data" )
    metadata_resolved = metadata_path.resolve()
    # if not metadata_resolved.exists():
    #     raise FileNotFoundError(f"Metadata file not found: {metadata_resolved.as_posix()}")

    base_dir = args.base_dir.resolve() if args.base_dir else None

    evaluator_cls = None #_discover_evaluator_class(puzzle_type)
    evaluator = None

    processed = 0
    skipped = 0

    for vote_run_dir, attempt_dir, evaluation_path in _iter_evaluation_entries(input_vote_root):
        payload_text = evaluation_path.read_text(encoding="utf-8")
        payload_outer = json.loads(payload_text)
        stdout_blob = payload_outer.get("stdout")
        puzzle_id=None
        if not isinstance(stdout_blob, str) or not stdout_blob.strip():
            pass
            # skipped += 1
            # continue
        else:
            inner_payload = json.loads(stdout_blob)
            puzzle_id_raw = inner_payload.get("puzzle_id")
            puzzle_id = str(puzzle_id_raw).strip() if puzzle_id_raw is not None else ""
        if not puzzle_id:
            try:
                puzzle_id= payload_outer.get("vote_run_directory").split('/')[-1].split('_')[-3]
            except Exception:
                skipped += 1
                continue

        try:
            this_puzzle_type='_'.join(payload_outer.get("vote_run_directory").split('/')[-1].split('_')[:-3])
            if puzzle_type != this_puzzle_type:
                puzzle_type = this_puzzle_type
                evaluator_cls = _discover_evaluator_class(puzzle_type)
                evaluator = evaluator_cls(metadata_resolved/puzzle_type/'data.json', base_dir=base_dir)
                print(f"Switched to evaluator for puzzle type: {puzzle_type}")
        except Exception as e:
            print(f"Failed to discover evaluator for puzzle type from payload: {e}")
            skipped += 1
            continue
        candidate_image = payload_outer['result_png']
        print(puzzle_type, puzzle_id, candidate_image)
        result = evaluator.evaluate(puzzle_id, candidate_image)
        result_payload = _serialize_result(result)

        target_attempt_dir = output_root / attempt_dir.relative_to(input_vote_root)
        target_attempt_dir.mkdir(parents=True, exist_ok=True)

        if args.no_copy_result_image:
            candidate_destination = None
        else:
            candidate_destination = target_attempt_dir / candidate_image.name
            if candidate_destination.resolve() != candidate_image.resolve():
                candidate_destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(candidate_image, candidate_destination)
            else:
                candidate_destination = candidate_image

        target_vote_run = output_root / vote_run_dir.relative_to(input_vote_root)
        target_vote_run.mkdir(parents=True, exist_ok=True)

        refreshed_payload = dict(payload_outer)
        refreshed_payload["stdout"] = json.dumps(result_payload, indent=2, ensure_ascii=False) + "\n"
        refreshed_payload["returncode"] = 0
        refreshed_payload["stderr"] = ""
        refreshed_payload["reevaluated"] = True
        refreshed_payload["reevaluated_at"] = datetime.now(timezone.utc).isoformat()
        refreshed_payload["source_vote_run_directory"] = vote_run_dir.as_posix()
        refreshed_payload["source_vote_output_directory"] = attempt_dir.as_posix()
        refreshed_payload["vote_run_directory"] = target_vote_run.as_posix()
        refreshed_payload["vote_output_directory"] = target_attempt_dir.as_posix()
        if candidate_destination is not None:
            refreshed_payload["vote_result_png"] = candidate_destination.as_posix()
            refreshed_payload["result_png"] = candidate_destination.as_posix()

        refreshed_text = json.dumps(refreshed_payload, indent=2, ensure_ascii=False)
        (target_attempt_dir / "evaluation.json").write_text(refreshed_text + "\n", encoding="utf-8")

        processed += 1

    summary = {
        "input_vote_root": input_vote_root.as_posix(),
        "output_vote_root": output_root.as_posix(),
        "processed_attempts": processed,
        "skipped_attempts": skipped,
        "copy_result_image": not args.no_copy_result_image,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
