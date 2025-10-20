import argparse
import ast
import json
import os
import multiprocessing as mp
import subprocess
import sys
from pathlib import Path
from typing import Any, List, Sequence, Tuple

# Utility script to batch-generate puzzles via CLI and then run mirrorVote.py on each.

GENERATOR_MODULES = {
}
for dirpath, dirnames, filenames in os.walk('puzzle'):
    for dirname in dirnames:
        if os.path.exists(os.path.join(dirpath, dirname, 'generator.py')) and os.path.exists(os.path.join(dirpath, dirname, 'generator.py')):
            GENERATOR_MODULES[dirname] = f"puzzle.{dirname}.generator"


def parse_generator_tokens(entries: Sequence[str]) -> List[Tuple[str, Any]]:
    tokens: List[Tuple[str, Any]] = []
    for entry in entries:
        raw = entry.strip()
        if not raw:
            continue
        if "=" in raw:
            key, value = raw.split("=", 1)
            key = key.strip()
            value = value.strip()
            try:
                parsed: Any = ast.literal_eval(value)
            except Exception:
                parsed = value
            tokens.append((key, parsed))
        else:
            tokens.append((raw, True))
    return tokens


def build_generator_command(
    module_name: str,
    count: int,
    output_dir: Path,
    option_tokens: Sequence[Tuple[str, Any]],
    use_gpt_5: bool,
) -> List[str]:
    command: List[str] = [sys.executable, "-m", module_name, str(count)]
    command.extend(["--output-dir", output_dir.as_posix()])

    for key, value in option_tokens:
        flag_key = key.lstrip("-").replace("_", "-")
        flag = f"--{flag_key}"
        if isinstance(value, bool):
            if value:
                command.append(flag)
            continue
        if value is None:
            continue
        command.extend([flag, str(value)])
    if use_gpt_5:
        command.append("--use-gpt-5")
    return command


def read_metadata(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"Metadata file {path} does not contain a list")
    return payload


def run_generator_cli(command: Sequence[str]) -> None:
    completed = subprocess.run(command)
    if completed.returncode != 0:
        joined = " ".join(command)
        raise RuntimeError(f"Generator command failed ({completed.returncode}): {joined}")


def collect_new_records(
    metadata_path: Path,
    before_count: int,
    expected_new: int,
) -> List[dict]:
    records = read_metadata(metadata_path)
    if len(records) < before_count + expected_new:
        raise ValueError(
            "Generator did not append the expected number of puzzles ("
            f"expected {expected_new}, found {len(records) - before_count})."
        )
    return records[before_count : before_count + expected_new]


def run_mirror_vote(puzzle_type: str, puzzle_id: str, attempts: int, puzzles_path: Path, use_gpt_5: bool) -> None:
    command = [
        sys.executable,
        "scripts/mirrorVote.py",
        puzzle_type,
        puzzle_id,
        str(attempts),
        puzzles_path.as_posix(),
        str(use_gpt_5)
    ]
    completed = subprocess.run(command)
    if completed.returncode != 0:
        raise RuntimeError(
            f"mirrorVote.py failed for puzzle {puzzle_id} (type {puzzle_type}) with exit code {completed.returncode}"
        )


def _run_vote_task(params: Tuple[str, str, int, Path, bool]) -> str:
    puzzle_type, puzzle_id, attempts, puzzles_path, use_gpt_5 = params
    # Execute the vote run for a single puzzle and return its id on success.
    run_mirror_vote(puzzle_type, puzzle_id, attempts, puzzles_path, use_gpt_5)
    return puzzle_id


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate puzzles of one type via CLI and run mirrorVote.py for each."
    )
    parser.add_argument("puzzle_type", choices=sorted(GENERATOR_MODULES.keys()))
    parser.add_argument("count", type=int, help="Number of puzzles to create")
    parser.add_argument(
        "--attempts",
        type=int,
        default=3,
        help="Number of voting attempts for mirrorVote.py (default: 3)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override generator output directory (default: data/<puzzle_type>)",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="Metadata path (must be <output-dir>/data.json)",
    )
    parser.add_argument(
        "--generator-option",
        action="append",
        default=[],
        help="Additional generator CLI arguments as key=value entries (repeatable).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel worker processes for voting (default: min(count, CPU cores))",
    )
    parser.add_argument(
        "--use-gpt-5",
        action="store_true",
        help="Use gpt-5 instead of sora"
    )

    args = parser.parse_args()

    if args.count <= 0:
        raise ValueError("count must be positive")
    if args.attempts <= 0:
        raise ValueError("attempts must be positive")

    puzzle_type = args.puzzle_type
    module_name = GENERATOR_MODULES[puzzle_type]

    output_dir = args.output_dir or (Path("data") / puzzle_type)
    metadata_path = args.metadata or (output_dir / "data.json")
    expected_metadata_path = output_dir / "data.json"
    if metadata_path.resolve() != expected_metadata_path.resolve():
        raise ValueError(
            "Metadata path must match <output-dir>/data.json when invoking the generator CLI."
        )

    option_tokens = parse_generator_tokens(tuple(args.generator_option))

    before_records = read_metadata(metadata_path)
    generator_command = build_generator_command(module_name, args.count, output_dir, option_tokens, args.use_gpt_5)
    print("Running:", " ".join(generator_command))
    run_generator_cli(generator_command)

    new_records = collect_new_records(metadata_path, len(before_records), args.count)
    print(f"Generated {len(new_records)} {puzzle_type} puzzle(s). Metadata recorded at {metadata_path}.")

    # Parallelize mirrorVote executions across generated puzzles.
    tasks: List[Tuple[str, str, int, Path, bool]] = []
    for record in new_records:
        puzzle_id = record.get("id")
        if not puzzle_id:
            raise ValueError("Generated puzzle record missing 'id'")
        tasks.append((puzzle_type, puzzle_id, args.attempts, metadata_path, args.use_gpt_5))

    if not tasks:
        print("No new puzzles to process.")
        return

    cpu_cores = os.cpu_count() or 1
    workers = args.workers or min(len(tasks), cpu_cores)
    workers = max(1, workers)

    print(f"Starting mirrorVote.py in parallel: {len(tasks)} puzzle(s), {workers} worker(s)...")
    with mp.Pool(processes=workers) as pool:
        try:
            for idx, done_id in enumerate(pool.imap_unordered(_run_vote_task, tasks), 1):
                print(f"Completed voting for {done_id} ({idx}/{len(tasks)})")
        finally:
            pool.close()
            pool.join()

    print("All puzzles processed.")


if __name__ == "__main__":
    main()
