"""Populate GPT-5 prompts across the dataset."""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List


def project_root() -> Path:
	return Path(__file__).resolve().parents[1]


def ensure_repo_on_path() -> None:
	root = project_root()
	root_str = str(root)
	if root_str not in sys.path:
		sys.path.insert(0, root_str)


def find_generator_prompt(dataset_name: str) -> str:
	module_name = f"puzzle.{dataset_name}.generator"
	module = importlib.import_module(module_name)
	for attribute_name in dir(module):
		attribute = getattr(module, attribute_name)
		if inspect.isclass(attribute) and attribute_name.lower().endswith("generator"):
			if hasattr(attribute, "DEFAULT_GPT5_PROMPT"):
				prompt = getattr(attribute, "DEFAULT_GPT5_PROMPT")
				if isinstance(prompt, str) and prompt.strip():
					return prompt
	raise AttributeError(f"No DEFAULT_GPT5_PROMPT found for {dataset_name}")


def load_records(container: object) -> Iterable[Dict[str, object]]:
	if isinstance(container, list):
		return (record for record in container if isinstance(record, dict))
	if isinstance(container, dict):
		if "puzzles" in container and isinstance(container["puzzles"], list):
			return (record for record in container["puzzles"] if isinstance(record, dict))
	raise TypeError("Unsupported dataset JSON structure")


def update_json_file(json_path: Path, prompt: str) -> int:
	data = json.loads(json_path.read_text())
	records = list(load_records(data))
	for record in records:
		record["gpt5_prompt"] = prompt
	json_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
	return len(records)


def update_dataset(dataset_dir: Path) -> Dict[str, int]:
	dataset_name = dataset_dir.name
	prompt = find_generator_prompt(dataset_name)
	json_files = sorted(path for path in dataset_dir.glob("*.json") if path.is_file())
	puzzle_count = 0
	for json_file in json_files:
		puzzle_count += update_json_file(json_file, prompt)
	return {"dataset": dataset_name, "files": len(json_files), "puzzles": puzzle_count}


def collect_dataset_dirs(root: Path) -> List[Path]:
	return sorted(path for path in root.iterdir() if path.is_dir())


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Add GPT-5 prompts to dataset JSON files")
	parser.add_argument("--dataset-dir", default="dataset", help="Root directory containing dataset splits")
	return parser.parse_args()


def main() -> None:
	ensure_repo_on_path()
	args = parse_args()
	dataset_root = Path(args.dataset_dir).resolve()
	if not dataset_root.exists():
		raise FileNotFoundError(f"Dataset directory not found: {dataset_root}")
	summaries: List[Dict[str, int]] = []
	for dataset_dir in collect_dataset_dirs(dataset_root):
		summary = update_dataset(dataset_dir)
		summaries.append(summary)
		print(f"Updated {summary['puzzles']} puzzles across {summary['files']} files in {summary['dataset']}")
	total_puzzles = sum(item["puzzles"] for item in summaries)
	print(f"Completed. Total puzzles updated: {total_puzzles}")


if __name__ == "__main__":
	main()
