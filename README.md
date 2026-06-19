
# VideoThinkBench Vision-Centric Toolkit

This repository hosts the assets behind the vision-centric portion of the “Thinking with Video” study. It covers the eyeballing puzzles, ARC-AGI-2 abstractions, and maze families used to evaluate Sora-2 and contemporary VLMs within VideoThinkBench. Text-centric benchmarks and visual puzzle variations live elsewhere to keep this codebase focused on spatial reasoning through video generation.

## Getting started

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

All generators and evaluators run on Python 3.10+. Optional dependencies such as Whisper, ffmpeg, and GPU-accelerated OpenCV improve throughput for large batches but are not required for small experiments.

## Repository layout

- `puzzle/`: generators and evaluators for the three task families.
	- Eyeballing puzzles live in directories named after the geometric target (`circle_center/`, `angle_bisector/`, …).
	- ARC-AGI-2 abstractions are implemented in `arcagi/`.
	- Maze variants (`maze/`, `maze_hexagon/`, `maze_labyrinth/`) share common helpers in `maze_base.py`.
- `data/`: default output root.
- `scripts/`: orchestration utilities for bulk generation, multi-sample voting, transcription, and result summaries.

> **Note**: The repository still carries earlier puzzle prototypes (jigsaw, Sudoku, mirror, rectangles, etc.). They are preserved for completeness but were not part of the published experiments.

## General scripts

`scripts/veo3.py` and `scripts/gpt5.py` call corresponding API to get model generations. `veo3.py` is for video generation models, and `gpt5.py` is for VLMs.
`scripts/mirrorVote.py` generates multiple responses for one puzzle.
`scripts/generate_and_vote.py` generates new puzzles and calls mirrorVote.py to generate the responses.
set --use-gpt-5 or [use_gpt_5] when running the above two scripts for VLMs.
`scripts/fixed_dataset.py` evaluates puzzles on fixed dataset instead of generating new puzzles. Our dataset and mini testset can be found ![here](https://huggingface.co/datasets/OpenMOSS-Team/VideoThinkBench). Ensure the files are arranged as example below:

- `dataset/`: arbitrary folder name.
	- `maze_square/`: puzzle type name.
		- `puzzles/`: input images of puzzles.
		- `solutions/`: solution images of puzzles.
		- `data.json`: data of puzzles.
	- `.../`: other puzzle types, with same `puzzles/` `solutions/` and `data.json` inside.

Then run `scripts/fixed_dataset.py`, for example `python scripts/fixed_dataset.py --dataset-root dataset --workers 16 --resume` to evaluate on a fixed dataset.

## Eyeballing puzzles

Eyeballing puzzles require the model to mark the correct geometric element from five options while optionally verbalizing the choice. We evaluate three groups:

- **Point Tasks**: `circle_center`, `circumcenter`, `fermat_point`, `incenter`, `midpoint`, `orthocenter`, `point_reflection`, `ray_intersection`, `triangle_center`.
- **Line Tasks**: `angle_bisector`, `arc_connect`, `circle_tangent_line`, `circle_tangent_point`, `parallel`, `perpendicular`, `perpendicular_bisector`, `ray_reflect`.
- **Shape Tasks**: `isosceles_trapezoid`, `parallelogram`, `right_triangle`, `square_outlier`.

Each task inherits from the shared point-target scaffolding in `point_target_base.py`, so the CLI and output layout are consistent.

### Generate puzzles

```bash
python -m puzzle.circle_center.generator 25 --output-dir data/circle_center --seed 2025
```

Common flags:

- `count`: number of puzzles to create.
- `--canvas-width` and `--aspect`: customize resolution. Aspect controls portrait vs. landscape layout.
- `--prompt`: override the default instruction. Use `--use-gpt-5` to inject the multiple-choice wording used for GPT-5 baselines.

Metadata is stored in `<output-dir>/data.json`, while puzzles and reference solutions land in `puzzles/` and `solutions/` subfolders.

### Evaluate predictions

```bash
python -m puzzle.circle_center.evaluator data/circle_center/data.json <PUZZLE_ID> attempts/0001/final.png --video-stride 3
```

The evaluator reports the option inferred from:

- the red highlight in the candidate image,
- parsed captions or transcripts located next to the attempt,
- sampled frames from the accompanying video.

Most leaderboard scores quoted in the paper use majority voting over frames (“Major Frame”), last-frame inspection, or the audio transcript.

`scripts/gen_point.sh` can generate training data for eyeballing puzzles. The "VIDEO" argument can be set to true to also generate ground truth videos.

`scripts/run_point.sh` calls `scripts/generate_and_vote.py` that generates new puzzles for multiple puzzle types and then evaluate them.

`scripts/multiple_choice_summary.py` outputs summary for all eyeballing puzzles.

## ARC-AGI-2 abstractions

Our ARC implementation turns few-shot grid reasoning into a video-friendly format: training exemplars appear on the left, the target input is rendered on the right, and the answer grid remains blank for the model to fill.

```bash
python -m puzzle.arcagi.generator 10 --dataset data/training --output-dir data/arcagi --seed 17
python -m puzzle.arcagi.evaluator data/arcagi/data.json <PUZZLE_ID> attempts/arcagi/final.png
```

Key helpers:

- `scripts/generate_all_arc_puzzles.py`: bulk generation using all jsons in ARC-AGI-2 dataset. use --video to generate video solutions besides image solutions. use --split *an integer* to generate more than ARC-AGI-2 dataset amount by combination over training instances.
- `scripts/arcagi_range_vote.py`: aggregates self-consistency runs (supports GPT-5, Claude 4.5, Gemini 2.5 Pro, and Sora-2 outputs). The paper’s ablations rely on these ranges.

Evaluation converts colored cells back to ARC palette indices and prints JSON with per-cell agreement, enabling downstream voting or qualitative review.

## Maze families

Maze benchmarks test dynamic path drawing. Three generators ship in this repo:

- `puzzle.maze_square.generator`: rectangular grids.
- `puzzle.maze_hexagon.generator`: hex-tiling mazes.
- `puzzle.maze_labyrinth.generator`: labyrinth variants with preset motifs.

```bash
python -m puzzle.maze_square.generator 20 --output-dir data/maze --rows 21 --cols 21 --cell-size 32
python -m puzzle.maze_square.evaluator data/maze/data.json <PUZZLE_ID> attempts/maze/final.png
```

Mazes highlight the start cell and the goal in red. The evaluator verifies that a continuous red stroke connects them without bleeding into walls. `scripts/maze_summary.py` collects aggregate accuracy from batches of attempts.

Generators can use --video to generate video solutions, and --use-gpt-5 to print cell ids on the image, for VLMs to answer.

## Legacy generators not in the paper

Directories such as `puzzle/jigsaw/`, `puzzle/sudoku/`, `puzzle/mirror/`, and `puzzle/rects/` remain in the tree for archival reasons. They can still be executed with the same CLI pattern as before, but their outputs were not included in the “Thinking with Video” results. 


