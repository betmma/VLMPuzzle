# Video LM Puzzle Toolkit

Utilities for generating shuffled jigsaw puzzles (input images plus metadata) and compact 4x4 Sudoku board challenges, along with evaluators for verifying model reconstructions.

## Setup

```
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows
.venv\Scripts\activate
pip install -r requirements.txt
```

## Generating jigsaw puzzles

Run the generator CLI to download random images from https://picsum.photos, slice them into a grid, scatter the tiles, and store metadata and images on disk.

```
python -m puzzle.jigsaw.generator 5 --rows 3 --cols 3 --size 512 512 --output-dir data/jigsaw --metadata data/jigsaw/puzzles.json --prompt "Solve the jigsaw puzzle" --seed 42
```

Key outputs per puzzle:

- data/original/<id>_original.png: solved reference image.
- data/inputs/<id>_input.png: scattered puzzle layout for the model input.
- data/puzzles.json: metadata records with id, source URL, prompt, grid definition, per-piece bounding boxes, and scatter layout.

You can also build a puzzle from a local image:

```
from puzzle.jigsaw import JigsawGenerator

gen = JigsawGenerator(output_dir="data", rows=4, cols=4)
record = gen.create_puzzle_from_path("my_image.jpg")
```

## Evaluating jigsaw outputs

Given a stored puzzle id and a candidate solution image (for example the final frame from a model), run the evaluator. It trims borders, resizes to the reference dimensions, compares per-tile similarity, and reports accuracy.

```
python -m puzzle.jigsaw.evaluator data/puzzles.json <PUZZLE_ID> path/to/model_output.png --threshold 0.92 --trim-tolerance 10
```

Example JSON result:

```
{
  "puzzle_id": "...",
  "correct_pieces": 8,
  "total_pieces": 9,
  "accuracy": 0.8889,
  "per_piece": [
    {"piece_id": "0-0", "similarity": 0.99, "is_correct": true},
    {"piece_id": "0-1", "similarity": 0.85, "is_correct": false}
  ]
}
```

Adjust --threshold to control tolerance for per-piece correctness. The default similarity metric uses 1 - mean absolute error between RGB tiles (scaled to [0,1]).

## Generating Sudoku puzzles

```
python -m puzzle.sudoku.generator 10 --output-dir data/sudoku --clue-target 12 --seed 7
```

Artifacts per puzzle:
- Sudoku clues remain black in both puzzle and solution images; filled cells are rendered in blue for clarity.

- data/sudoku/puzzles/<id>_puzzle.png: printable puzzle grid with blanks.
- use `--aspect-ratio` to add black padding around the outer border while keeping the inner grid square.
- data/sudoku/solutions/<id>_solution.png: colored solution grid for reference.
- data/sudoku/puzzles.json: metadata storing puzzle/solution grids, clue counts, and prompts.

Programmatic example:

```
from puzzle.sudoku import SudokuGenerator

gen = SudokuGenerator(output_dir="data/sudoku", clue_target=10, seed=101)
record = gen.create_puzzle()
```

## Evaluating Sudoku solutions

Provide the evaluator with the metadata file, puzzle id, and a candidate solution image (final frame from the model or a rendered board).

```
python -m puzzle.sudoku.evaluator data/sudoku/puzzles.json <PUZZLE_ID> candidate.png
```

The evaluator trims borders, rescales the candidate to the reference solution, reads the digit in each cell, and reports accuracy plus invalid positions.

## Generating mirror puzzles

```
python -m puzzle.mirror.generator 10 --output-dir data/mirror --rows 6 --cols 8 --cell-size 48 --fill 0.6 --seed 7 --monochrome
```

Outputs:
- data/mirror/puzzles/<id>_puzzle.png: left half colored, right half blank.
- add `--monochrome` to keep a single hue across all filled cells.
- use `--aspect-ratio` to add black padding around the outer border.
- data/mirror/solutions/<id>_solution.png: full mirrored grid reference.
- data/mirror/puzzles.json: metadata with per-cell colors and layout details.

## Evaluating mirror outputs

```
python -m puzzle.mirror.evaluator data/mirror/puzzles.json <PUZZLE_ID> candidate.png --color-tolerance 20
```

Each right-half cell is compared against its mirrored counterpart by averaging RGB values and measuring color distance.

## Notes

- Sudoku OCR relies on Tesseract via `pytesseract`; install the Tesseract binary and ensure it is available on PATH.
- Jigsaw puzzles fetch images from the network; ensure outbound access or rely on `create_puzzle_from_path` for offline use.
- Sudoku generation enforces uniqueness by default; use `--no-unique` to accelerate dataset builds when uniqueness is not required.
- Scatter layout avoids overlaps via random sampling with a deterministic fallback when space is tight.
- Similarity metrics are lightweight; replace `_piece_similarity` (jigsaw) or Sudoku validation helpers with stronger perceptual or domain-specific checks if tighter validation is required.

## Generating ARC puzzles

```
python -m puzzle.arcagi.generator 5 --dataset data/training --output-dir data/arcagi --metadata data/arcagi/puzzles.json --cell-size 28
```

Each puzzle image renders every training example as an input/output pair with an arrow pointing from example input to example output. The first evaluation pair is shown with its input grid and a blank answer grid. The matching solution image fills in the correct test output.

Artifacts per puzzle:
- data/arcagi/puzzles/<id>_puzzle.png: composite image with examples and blank test output.
- data/arcagi/solutions/<id>_solution.png: identical layout with the correct test output filled in.
- data/arcagi/puzzles.json: metadata that records the task source and grid placements (including the test output region).

## Evaluating ARC outputs

```
python -m puzzle.arcagi.evaluator data/arcagi/puzzles.json <PUZZLE_ID> candidate.png
```

The evaluator aligns the candidate image to the puzzle layout, reads the coloured cells from the test output region, maps RGB values back to ARC palette digits, and compares them to the ground-truth grid.

## Generating maze puzzles

```
python -m puzzle.maze.generator 5 --output-dir data/maze --rows 15 --cols 15 --cell-size 32 --aspect-ratio 1.6
```

Each maze shows black walls, a red start cell, and a green goal cell. The prompt tells the model to draw a single red line from start to goal while staying on the white passages. Solution images include the reference path, but the puzzle frames remain blank so models must supply the line.

Artifacts per puzzle:
- data/maze/puzzles/<id>_puzzle.png: maze without the path.
- data/maze/solutions/<id>_solution.png: same maze with the red reference path.
- data/maze/puzzles.json: metadata with the maze grid, start/goal coordinates, padding, and bounding boxes for each cell.

## Evaluating maze outputs

```
python -m puzzle.maze.evaluator data/maze/puzzles.json <PUZZLE_ID> candidate.png
```

The evaluator aligns the candidate image, checks that a continuous red path connects the start and goal cells, and verifies that no red pixels spill into wall cells.




