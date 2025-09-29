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
python -m puzzle.jigsaw.generator 5 --rows 3 --cols 3 --size 512 512 --output-dir data --metadata data/puzzles.json --prompt "Solve the jigsaw puzzle" --seed 42
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

## Notes

- Sudoku OCR relies on Tesseract via `pytesseract`; install the Tesseract binary and ensure it is available on PATH.
- Jigsaw puzzles fetch images from the network; ensure outbound access or rely on `create_puzzle_from_path` for offline use.
- Sudoku generation enforces uniqueness by default; use `--no-unique` to accelerate dataset builds when uniqueness is not required.
- Scatter layout avoids overlaps via random sampling with a deterministic fallback when space is tight.
- Similarity metrics are lightweight; replace `_piece_similarity` (jigsaw) or Sudoku validation helpers with stronger perceptual or domain-specific checks if tighter validation is required.
