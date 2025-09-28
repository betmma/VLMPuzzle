# Video LM Jigsaw Puzzle Toolkit

Utilities for generating shuffled jigsaw puzzles (input images plus metadata) and evaluating model reconstructions for video-language model training.

## Setup

```
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows
.venv\Scripts\activate
pip install -r requirements.txt
```

## Generating puzzles

Run the generator CLI to download random images from https://picsum.photos, slice them into a grid, scatter the tiles, and store metadata and images on disk.

```
python -m puzzle.generator 5 --rows 3 --cols 3 --size 512 512 --output-dir data --metadata data/puzzles.json --prompt "Solve the jigsaw puzzle" --seed 42
```

Key outputs per puzzle:

- data/original/<id>_original.png: solved reference image.
- data/inputs/<id>_input.png: scattered puzzle layout for the model input.
- data/puzzles.json: metadata records with id, source URL, prompt, grid definition, per-piece bounding boxes, and scatter layout.

You can also build a puzzle from a local image:

```
from puzzle.generator import PuzzleGenerator

gen = PuzzleGenerator(output_dir="data", rows=4, cols=4)
record = gen.create_puzzle_from_path("my_image.jpg")
```

## Evaluating model outputs

Given a stored puzzle id and a candidate solution image (for example the final frame from a model), run the evaluator. It trims borders, resizes to the reference dimensions, compares per-tile similarity, and reports accuracy.

```
python -m puzzle.evaluator data/puzzles.json <PUZZLE_ID> path/to/model_output.png --threshold 0.85 --trim-tolerance 10
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

## Notes

- The generator fetches images from the network; ensure outbound access or replace with create_puzzle_from_path for offline use.
- Scatter layout avoids overlaps via random sampling with a deterministic fallback when space is tight.
- Similarity is a lightweight metric; replace _piece_similarity with a stronger perceptual measure if tighter validation is required.
