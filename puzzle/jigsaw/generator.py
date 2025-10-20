"""Jigsaw puzzle generator implementation."""

from __future__ import annotations

import argparse
import random
import uuid
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from PIL import Image

from ..base import AbstractPuzzleGenerator, PathLike

try:  # Pillow 9/10 compatibility
    RESAMPLE_LANCZOS = Image.Resampling.LANCZOS
except AttributeError:  # pragma: no cover - older Pillow
    RESAMPLE_LANCZOS = Image.LANCZOS


@dataclass
class PieceSpec:
    """Piece specification within the solved image."""

    id: str
    row: int
    col: int
    bbox: Tuple[int, int, int, int]
    width: int
    height: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "row": self.row,
            "col": self.col,
            "bbox": list(self.bbox),
            "width": self.width,
            "height": self.height,
        }


@dataclass
class ScatterPlacement:
    """Placement of a shuffled piece inside the input puzzle image."""

    piece_id: str
    position: Tuple[int, int]
    rotation: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "piece_id": self.piece_id,
            "position": list(self.position),
            "rotation": self.rotation,
        }


@dataclass
class JigsawPuzzleRecord:
    """Metadata persisted for a single puzzle instance."""

    id: str
    prompt: str
    image_source: str
    original_image_path: str
    image: str
    grid: Dict[str, int]
    piece_edges: Dict[str, List[int]]
    pieces: List[PieceSpec]
    scatter_layout: List[ScatterPlacement]
    image_size: Tuple[int, int]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "prompt": self.prompt,
            "image_source": self.image_source,
            "original_image_path": self.original_image_path,
            "image": self.image,
            "grid": self.grid,
            "piece_edges": {k: list(v) for k, v in self.piece_edges.items()},
            "pieces": [piece.to_dict() for piece in self.pieces],
            "scatter_layout": [placement.to_dict() for placement in self.scatter_layout],
            "image_size": list(self.image_size),
        }


class JigsawGenerator(AbstractPuzzleGenerator[JigsawPuzzleRecord]):
    """Create shuffled jigsaw puzzles from random imagery."""

    def __init__(
        self,
        output_dir: PathLike = "data",
        *,
        rows: int = 3,
        cols: int = 3,
        image_size: Tuple[int, int] = (512, 512),
        prompt: str = "Reconstruct the shuffled tiles into the solved image.",
        allow_rotation: bool = True,
        scatter_scale: float = 1.8,
        scatter_margin: int = 24,
        max_scatter_attempts: int = 200,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(output_dir)
        self.rows = rows
        self.cols = cols
        self.image_size = image_size
        self.prompt = prompt
        self.allow_rotation = allow_rotation
        self.scatter_scale = scatter_scale
        self.scatter_margin = scatter_margin
        self.max_scatter_attempts = max_scatter_attempts
        self._rng = random.Random(seed)

        self.original_dir = self.output_dir / "original"
        self.input_dir = self.output_dir / "inputs"
        for path in (self.original_dir, self.input_dir):
            path.mkdir(parents=True, exist_ok=True)

    def create_puzzle(
        self,
        *,
        image: Image.Image,
        image_source: str,
        puzzle_id: Optional[str] = None,
    ) -> JigsawPuzzleRecord:
        """Create a puzzle from a PIL image and return its metadata."""

        puzzle_uuid = puzzle_id or str(uuid.uuid4())
        solved_image = image.convert("RGB").resize(self.image_size, RESAMPLE_LANCZOS)
        original_path = self.original_dir / f"{puzzle_uuid}_original.png"
        solved_image.save(original_path)

        pieces = self._slice_image(solved_image)
        shuffled_image, scatter_layout = self._scatter_pieces(pieces)
        input_path = self.input_dir / f"{puzzle_uuid}_input.png"
        shuffled_image.save(input_path)

        piece_edges = {
            "x": self._compute_axis_edges(solved_image.width, self.cols),
            "y": self._compute_axis_edges(solved_image.height, self.rows),
        }

        record = JigsawPuzzleRecord(
            id=puzzle_uuid,
            prompt=self.prompt,
            image_source=image_source,
            original_image_path=self.relativize_path(original_path),
            image=self.relativize_path(input_path),
            grid={"rows": self.rows, "cols": self.cols},
            piece_edges=piece_edges,
            pieces=[piece_spec for piece_spec, _ in pieces],
            scatter_layout=scatter_layout,
            image_size=solved_image.size,
        )
        return record

    def create_random_puzzle(self) -> JigsawPuzzleRecord:
        width, height = self.image_size
        random_token = self._rng.randint(0, 1_000_000_000)
        image_url = f"https://picsum.photos/{width}/{height}?random={random_token}"
        image = self._download_image(image_url)
        return self.create_puzzle(image=image, image_source=image_url)

    def create_puzzle_from_path(
        self,
        image_path: PathLike,
        *,
        image_source: Optional[str] = None,
        puzzle_id: Optional[str] = None,
    ) -> JigsawPuzzleRecord:
        """Utility wrapper that loads an image from disk and creates a puzzle."""

        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image path not found: {image_path}")
        with Image.open(image_path) as image:
            return self.create_puzzle(
                image=image,
                image_source=image_source or image_path.as_posix(),
                puzzle_id=puzzle_id,
            )

    def _download_image(self, url: str, *, timeout: int = 20) -> Image.Image:
        try:
            response = requests.get(url, timeout=timeout, proxies={})
            response.raise_for_status()
        except requests.RequestException as exc:  # pragma: no cover - network failure
            raise RuntimeError(f"Failed to download image from {url}") from exc
        image = Image.open(BytesIO(response.content))
        image.load()
        return image.convert("RGB")

    def _slice_image(self, image: Image.Image) -> List[Tuple[PieceSpec, Image.Image]]:
        x_edges = self._compute_axis_edges(image.width, self.cols)
        y_edges = self._compute_axis_edges(image.height, self.rows)

        pieces: List[Tuple[PieceSpec, Image.Image]] = []
        for row in range(self.rows):
            for col in range(self.cols):
                left, right = x_edges[col], x_edges[col + 1]
                top, bottom = y_edges[row], y_edges[row + 1]
                tile = image.crop((left, top, right, bottom))
                spec = PieceSpec(
                    id=f"{row}-{col}",
                    row=row,
                    col=col,
                    bbox=(left, top, right, bottom),
                    width=right - left,
                    height=bottom - top,
                )
                pieces.append((spec, tile))
        return pieces

    def _scatter_pieces(
        self,
        pieces: List[Tuple[PieceSpec, Image.Image]],
    ) -> Tuple[Image.Image, List[ScatterPlacement]]:
        canvas_width = int(self.image_size[0] * self.scatter_scale)
        canvas_height = int(self.image_size[1] * self.scatter_scale)
        background = Image.new("RGBA", (canvas_width, canvas_height), (32, 32, 32, 255))

        placements: List[ScatterPlacement] = []
        occupied: List[Tuple[int, int, int, int]] = []
        shuffled_order = list(pieces)
        self._rng.shuffle(shuffled_order)

        rotation_choices = [0, 90, 180, 270] if self.allow_rotation else [0]

        for spec, tile in shuffled_order:
            rotation = self._rng.choice(rotation_choices)
            tile_rgba = tile.convert("RGBA")
            rotated = tile_rgba.rotate(rotation, expand=True)
            placed = False

            max_x = canvas_width - rotated.width - self.scatter_margin
            max_y = canvas_height - rotated.height - self.scatter_margin
            if max_x <= self.scatter_margin or max_y <= self.scatter_margin:
                raise ValueError("Scatter canvas too small for given configuration")

            for _ in range(self.max_scatter_attempts):
                x = self._rng.randint(self.scatter_margin, max_x)
                y = self._rng.randint(self.scatter_margin, max_y)
                bbox = (x, y, x + rotated.width, y + rotated.height)
                if not any(self._intersects(bbox, other) for other in occupied):
                    background.alpha_composite(rotated, dest=(x, y))
                    placements.append(
                        ScatterPlacement(piece_id=spec.id, position=(x, y), rotation=rotation)
                    )
                    occupied.append(bbox)
                    placed = True
                    break

            if not placed:
                columns = max(
                    1,
                    (canvas_width - self.scatter_margin)
                    // max(1, tile.width + self.scatter_margin),
                )
                index = len(occupied)
                row_idx, col_idx = divmod(index, columns)
                x = min(
                    self.scatter_margin + col_idx * (tile.width + self.scatter_margin),
                    canvas_width - tile.width - self.scatter_margin,
                )
                y = min(
                    self.scatter_margin + row_idx * (tile.height + self.scatter_margin),
                    canvas_height - tile.height - self.scatter_margin,
                )
                fallback = tile_rgba
                background.alpha_composite(fallback, dest=(x, y))
                placements.append(ScatterPlacement(piece_id=spec.id, position=(x, y), rotation=0))
                bbox = (x, y, x + fallback.width, y + fallback.height)
                occupied.append(bbox)

        return background.convert("RGB"), placements

    @staticmethod
    def _compute_axis_edges(size: int, segments: int) -> List[int]:
        return [round(i * size / segments) for i in range(segments + 1)]

    @staticmethod
    def _intersects(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> bool:
        return not (a[2] <= b[0] or a[0] >= b[2] or a[3] <= b[1] or a[1] >= b[3])


__all__ = [
    "JigsawGenerator",
    "JigsawPuzzleRecord",
    "PieceSpec",
    "ScatterPlacement",
]


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate jigsaw puzzles for video LM training")
    parser.add_argument("count", type=int, help="Number of puzzles to generate")
    parser.add_argument("--rows", type=int, default=3, help="Number of rows to slice the image into")
    parser.add_argument("--cols", type=int, default=3, help="Number of columns to slice the image into")
    parser.add_argument(
        "--size",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=(512, 512),
        help="Final solved image size (pixels)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Directory where images and metadata will be saved",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="Optional metadata JSON path (defaults to <output>/data.json)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Reconstruct the shuffled tiles into the solved image.",
        help="Prompt stored with each puzzle",
    )
    parser.add_argument(
        "--no-rotation",
        action="store_true",
        help="Disable random 90-degree rotations when scattering pieces",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for deterministic dataset generation",
    )
    parser.add_argument(
        "--scatter-scale",
        type=float,
        default=1.8,
        help="Canvas scale factor for the shuffled puzzle image",
    )
    parser.add_argument(
        "--scatter-margin",
        type=int,
        default=24,
        help="Minimum pixel margin between puzzle pieces and canvas edges",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)
    metadata_path = args.metadata or (args.output_dir / "data.json")
    generator = JigsawGenerator(
        output_dir=args.output_dir,
        rows=args.rows,
        cols=args.cols,
        image_size=tuple(args.size),
        prompt=args.prompt,
        allow_rotation=not args.no_rotation,
        scatter_scale=args.scatter_scale,
        scatter_margin=args.scatter_margin,
        seed=args.seed,
    )
    generator.generate_dataset(args.count, metadata_path=metadata_path)


if __name__ == "__main__":
    main()
