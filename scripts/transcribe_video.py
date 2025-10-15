"""Transcribe speech from a video file and extract the first NATO code word.

CLI usage examples:

  - Basic transcription to stdout:
      python scripts/transcribe_video.py path/to/video.mp4

  - Save JSON output (transcript + first_nato_word) to a file:
      python scripts/transcribe_video.py path/to/video.mp4 --output-json out.json

  - Only print the first NATO code word if present:
      python scripts/transcribe_video.py path/to/video.mp4 --nato-only

Notes:
  - Requires ffmpeg on PATH to extract audio.
  - Supports local transcription with Whisper (default) or an API engine.
    Local mode uses the `openai-whisper` package (install via `pip install openai-whisper`).
  - API mode uses an OpenAI-compatible audio transcription API. Reads API key
    from repo-root `api_key.txt` by default, or the environment variable
    OPENAI_API_KEY if present.
  - You can override the API base URL via --base-url or OPENAI_BASE_URL.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

# Ensure repository root on sys.path to align with other scripts
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - optional at import time
    OpenAI = None  # type: ignore[assignment]

try:
    import whisper  # local inference engine (openai-whisper)
except Exception:  # pragma: no cover
    whisper = None  # type: ignore[assignment]


# ---------- NATO Option Extractor ----------

_NATO_WORDS = [
    # Variants first where applicable
    "ALPHA", "ALFA",
    "BRAVO",
    "CHARLIE",
    "DELTA",
    "ECHO",
    "FOXTROT",
    "GOLF",
    "HOTEL",
    "INDIA",
    "JULIET", "JULIETT",
    "KILO",
    "LIMA",
    "MIKE",
    "NOVEMBER",
    "OSCAR",
    "PAPA",
    "QUEBEC",
    "ROMEO",
    "SIERRA",
    "TANGO",
    "UNIFORM",
    "VICTOR",
    "WHISKEY",
    "XRAY", "X-RAY",
    "YANKEE",
    "ZULU",
]


def extract_first_nato_word(text: str) -> Optional[str]:
    """Return the first NATO phonetic code word found in `text`.

    - Matches are case-insensitive and must align to word boundaries.
    - Accepts common variants (e.g., ALFA/ALPHA, JULIET/JULIETT, XRAY/X-RAY).
    - Returns the normalized upper-case code word as found in `_NATO_WORDS`.
    - Returns None when no code word is present.
    """

    if not text:
        return None
    import re

    # Build a single regex from the word list with word boundaries.
    # Use alternation ordered as in _NATO_WORDS so variants keep priority.
    pattern = r"\b(?:(%s))\b" % ("|".join(map(re.escape, _NATO_WORDS)))
    regex = re.compile(pattern, flags=re.IGNORECASE)
    match = regex.findall(text)
    if not match:
        return None
    # Normalize to the canonical upper-case variant from the list
    found = match[0]
    found_upper = found.upper()
    return found_upper[0]


# ---------- Transcription pipeline ----------


def _read_api_key(default_path: Path) -> Optional[str]:
    # Prefer environment variable if present
    from_env = os.environ.get("OPENAI_API_KEY")
    if from_env:
        return from_env.strip()
    try:
        key = default_path.read_text(encoding="utf-8").splitlines()[0].strip()
        return key or None
    except Exception:
        return None


def _ffmpeg_available() -> bool:
    try:
        proc = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
        return proc.returncode == 0
    except FileNotFoundError:
        return False


def extract_audio_with_ffmpeg(video_path: Path, audio_out: Path, *, sample_rate: int = 16000) -> Path:
    """Extract mono WAV audio from a video using ffmpeg.

    Requires ffmpeg to be installed and available on PATH.
    """

    audio_out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",  # no video
        "-ac", "1",  # mono
        "-ar", str(sample_rate),  # sample rate
        "-f", "wav",
        str(audio_out),
    ]
    completed = subprocess.run(cmd, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed (exit {completed.returncode}). stderr: {completed.stderr.strip()[:500]}"
        )
    if not audio_out.exists():
        raise FileNotFoundError(f"Expected audio file not created: {audio_out}")
    return audio_out


@dataclass
class TranscriptionResult:
    transcript: str
    first_nato_word: Optional[str]

    def to_dict(self) -> dict:
        return {
            "transcript": self.transcript,
            "first_nato_word": self.first_nato_word,
        }


def transcribe_audio_openai(
    audio_path: Path,
    *,
    model: str = "whisper-1",
    base_url: Optional[str] = None,
    api_key_path: Path = REPO_ROOT / "api_key.txt",
) -> str:
    """Transcribe an audio file using an OpenAI-compatible API.

    - `model` can be "whisper-1", "gpt-4o-mini-transcribe", or any server-supported name.
    - `base_url` can override the API endpoint (e.g., proxies). If not provided,
      uses environment OPENAI_BASE_URL when set; otherwise the library default.
    """

    if OpenAI is None:
        raise RuntimeError("The 'openai' Python package is required but not available.")

    api_key = _read_api_key(api_key_path)
    if not api_key:
        raise RuntimeError(
            "API key not found. Provide OPENAI_API_KEY env var or repo-root api_key.txt."
        )

    base_url = base_url or os.environ.get("OPENAI_BASE_URL")
    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    client = OpenAI(**client_kwargs)

    # Prefer plain text response to avoid SDK object shape differences across servers
    with audio_path.open("rb") as fh:
        try:
            resp = client.audio.transcriptions.create(
                model=model,
                file=fh,
                response_format="text",
            )
        except Exception as exc:
            raise RuntimeError(f"Transcription API call failed: {exc}") from exc

    # Some servers return a dict-like or object; response_format="text" should give a str
    if isinstance(resp, str):
        return resp
    # Fallbacks: look for common shapes
    text = getattr(resp, "text", None) or getattr(resp, "data", None)
    if isinstance(text, str):
        return text
    try:
        return json.dumps(resp)  # last resort: return raw payload as string
    except Exception:
        return str(resp)


def transcribe_audio_local(
    audio_path: Path,
    *,
    model: str = "base",
    language: Optional[str] = None,
) -> str:
    """Transcribe an audio file locally using Whisper.

    Requires `openai-whisper` to be installed (pip install openai-whisper).
    The first run will download model weights (network required once).
    """
    if whisper is None:
        raise RuntimeError(
            "Local transcription requires 'openai-whisper'. Install with: pip install openai-whisper"
        )
    # CPU-friendly defaults; enable fp16 on GPUs if needed by environment
    model_obj = whisper.load_model(model)
    options = dict(language=language) if language else {}
    result = model_obj.transcribe(str(audio_path), **options)
    text = result.get("text") if isinstance(result, dict) else None
    if not text:
        # Fallback to raw repr to aid debugging rather than raising
        try:
            return json.dumps(result)
        except Exception:
            return str(result)
    return text


def run_pipeline(
    video_path: Path,
    *,
    audio_out: Optional[Path] = None,
    model: str = "whisper-1",
    base_url: Optional[str] = None,
    engine: str = "local",
    whisper_model: str = "base",
    language: Optional[str] = None,
) -> TranscriptionResult:
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    # Prepare audio path
    if audio_out is None:
        audio_out = video_path.with_suffix("")
        audio_out = audio_out.parent / (audio_out.name + "_audio.wav")

    if not _ffmpeg_available():
        raise EnvironmentError(
            "ffmpeg is required to extract audio but was not found on PATH."
        )

    wav_path = extract_audio_with_ffmpeg(video_path, audio_out)

    if engine.lower() == "api":
        transcript = transcribe_audio_openai(wav_path, model=model, base_url=base_url)
    elif engine.lower() == "local":
        transcript = transcribe_audio_local(wav_path, model=whisper_model, language=language)
    else:
        raise ValueError("engine must be 'local' or 'api'")
    first_nato = extract_first_nato_word(transcript)
    return TranscriptionResult(transcript=transcript, first_nato_word=first_nato)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Transcribe a video's audio and extract the first NATO code word.")
    p.add_argument("video", type=str, help="Path to the input video file")
    p.add_argument("--audio-out", dest="audio_out", type=str, default=None, help="Optional path to save extracted WAV")
    # Engine selection
    p.add_argument("--engine", choices=["local", "api"], default="local", help="Transcription engine: local (Whisper) or api")
    # API engine options
    p.add_argument("--model", type=str, default="whisper-1", help="API model name (e.g., whisper-1)")
    p.add_argument("--base-url", dest="base_url", type=str, default=None, help="Override API base URL (or use OPENAI_BASE_URL)")
    # Local engine options
    p.add_argument("--whisper-model", dest="whisper_model", type=str, default="base", help="Local Whisper model size (tiny, base, small, medium, large)")
    p.add_argument("--language", dest="language", type=str, default="en", help="Language hint for local Whisper (e.g., en, zh, etc.)")
    p.add_argument("--output-json", dest="output_json", type=str, default=None, help="Write results to JSON file")
    return p


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    video_path = Path(args.video)
    audio_out = Path(args.audio_out) if args.audio_out else None

    try:
        result = run_pipeline(
            video_path,
            audio_out=audio_out,
            model=args.model,
            base_url=args.base_url,
            engine=args.engine,
            whisper_model=args.whisper_model,
            language=args.language,
        )
    except Exception as exc:
        print(f"Error: {exc}")
        return 2

    payload = result.to_dict()

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(out_path.as_posix())
    else:
        # Pretty-print to stdout
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
