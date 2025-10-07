import os
import time
import tempfile
import pathlib
import requests
from typing import Optional
from datetime import datetime
from veo3 import extract_last_frame

OPENAI_API_BASE = "https://jy.ai-wx.cn/v1"
DEFAULT_MODEL = "sora_video2"          # or "sora-2-pro" if enabled on your account
DEFAULT_SECONDS = 10              # typical caps right now are 10–20s depending on plan
DEFAULT_SIZE = "1280x720"         # match your input image aspect; see docs

class SoraError(Exception):
    pass

def _auth_headers() -> dict:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SoraError("OPENAI_API_KEY not set.")
    return {"Authorization": f"Bearer {api_key}"}

def _create_video(prompt: str, image_path: Optional[str], seconds: int, size: str, model: str) -> str:
    """
    POST /v1/videos
    Returns video_id on success.
    """
    url = f"{OPENAI_API_BASE}/videos"
    data = {
        "model": model,
        "prompt": prompt,
        "seconds": str(seconds),
        "size": size,
    }
    files = None
    if image_path:
        # Per Sora cookbook, the guiding image goes in input_reference as a file
        # Supported: jpg/png/webp and resolution should match target size.
        files = {
            "input_reference": (pathlib.Path(image_path).name, open(image_path, "rb"), "application/octet-stream")
        }

    resp = requests.post(url, headers=_auth_headers(), data=data, files=files, timeout=60, proxies={})
    try:
        payload = resp.json()
    except Exception:
        resp.raise_for_status()
        raise

    if resp.status_code >= 400:
        raise SoraError(f"Video creation failed: {payload}")

    vid = payload.get("id") or payload.get("video_id")
    if not vid:
        raise SoraError(f"Malformed create response: {payload}")
    return vid

def _poll_video(video_id: str, timeout_s: int = 600, poll_every_s: float = 2.5) -> dict:
    """
    GET /v1/videos/{id} until status in {'succeeded','failed'}
    Returns the final video object (JSON).
    """
    url = f"{OPENAI_API_BASE}/videos/{video_id}"
    start = time.time()
    while True:
        resp = requests.get(url, headers=_auth_headers(), timeout=30, proxies={})
        payload = resp.json()
        status = payload.get("status") or payload.get("state")
        if status in {"succeeded", "completed", "ready"}:
            return payload
        if status in {"failed", "error"}:
            raise SoraError(f"Video failed: {payload}")

        if time.time() - start > timeout_s:
            raise SoraError(f"Timed out waiting for video {video_id}, last status: {status} payload={payload}")

        time.sleep(poll_every_s)

def _download_video_content(video_id: str, dest_dir: Optional[str] = None) -> str:
    """
    GET /v1/videos/{id}/content streaming MP4 bytes to disk.
    Returns absolute path.
    """
    url = f"{OPENAI_API_BASE}/videos/{video_id}/content"
    with requests.get(url, headers=_auth_headers(), stream=True, timeout=120, proxies={}) as r:
        if r.status_code >= 400:
            try:
                err = r.json()
            except Exception:
                err = {"status": r.status_code}
            raise SoraError(f"Download failed: {err}")

        suffix = ".mp4"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ddir = pathlib.Path(dest_dir) if dest_dir else f'data/output/output_{timestamp}'
        ddir.mkdir(parents=True, exist_ok=True)
        fpath = ddir / f"sora_{video_id}{suffix}"
        with open(fpath, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        extract_last_frame(fpath.as_posix(), ddir.as_posix())
        return str(fpath.resolve())

def generate_video_output(input_image_path, prompt_text) -> str:
    """
    Creates a Sora 2 video from an optional guiding image + prompt, downloads it, and returns local file path.
    Environment: requires OPENAI_API_KEY.
    """
    # Pro tips:
    # - Keep the image aspect ratio aligned with `size`, per cookbook guidance.
    # - The API may throttle; feel free to increase timeout if you live dangerously.
    video_id = _create_video(
        prompt=prompt_text,
        image_path=input_image_path,
        seconds=DEFAULT_SECONDS,
        size=DEFAULT_SIZE,
        model=DEFAULT_MODEL,
    )

    # Poll until the job finishes, or we lose patience with reality.
    _ = _poll_video(video_id, timeout_s=900, poll_every_s=3)

    # Download the final MP4 and hand back the path.
    out_path = _download_video_content(video_id)
    return out_path

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python sora2.py <prompt> [<image_path>]")
        sys.exit(1)
    prompt = sys.argv[1]
    image_path = sys.argv[2] if len(sys.argv) >= 3 else None
    try:
        output_path = generate_video_output(image_path, prompt)
        print(f"Video saved to: {output_path}")
    except SoraError as e:
        print(f"Error: {e}")
        sys.exit(1)