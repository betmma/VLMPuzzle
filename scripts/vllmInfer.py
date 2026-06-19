#!/usr/bin/env python3
"""Send visual prompt records to an OpenAI-compatible vLLM server.

Each request is built from one metadata record and contains the record prompt
plus the corresponding image. Every response gets its own timestamped directory
under the configured output root.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional

import requests


DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_MODEL = "Qwen3.5"
DEFAULT_METADATA = Path("data/arcagi/data.json")
DEFAULT_OUTPUT_ROOT = Path("output")


@dataclass
class VllmInferenceConfig:
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    output_root: str = str(DEFAULT_OUTPUT_ROOT)
    timeout: int = 600
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    stream: bool = False
    request_retries: int = 1
    retry_delay: float = 0.0
    api_key: str = ""
    no_proxy: bool = True


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def image_mime_type(image_path: Path) -> str:
    ext = image_path.suffix.lower()
    if ext in (".jpg", ".jpeg"):
        return "image/jpeg"
    if ext == ".webp":
        return "image/webp"
    return "image/png"


def image_to_data_url(image_path: Path) -> str:
    encoded = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return f"data:{image_mime_type(image_path)};base64,{encoded}"


def build_messages(prompt_text: str, image_path: Path) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {"type": "image_url", "image_url": {"url": image_to_data_url(image_path)}},
            ],
        }
    ]


def create_output_directory(output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    for counter in range(10000):
        suffix = "" if counter == 0 else f"_{counter:04d}"
        candidate = output_root / f"output_{timestamp}{suffix}"
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        except FileExistsError:
            continue
    raise RuntimeError(f"Could not create a unique output directory under {output_root}")


def public_config(config: VllmInferenceConfig) -> dict[str, Any]:
    data = asdict(config)
    data["api_key"] = "***" if data["api_key"] else ""
    return data


def clean_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if value is not None}


def request_with_retries(
    method: str,
    url: str,
    *,
    attempts: int,
    retry_delay: float,
    **kwargs: Any,
) -> requests.Response:
    last_error: Optional[BaseException] = None
    for attempt in range(1, attempts + 1):
        try:
            response = requests.request(method, url, proxies={}, **kwargs)
            response.raise_for_status()
            return response
        except requests.RequestException as error:
            last_error = error
            if attempt == attempts:
                break
            if retry_delay > 0:
                time.sleep(retry_delay)
    if last_error is None:
        raise RuntimeError("Request failed without an exception")
    raise last_error


def read_streaming_response(response: requests.Response, output_dir: Path) -> dict[str, Any]:
    full_content = ""
    chunks: list[dict[str, Any]] = []
    for line in response.iter_lines():
        if not line:
            continue
        line_text = line.decode("utf-8")
        if not line_text.startswith("data: "):
            continue
        data_text = line_text[6:]
        if data_text == "[DONE]":
            continue
        try:
            chunk = json.loads(data_text)
        except json.JSONDecodeError:
            continue
        chunks.append(chunk)
        choices = chunk.get("choices", [])
        if choices:
            delta = choices[0].get("delta", {})
            full_content += delta.get("content", "") or ""
    write_json(output_dir / "stream_chunks.json", chunks)
    return {"choices": [{"message": {"role": "assistant", "content": full_content}}], "stream_chunks": chunks}


def extract_response_content(response: Any) -> str:
    if not isinstance(response, dict):
        return str(response)
    choices = response.get("choices", [])
    if not choices:
        return ""
    message = choices[0].get("message", {})
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(parts)
    return str(content)


def build_payload(config: VllmInferenceConfig, messages: list[dict[str, Any]]) -> dict[str, Any]:
    return clean_payload(
        {
            "model": config.model,
            "messages": messages,
            "stream": config.stream,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "max_tokens": config.max_tokens,
        }
    )


def call_vllm(config: VllmInferenceConfig, messages: list[dict[str, Any]], output_dir: Path) -> dict[str, Any]:
    url = f"{config.base_url.rstrip('/')}/chat/completions"
    headers = {"Content-Type": "application/json"}
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"
    payload = build_payload(config, messages)
    write_json(output_dir / "request_payload.json", payload)
    response = request_with_retries(
        "POST",
        url,
        attempts=config.request_retries,
        retry_delay=config.retry_delay,
        headers=headers,
        json=payload,
        timeout=config.timeout,
        stream=config.stream,
    )
    if config.stream:
        return read_streaming_response(response, output_dir)
    return response.json()


def write_input_text(output_dir: Path, record: dict[str, Any], image_path: Path, prompt_text: str) -> None:
    record_id = str(record.get("id") or record.get("task_id") or "")
    lines = [
        f"Record id: {record_id}",
        f"Input image path: {image_path.as_posix()}",
        "",
        "Prompt:",
        prompt_text,
        "",
    ]
    write_text(output_dir / "input.txt", "\n".join(lines))


def write_metadata(
    output_dir: Path,
    config: VllmInferenceConfig,
    record: dict[str, Any],
    image_path: Path,
    prompt_text: str,
    *,
    success: bool,
    content: str = "",
    error: Optional[str] = None,
) -> None:
    artifacts = {
        "content_path": (output_dir / "content.txt").as_posix(),
        "original_content_path": (output_dir / "original_content.txt").as_posix(),
        "response_content_preview_path": (output_dir / "response_content_preview.txt").as_posix(),
    }
    metadata = {
        "success": success,
        "error": error,
        "created_at": datetime.now().isoformat(),
        "record_id": record.get("id"),
        "task_id": record.get("task_id"),
        "task_path": record.get("task_path"),
        "input_images": [image_path.as_posix()],
        "prompt": prompt_text,
        "config": public_config(config),
        "content_length": len(content),
        "artifacts": artifacts,
    }
    write_json(output_dir / "metadata.json", metadata)


def save_success_artifacts(
    output_dir: Path,
    config: VllmInferenceConfig,
    record: dict[str, Any],
    image_path: Path,
    prompt_text: str,
    response_payload: dict[str, Any],
) -> str:
    content = extract_response_content(response_payload)
    write_json(output_dir / "raw_api_response.json", response_payload)
    write_text(output_dir / "response_content_preview.txt", content[:4000])
    write_text(output_dir / "content.txt", content)
    write_text(output_dir / "original_content.txt", content)
    write_metadata(output_dir, config, record, image_path, prompt_text, success=True, content=content)
    return content


def read_metadata(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Metadata must be a list of records: {path}")
    records: list[dict[str, Any]] = []
    for index, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Metadata item {index} is not an object")
        records.append(item)
    return records


def resolve_image_path(metadata_path: Path, record: dict[str, Any]) -> Path:
    image_rel = record.get("image")
    if not isinstance(image_rel, str) or not image_rel:
        raise ValueError(f"Record {record.get('id')!r} missing image")
    image_path = (metadata_path.parent / image_rel).resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")
    return image_path


def selected_records(
    records: list[dict[str, Any]],
    *,
    start: Optional[int],
    end: Optional[int],
    limit: Optional[int],
    ids: Optional[set[str]],
) -> Iterable[tuple[int, dict[str, Any]]]:
    if start is not None and start <= 0:
        raise ValueError("--start must be a positive 1-based index")
    if end is not None and end <= 0:
        raise ValueError("--end must be a positive 1-based index")
    if start is not None and end is not None and end < start:
        raise ValueError("--end must be >= --start")
    start_index = 0 if start is None else start - 1
    end_index = len(records) if end is None else end
    emitted = 0
    for zero_index, record in enumerate(records[start_index:end_index], start=start_index):
        record_id = str(record.get("id") or record.get("task_id") or "")
        if ids is not None and record_id not in ids:
            continue
        yield zero_index + 1, record
        emitted += 1
        if limit is not None and emitted >= limit:
            break


def generate_one(
    config: VllmInferenceConfig,
    metadata_path: Path,
    record: dict[str, Any],
    *,
    attempt: int,
) -> Path:
    prompt_value = record.get('gpt5_prompt') or record.get("prompt")
    if not isinstance(prompt_value, str) or not prompt_value.strip():
        raise ValueError(f"Record {record.get('id')!r} missing prompt")
    prompt_text = prompt_value.strip()
    image_path = resolve_image_path(metadata_path, record)
    output_dir = create_output_directory(Path(config.output_root))
    write_input_text(output_dir, record, image_path, prompt_text)
    messages = build_messages(prompt_text, image_path)
    write_json(output_dir / "input_messages.json", messages)
    try:
        response_payload = call_vllm(config, messages, output_dir)
        save_success_artifacts(output_dir, config, record, image_path, prompt_text, response_payload)
    except Exception as error:
        write_text(output_dir / "raw_api_error.txt", str(error))
        write_metadata(output_dir, config, record, image_path, prompt_text, success=False, error=str(error))
        raise
    write_json(
        output_dir / "run_record.json",
        {
            "record": record,
            "attempt": attempt,
            "output_directory": output_dir.as_posix(),
        },
    )
    return output_dir


def generate_task(task: dict[str, Any]) -> tuple[str, int, Path]:
    config = task["config"]
    metadata_path = task["metadata_path"]
    record = task["record"]
    attempt = task["attempt"]
    metadata_index = task["metadata_index"]
    record_id = str(record.get("id") or record.get("task_id") or f"index_{metadata_index}")
    output_dir = generate_one(config, metadata_path, record, attempt=attempt)
    return record_id, attempt, output_dir


def build_tasks(
    config: VllmInferenceConfig,
    metadata_path: Path,
    selected: list[tuple[int, dict[str, Any]]],
    attempts: int,
) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for attempt in range(1, attempts + 1):
        for metadata_index, record in selected:
            tasks.append(
                {
                    "config": config,
                    "metadata_path": metadata_path,
                    "metadata_index": metadata_index,
                    "record": record,
                    "attempt": attempt,
                }
            )
    return tasks


def run_tasks_parallel(tasks: list[dict[str, Any]], *, workers: int, delay: float) -> list[Path]:
    total = len(tasks)
    completed = 0
    output_dirs: list[Path] = []
    errors: list[tuple[str, int, BaseException]] = []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_task = {}
        for task in tasks:
            record = task["record"]
            metadata_index = task["metadata_index"]
            record_id = record.get("id") or record.get("task_id") or f"index_{metadata_index}"
            future = executor.submit(generate_task, task)
            future_to_task[future] = task
            if delay > 0 and len(future_to_task) < total:
                time.sleep(delay)

        for future in as_completed(future_to_task):
            task = future_to_task[future]
            record = task["record"]
            metadata_index = task["metadata_index"]
            record_id = str(record.get("id") or record.get("task_id") or f"index_{metadata_index}")
            attempt = int(task["attempt"])
            try:
                done_record_id, done_attempt, output_dir = future.result()
                completed += 1
                output_dirs.append(output_dir)
                print(f"[done {completed}/{total}] {done_record_id} attempt {done_attempt}: {output_dir}")
            except Exception as error:
                completed += 1
                errors.append((record_id, attempt, error))
                print(f"[failed {completed}/{total}] {record_id} attempt {attempt}: {error}")

    if errors:
        summary = "; ".join(f"{record_id} attempt {attempt}: {error}" for record_id, attempt, error in errors[:5])
        extra = "" if len(errors) <= 5 else f"; plus {len(errors) - 5} more"
        raise RuntimeError(f"{len(errors)} request(s) failed: {summary}{extra}")
    return output_dirs


def run_tasks_serial(tasks: list[dict[str, Any]], *, delay: float) -> list[Path]:
    total = len(tasks)
    output_dirs: list[Path] = []
    for index, task in enumerate(tasks, start=1):
        record = task["record"]
        metadata_index = task["metadata_index"]
        record_id = record.get("id") or record.get("task_id") or f"index_{metadata_index}"
        print(f"[{index}/{total}] sending {record_id} attempt {task['attempt']}")
        _, _, output_dir = generate_task(task)
        output_dirs.append(output_dir)
        print(f"    wrote {output_dir}")
        if delay > 0 and index < total:
            time.sleep(delay)
    return output_dirs


def parse_ids(value: Optional[str]) -> Optional[set[str]]:
    if value is None:
        return None
    ids = {part.strip() for part in value.split(",") if part.strip()}
    return ids or None


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA, help="Path to metadata JSON")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Directory where output_* folders are written")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="OpenAI-compatible vLLM base URL")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model name as exposed by the vLLM server")
    parser.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY", ""), help="Optional bearer token")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--stream", action="store_true", help="Use streaming responses and save stream_chunks.json")
    parser.add_argument("--request-retries", type=int, default=1)
    parser.add_argument("--retry-delay", type=float, default=0.0)
    parser.add_argument("--start", type=int, default=None, help="1-based start index in metadata, inclusive")
    parser.add_argument("--end", type=int, default=None, help="1-based end index in metadata, inclusive")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of selected records to send")
    parser.add_argument("--ids", type=str, default=None, help="Comma-separated puzzle ids to send")
    parser.add_argument("--attempts", type=int, default=1, help="Requests per selected record")
    parser.add_argument("--workers", type=int, default=16, help="Parallel request workers. Use 1 for serial execution")
    parser.add_argument("--delay", type=float, default=0.0, help="Seconds to wait between request submissions")
    parser.add_argument("--no-proxy", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    if args.attempts <= 0:
        raise ValueError("--attempts must be positive")
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    metadata_path = args.metadata.resolve()
    records = read_metadata(metadata_path)
    config = VllmInferenceConfig(
        base_url=args.base_url,
        model=args.model,
        output_root=str(args.output_root),
        timeout=args.timeout,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stream=args.stream,
        request_retries=args.request_retries,
        retry_delay=args.retry_delay,
        api_key=args.api_key,
        no_proxy=args.no_proxy,
    )
    if config.no_proxy:
        os.environ["NO_PROXY"] = "*"

    ids = parse_ids(args.ids)
    selected = list(selected_records(records, start=args.start, end=args.end, limit=args.limit, ids=ids))
    if not selected:
        raise ValueError("No records selected")

    tasks = build_tasks(config, metadata_path, selected, args.attempts)
    if args.workers == 1:
        run_tasks_serial(tasks, delay=args.delay)
    else:
        run_tasks_parallel(tasks, workers=args.workers, delay=args.delay)
    print("Done.")


if __name__ == "__main__":
    main()
