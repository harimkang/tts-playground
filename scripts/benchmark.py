#!/usr/bin/env python3
"""Run the fixed Korean TTS benchmark against the public gateway API."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import re
import wave
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Final

import httpx

RESULT_COLUMNS: Final = [
    "timestamp",
    "case_id",
    "model",
    "repeat",
    "status",
    "latency_ms",
    "queue_ms",
    "processing_ms",
    "audio_duration_s",
    "rtf",
    "response_bytes",
    "error_code",
]

_CASE_ID_PATTERN: Final = re.compile(r"^[a-z0-9][a-z0-9_]*$")
_MAX_TEXT_CHARACTERS: Final = 2_000
_MAX_INSTRUCTION_CHARACTERS: Final = 1_000


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    case_id: str
    text: str
    language: str
    instructions: str | None


@dataclass(frozen=True, slots=True)
class BenchmarkModel:
    model: str
    voice: str
    accepts_instructions: bool


DEFAULT_MODELS: Final = (
    BenchmarkModel("supertonic-3", "F1", False),
    BenchmarkModel("qwen3-tts-1.7b-custom", "Sohee", True),
)


def load_corpus(path: Path) -> list[BenchmarkCase]:
    """Load and validate a corpus without silently coercing invalid values."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read corpus: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError("Corpus must be a JSON object.")

    language = payload.get("language")
    if not isinstance(language, str) or not 2 <= len(language) <= 10:
        raise ValueError("Corpus language must contain 2 to 10 characters.")
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("Corpus cases must be a non-empty list.")

    seen_ids: set[str] = set()
    cases: list[BenchmarkCase] = []
    for index, raw_case in enumerate(raw_cases):
        if not isinstance(raw_case, dict):
            raise ValueError(f"Corpus case {index} must be an object.")
        case_id = raw_case.get("id")
        if not isinstance(case_id, str) or _CASE_ID_PATTERN.fullmatch(case_id) is None:
            raise ValueError(f"Corpus case id at index {index} is invalid.")
        if case_id in seen_ids:
            raise ValueError("Corpus case ids must be unique.")
        seen_ids.add(case_id)

        text = raw_case.get("text")
        if not isinstance(text, str) or not 1 <= len(text) <= _MAX_TEXT_CHARACTERS:
            raise ValueError(
                f"Corpus case '{case_id}' text must contain 1 to {_MAX_TEXT_CHARACTERS} characters."
            )
        instructions = raw_case.get("instructions")
        if instructions is not None and (
            not isinstance(instructions, str)
            or not 1 <= len(instructions) <= _MAX_INSTRUCTION_CHARACTERS
        ):
            raise ValueError(
                f"Corpus case '{case_id}' instructions must contain 1 to "
                f"{_MAX_INSTRUCTION_CHARACTERS} characters."
            )
        cases.append(BenchmarkCase(case_id, text, language, instructions))
    return cases


def wav_duration_seconds(content: bytes) -> float:
    """Return the duration encoded in a PCM WAV header."""
    try:
        with wave.open(io.BytesIO(content), "rb") as wav_file:
            frame_rate = wav_file.getframerate()
            frame_count = wav_file.getnframes()
    except (EOFError, wave.Error) as error:
        raise ValueError("Response is not a readable WAV file.") from error
    if frame_rate <= 0 or frame_count <= 0:
        raise ValueError("WAV response contains no audio frames.")
    return frame_count / frame_rate


def calculate_rtf(*, latency_ms: float, audio_duration_s: float) -> float:
    """Calculate end-to-end real-time factor."""
    if latency_ms < 0 or audio_duration_s <= 0:
        raise ValueError("RTF inputs must be non-negative and duration must be positive.")
    return latency_ms / 1_000 / audio_duration_s


def read_api_key(environ: Mapping[str, str] | None = None) -> str:
    """Read the bearer token without writing it to output or command arguments."""
    environment = os.environ if environ is None else environ
    key_file = environment.get("TTS_API_KEY_FILE")
    if key_file:
        try:
            key = Path(key_file).read_text(encoding="utf-8").strip()
        except OSError as error:
            raise ValueError(f"Could not read TTS_API_KEY_FILE: {key_file}") from error
    else:
        key = environment.get("TTS_API_KEY", "").strip()
    if not key:
        raise ValueError("Set TTS_API_KEY_FILE or TTS_API_KEY to a non-empty API key.")
    return key


def run_benchmark(
    *,
    cases: Sequence[BenchmarkCase],
    client: httpx.Client,
    base_url: str,
    output_directory: Path,
    timestamp: str,
    repeats: int = 3,
    models: Sequence[BenchmarkModel] = DEFAULT_MODELS,
) -> list[dict[str, Any]]:
    """Run every request serially and return one complete row per attempt."""
    if repeats < 1:
        raise ValueError("repeats must be at least one.")
    if not cases or not models:
        raise ValueError("cases and models must be non-empty.")
    endpoint = f"{base_url.rstrip('/')}/v1/audio/speech"
    if not base_url.strip():
        raise ValueError("base_url must be non-empty.")
    audio_directory = output_directory / "audio"
    audio_directory.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for model in models:
        for case in cases:
            for repeat in range(1, repeats + 1):
                payload: dict[str, object] = {
                    "model": model.model,
                    "input": case.text,
                    "voice": model.voice,
                    "language": case.language,
                    "response_format": "wav",
                }
                if model.accepts_instructions and case.instructions is not None:
                    payload["instructions"] = case.instructions

                started_at = perf_counter()
                try:
                    response = client.post(endpoint, json=payload)
                    latency_ms = (perf_counter() - started_at) * 1_000
                    row = _response_row(
                        response=response,
                        latency_ms=latency_ms,
                        timestamp=timestamp,
                        case=case,
                        model=model,
                        repeat=repeat,
                        audio_directory=audio_directory,
                    )
                except httpx.RequestError as error:
                    latency_ms = (perf_counter() - started_at) * 1_000
                    row = _base_row(
                        timestamp=timestamp,
                        case=case,
                        model=model,
                        repeat=repeat,
                        status=0,
                        latency_ms=latency_ms,
                    )
                    row["error_code"] = _request_error_code(error)
                rows.append(row)
    return rows


def write_results(
    rows: Sequence[Mapping[str, object]], output_directory: Path
) -> tuple[Path, Path]:
    """Write equivalent JSONL and CSV result sets with a stable schema."""
    output_directory.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_directory / "results.jsonl"
    csv_path = output_directory / "results.csv"

    with jsonl_path.open("w", encoding="utf-8") as jsonl_file:
        for row in rows:
            _validate_result_columns(row)
            jsonl_file.write(json.dumps(dict(row), ensure_ascii=False, separators=(",", ":")))
            jsonl_file.write("\n")

    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        for row in rows:
            _validate_result_columns(row)
            writer.writerow(row)
    return jsonl_path, csv_path


def _response_row(
    *,
    response: httpx.Response,
    latency_ms: float,
    timestamp: str,
    case: BenchmarkCase,
    model: BenchmarkModel,
    repeat: int,
    audio_directory: Path,
) -> dict[str, Any]:
    row = _base_row(
        timestamp=timestamp,
        case=case,
        model=model,
        repeat=repeat,
        status=response.status_code,
        latency_ms=latency_ms,
    )
    row["response_bytes"] = len(response.content)
    row["queue_ms"] = _nonnegative_header(response, "X-Queue-Ms")
    row["processing_ms"] = _nonnegative_header(response, "X-Processing-Ms")

    if not response.is_success:
        row["error_code"] = _error_code(response)
        return row
    try:
        audio_duration_s = wav_duration_seconds(response.content)
    except ValueError:
        row["error_code"] = "invalid_wav"
        return row

    row["audio_duration_s"] = round(audio_duration_s, 6)
    row["rtf"] = round(
        calculate_rtf(latency_ms=latency_ms, audio_duration_s=audio_duration_s), 6
    )
    audio_path = audio_directory / f"{case.case_id}__{model.model}__r{repeat}.wav"
    audio_path.write_bytes(response.content)
    return row


def _base_row(
    *,
    timestamp: str,
    case: BenchmarkCase,
    model: BenchmarkModel,
    repeat: int,
    status: int,
    latency_ms: float,
) -> dict[str, Any]:
    return {
        "timestamp": timestamp,
        "case_id": case.case_id,
        "model": model.model,
        "repeat": repeat,
        "status": status,
        "latency_ms": round(latency_ms, 3),
        "queue_ms": None,
        "processing_ms": None,
        "audio_duration_s": None,
        "rtf": None,
        "response_bytes": 0,
        "error_code": None,
    }


def _nonnegative_header(response: httpx.Response, name: str) -> float | None:
    value = response.headers.get(name)
    if value is None:
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    if not math.isfinite(number) or number < 0:
        return None
    return round(number, 3)


def _error_code(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return "http_error"
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict) and isinstance(error.get("code"), str):
            return str(error["code"])
    return "http_error"


def _request_error_code(error: httpx.RequestError) -> str:
    if isinstance(error, httpx.TimeoutException):
        return "request_timeout"
    return "request_error"


def _validate_result_columns(row: Mapping[str, object]) -> None:
    if list(row) != RESULT_COLUMNS:
        raise ValueError("Result row columns do not match RESULT_COLUMNS.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        default=os.environ.get("TTS_BASE_URL", "http://127.0.0.1:8020"),
        help="Gateway base URL (default: TTS_BASE_URL or http://127.0.0.1:8020)",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("benchmarks/corpus.ko.json"),
        help="Validated benchmark corpus",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results"),
        help="Parent directory for timestamped benchmark results",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=float, default=420.0)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        api_key = read_api_key()
        cases = load_corpus(args.corpus)
        if args.timeout_seconds <= 0:
            raise ValueError("timeout-seconds must be positive.")
        run_started = datetime.now(UTC)
        timestamp = run_started.isoformat(timespec="milliseconds").replace("+00:00", "Z")
        directory_name = run_started.strftime("%Y%m%dT%H%M%S.%fZ")
        output_directory = args.output_root / directory_name
        with httpx.Client(
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=args.timeout_seconds,
        ) as client:
            rows = run_benchmark(
                cases=cases,
                client=client,
                base_url=args.base_url,
                output_directory=output_directory,
                timestamp=timestamp,
                repeats=args.repeats,
            )
        jsonl_path, csv_path = write_results(rows, output_directory)
    except ValueError as error:
        raise SystemExit(str(error)) from error

    successful = sum(row["status"] == 200 and row["error_code"] is None for row in rows)
    print(f"Completed {len(rows)} requests ({successful} successful).")
    print(f"JSONL: {jsonl_path}")
    print(f"CSV: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
