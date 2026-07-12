import csv
import importlib.util
import io
import json
import sys
import wave
from pathlib import Path

import httpx
import pytest


def _load_benchmark_module() -> object:
    path = Path(__file__).parents[2] / "scripts" / "benchmark.py"
    spec = importlib.util.spec_from_file_location("tts_benchmark", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load benchmark module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


benchmark = _load_benchmark_module()


def _wav_bytes(*, frames: int = 8_000, sample_rate: int = 8_000) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"\x00\x00" * frames)
    return buffer.getvalue()


def test_load_corpus_validates_shape_and_unique_case_ids(tmp_path: Path) -> None:
    corpus_path = tmp_path / "corpus.json"
    corpus_path.write_text(
        json.dumps(
            {
                "language": "ko",
                "cases": [
                    {"id": "short", "text": "안녕하세요."},
                    {
                        "id": "emotional_instruction",
                        "text": "정말 반가워요!",
                        "instructions": "밝고 반가운 어조로 말해 주세요.",
                    },
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    cases = benchmark.load_corpus(corpus_path)

    assert [case.case_id for case in cases] == ["short", "emotional_instruction"]
    assert cases[0].language == "ko"

    corpus_path.write_text(
        json.dumps(
            {
                "language": "ko",
                "cases": [
                    {"id": "duplicate", "text": "첫 번째"},
                    {"id": "duplicate", "text": "두 번째"},
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unique"):
        benchmark.load_corpus(corpus_path)


def test_checked_in_korean_corpus_contains_required_named_cases() -> None:
    corpus_path = Path(__file__).parents[2] / "benchmarks" / "corpus.ko.json"

    cases = benchmark.load_corpus(corpus_path)

    assert {case.case_id for case in cases}.issuperset(
        {
            "short",
            "medium",
            "long",
            "numbers_dates_currency",
            "mixed_korean_english",
            "emotional_instruction",
        }
    )
    assert all(case.language == "ko" for case in cases)
    emotional = next(case for case in cases if case.case_id == "emotional_instruction")
    assert emotional.instructions


@pytest.mark.parametrize(
    "payload, message",
    [
        ({"language": "ko", "cases": []}, "non-empty"),
        ({"language": "ko", "cases": [{"id": "bad id", "text": "본문"}]}, "case id"),
        ({"language": "ko", "cases": [{"id": "short", "text": ""}]}, "text"),
        ({"language": "k", "cases": [{"id": "short", "text": "본문"}]}, "language"),
    ],
)
def test_load_corpus_rejects_invalid_values(
    tmp_path: Path, payload: dict[str, object], message: str
) -> None:
    corpus_path = tmp_path / "corpus.json"
    corpus_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        benchmark.load_corpus(corpus_path)


def test_wav_duration_and_rtf_are_calculated_from_audio() -> None:
    audio = _wav_bytes(frames=12_000, sample_rate=8_000)

    duration = benchmark.wav_duration_seconds(audio)

    assert duration == pytest.approx(1.5)
    assert benchmark.calculate_rtf(latency_ms=750.0, audio_duration_s=duration) == pytest.approx(
        0.5
    )


def test_run_benchmark_defaults_to_three_serial_repeats_and_captures_headers(
    tmp_path: Path,
) -> None:
    audio = _wav_bytes()
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            content=audio,
            headers={
                "Content-Type": "audio/wav",
                "X-Queue-Ms": "12.5",
                "X-Processing-Ms": "345.25",
                "X-Audio-Duration": "999",  # The WAV remains the source of truth.
            },
        )

    case = benchmark.BenchmarkCase(
        case_id="short",
        text="안녕하세요.",
        language="ko",
        instructions=None,
    )
    transport = httpx.MockTransport(handler)
    with httpx.Client(
        transport=transport,
        headers={"Authorization": "Bearer benchmark-secret"},
    ) as client:
        rows = benchmark.run_benchmark(
            cases=[case],
            client=client,
            base_url="http://tts.example/",
            output_directory=tmp_path,
            timestamp="2026-07-12T00:00:00Z",
        )

    assert len(rows) == 6
    assert [row["repeat"] for row in rows] == [1, 2, 3, 1, 2, 3]
    assert [row["model"] for row in rows] == [
        "supertonic-3",
        "supertonic-3",
        "supertonic-3",
        "qwen3-tts-1.7b-custom",
        "qwen3-tts-1.7b-custom",
        "qwen3-tts-1.7b-custom",
    ]
    assert all(row["status"] == 200 for row in rows)
    assert all(row["queue_ms"] == 12.5 for row in rows)
    assert all(row["processing_ms"] == 345.25 for row in rows)
    assert all(row["audio_duration_s"] == pytest.approx(1.0) for row in rows)
    assert all(row["rtf"] is not None for row in rows)
    assert all(row["response_bytes"] == len(audio) for row in rows)
    assert all(
        request.headers["authorization"] == "Bearer benchmark-secret"
        for request in requests
    )
    assert len(list((tmp_path / "audio").glob("*.wav"))) == 6


def test_run_benchmark_records_http_errors_and_continues(tmp_path: Path) -> None:
    call_count = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return httpx.Response(
            503,
            json={"error": {"code": "insufficient_vram", "message": "not recorded"}},
        )

    case = benchmark.BenchmarkCase("short", "안녕하세요.", "ko", None)
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        rows = benchmark.run_benchmark(
            cases=[case],
            client=client,
            base_url="http://tts.example",
            output_directory=tmp_path,
            timestamp="2026-07-12T00:00:00Z",
        )

    assert call_count == 6
    assert len(rows) == 6
    assert all(row["status"] == 503 for row in rows)
    assert all(row["error_code"] == "insufficient_vram" for row in rows)
    assert not list((tmp_path / "audio").glob("*.wav"))


@pytest.mark.parametrize(
    ("exception_type", "expected_error_code"),
    [
        (httpx.ConnectError, "request_error"),
        (httpx.ReadTimeout, "request_timeout"),
    ],
)
def test_run_benchmark_records_request_failures_and_continues_subsequent_attempts(
    tmp_path: Path,
    exception_type: type[httpx.RequestError],
    expected_error_code: str,
) -> None:
    audio = _wav_bytes()
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise exception_type("injected request failure", request=request)
        return httpx.Response(200, content=audio, headers={"Content-Type": "audio/wav"})

    case = benchmark.BenchmarkCase("short", "안녕하세요.", "ko", None)
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        rows = benchmark.run_benchmark(
            cases=[case],
            client=client,
            base_url="http://tts.example",
            output_directory=tmp_path,
            timestamp="2026-07-12T00:00:00Z",
        )

    assert call_count == 6
    assert len(rows) == 6
    assert rows[0]["status"] == 0
    assert rows[0]["error_code"] == expected_error_code
    assert all(row["status"] == 200 for row in rows[1:])
    assert all(row["error_code"] is None for row in rows[1:])
    assert len(list((tmp_path / "audio").glob("*.wav"))) == 5


def test_write_results_uses_required_csv_columns(tmp_path: Path) -> None:
    row = {
        "timestamp": "2026-07-12T00:00:00Z",
        "case_id": "short",
        "model": "supertonic-3",
        "repeat": 1,
        "status": 200,
        "latency_ms": 100.0,
        "queue_ms": 0.0,
        "processing_ms": 80.0,
        "audio_duration_s": 1.0,
        "rtf": 0.1,
        "response_bytes": 16_044,
        "error_code": None,
    }

    jsonl_path, csv_path = benchmark.write_results([row], tmp_path)

    assert json.loads(jsonl_path.read_text(encoding="utf-8")) == row
    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        assert reader.fieldnames == benchmark.RESULT_COLUMNS
        assert list(reader)[0]["case_id"] == "short"


def test_read_api_key_prefers_file_and_never_prints_secret(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    key_file = tmp_path / "api-key"
    key_file.write_text("file-secret\n", encoding="utf-8")

    key = benchmark.read_api_key(
        {"TTS_API_KEY_FILE": str(key_file), "TTS_API_KEY": "environment-secret"}
    )

    assert key == "file-secret"
    captured = capsys.readouterr()
    assert "file-secret" not in captured.out + captured.err
    assert "environment-secret" not in captured.out + captured.err
