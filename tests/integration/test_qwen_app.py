from __future__ import annotations

import asyncio
import io
import re
import threading
import time
import warnings
from pathlib import Path

import httpx
import numpy as np
import pytest
import soundfile as sf
from prometheus_client.parser import text_string_to_metric_families
from starlette.exceptions import StarletteDeprecationWarning

with warnings.catch_warnings():
    warnings.simplefilter("ignore", StarletteDeprecationWarning)
    from starlette.testclient import TestClient

from tts_playground.common.models import ModelID
from tts_playground.common.settings import QwenSettings
from tts_playground.qwen.adapter import InternalSpeechRequest, SynthesisResult
from tts_playground.qwen.app import create_qwen_app


class FakeAdapter:
    def __init__(self) -> None:
        self.free_vram = 12 * 1024**3
        self.load_error: BaseException | None = None
        self.build_error: BaseException | None = None
        self.synthesis_error: BaseException | None = None
        self.synthesis_requests: list[InternalSpeechRequest] = []
        self.build_calls: list[tuple[Path, str | None]] = []
        self.paths_existing_during_build: list[bool] = []
        self.block_build = False
        self.build_started = threading.Event()
        self.release_build = threading.Event()
        self.device_memory_cleaned = threading.Event()
        self.audio = _wav_bytes(duration_seconds=1.25, sample_rate=16_000)

    def free_vram_bytes(self) -> int:
        return self.free_vram

    def load(self, repository: str, attention_implementation: str) -> object:
        if self.load_error is not None:
            raise self.load_error
        return {"repository": repository, "attention": attention_implementation}

    def cleanup_device_memory(self) -> None:
        self.device_memory_cleaned.set()

    def build_clone_prompt(
        self,
        model: object,
        audio_path: Path,
        transcript: str | None,
    ) -> object:
        self.build_calls.append((audio_path, transcript))
        self.paths_existing_during_build.append(audio_path.exists())
        if self.block_build:
            self.build_started.set()
            if not self.release_build.wait(timeout=2.0):
                raise AssertionError("test did not release blocked prompt build")
        if self.build_error is not None:
            raise self.build_error
        return (str(audio_path), transcript)

    def synthesize(
        self,
        model: object,
        request: InternalSpeechRequest,
        clone_prompt: object | None,
    ) -> SynthesisResult:
        if self.synthesis_error is not None:
            raise self.synthesis_error
        self.synthesis_requests.append(request)
        return SynthesisResult(
            audio=self.audio,
            sample_rate=16_000,
            duration_seconds=1.25,
        )


def _wav_bytes(duration_seconds: float = 4.0, sample_rate: int = 8_000) -> bytes:
    audio = np.zeros(round(duration_seconds * sample_rate), dtype=np.float32)
    output = io.BytesIO()
    sf.write(output, audio, sample_rate, format="WAV", subtype="PCM_16")
    return output.getvalue()


def _settings(tmp_path: Path, **updates: object) -> QwenSettings:
    return QwenSettings(
        voice_data_dir=tmp_path / "voices",
        min_free_vram_bytes=0,
        **updates,
    )


def _speech_body() -> dict[str, object]:
    return {
        "model": "qwen3-tts-1.7b-custom",
        "text": "안녕하세요",
        "language": "Korean",
        "voice": "Sohee",
        "instructions": "차분하게",
    }


def _clone_form(**updates: str) -> dict[str, str]:
    form = {
        "input": "새로 합성할 문장",
        "consent_confirmed": "true",
        "transcript": "exact text",
        "language": "auto",
        "response_format": "wav",
    }
    form.update(updates)
    return form


def _clone_multipart_parts() -> list[tuple[str, tuple[None, str] | tuple[str, bytes, str]]]:
    return [
        ("input", (None, "새로 합성할 문장")),
        ("consent_confirmed", (None, "true")),
        ("transcript", (None, "exact text")),
        ("language", (None, "auto")),
        ("response_format", (None, "wav")),
        ("audio", ("reference.wav", _wav_bytes(), "audio/wav")),
    ]


def _registry_files(voice_data_dir: Path) -> set[Path]:
    return {
        path.relative_to(voice_data_dir)
        for path in voice_data_dir.rglob("*")
    }


def test_app_construction_does_not_load_model_weights(tmp_path: Path) -> None:
    adapter = FakeAdapter()

    app = create_qwen_app(_settings(tmp_path), adapter)

    assert app.state.qwen_manager is None
    assert adapter.synthesis_requests == []


def test_health_ready_and_initial_status(tmp_path: Path) -> None:
    app = create_qwen_app(_settings(tmp_path), FakeAdapter())

    with TestClient(app) as client:
        assert client.get("/healthz").json() == {"status": "ok"}
        ready = client.get("/readyz")
        status = client.get("/internal/v1/status")

    assert ready.status_code == 200
    assert ready.json() == {"status": "ready", "state": "unloaded"}
    assert status.status_code == 200
    assert status.json() == {
        "state": "unloaded",
        "loaded_model": None,
        "queue_depth": 0,
        "last_error_code": None,
    }


def test_internal_speech_returns_wav_headers_and_propagates_request_id(
    tmp_path: Path,
) -> None:
    adapter = FakeAdapter()
    app = create_qwen_app(_settings(tmp_path), adapter)
    request_id = "req_qwen-worker"

    with TestClient(app) as client:
        response = client.post(
            "/internal/v1/speech",
            headers={"X-Request-ID": request_id},
            json=_speech_body(),
        )
        status = client.get("/internal/v1/status")

    assert response.status_code == 200
    assert response.content == adapter.audio
    assert response.headers["content-type"] == "audio/wav"
    assert response.headers["x-request-id"] == request_id
    assert response.headers["x-tts-model"] == "qwen3-tts-1.7b-custom"
    assert response.headers["x-audio-duration"] == "1.25"
    assert float(response.headers["x-processing-ms"]) >= 0
    assert adapter.synthesis_requests == [
        InternalSpeechRequest(
            model=ModelID.QWEN_CUSTOM,
            text="안녕하세요",
            language="Korean",
            voice="Sohee",
            instructions="차분하게",
        )
    ]
    assert status.json()["loaded_model"] == "qwen3-tts-1.7b-custom"


def test_invalid_request_id_is_replaced(tmp_path: Path) -> None:
    app = create_qwen_app(_settings(tmp_path), FakeAdapter())

    with TestClient(app) as client:
        response = client.post(
            "/internal/v1/speech",
            headers={"X-Request-ID": "invalid"},
            json=_speech_body(),
        )

    assert re.fullmatch(r"req_[0-9a-f]{32}", response.headers["x-request-id"])


def test_model_error_and_readiness_use_internal_503_error_envelope(tmp_path: Path) -> None:
    adapter = FakeAdapter()
    adapter.load_error = RuntimeError("private model detail")
    app = create_qwen_app(_settings(tmp_path), adapter)
    request_id = "req_qwen-error"

    with TestClient(app) as client:
        response = client.post(
            "/internal/v1/speech",
            headers={"X-Request-ID": request_id},
            json=_speech_body(),
        )
        ready = client.get("/readyz", headers={"X-Request-ID": request_id})

    assert response.status_code == 503
    assert response.json() == {
        "error": {
            "code": "model_load_failed",
            "message": "Qwen model could not be loaded.",
            "request_id": request_id,
        }
    }
    assert "private model detail" not in response.text
    assert ready.status_code == 503
    assert ready.json() == {
        "error": {
            "code": "model_unavailable",
            "message": "Qwen model is recovering from an error.",
            "request_id": request_id,
        }
    }


def test_readiness_recovers_after_error_cooldown(tmp_path: Path) -> None:
    adapter = FakeAdapter()
    adapter.load_error = RuntimeError("load failed")
    app = create_qwen_app(
        _settings(tmp_path, error_cooldown_seconds=0.01),
        adapter,
    )

    with TestClient(app) as client:
        failed = client.post("/internal/v1/speech", json=_speech_body())
        time.sleep(0.02)
        ready = client.get("/readyz")

    assert failed.status_code == 503
    assert ready.status_code == 200
    assert ready.json() == {"status": "ready", "state": "error"}


def test_unexpected_worker_error_is_sanitized_without_reraise_or_log_leak(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    marker = "PRIVATE_QWEN_EXCEPTION_MARKER"
    adapter = FakeAdapter()
    adapter.synthesis_error = RuntimeError(marker)
    app = create_qwen_app(_settings(tmp_path), adapter)

    with TestClient(app) as client:
        response = client.post(
            "/internal/v1/speech",
            headers={"X-Request-ID": "req_internal-error"},
            json=_speech_body(),
        )

    assert response.status_code == 500
    assert response.json() == {
        "error": {
            "code": "internal_error",
            "message": "Internal worker error.",
            "request_id": "req_internal-error",
        }
    }
    assert marker not in response.text
    assert marker not in caplog.text
    assert marker not in capsys.readouterr().out


def test_normalized_worker_method_error_preserves_allow_header(tmp_path: Path) -> None:
    app = create_qwen_app(_settings(tmp_path), FakeAdapter())

    with TestClient(app) as client:
        response = client.get("/internal/v1/speech")

    assert response.status_code == 405
    assert response.headers["Allow"] == "POST"
    assert response.json()["error"]["code"] == "method_not_allowed"


def test_ephemeral_clone_returns_wav_without_persisting_reference(tmp_path: Path) -> None:
    adapter = FakeAdapter()
    settings = _settings(tmp_path)
    app = create_qwen_app(settings, adapter)
    client_filename = "PRIVATE_CLIENT_FILENAME.wav"

    with TestClient(app) as client:
        registry_before = app.state.voice_registry.list_public()
        files_before = _registry_files(settings.voice_data_dir)
        response = client.post(
            "/internal/v1/speech/clone",
            data=_clone_form(),
            files={"audio": (client_filename, _wav_bytes(), "audio/wav")},
        )
        registry_after = app.state.voice_registry.list_public()
        files_after = _registry_files(settings.voice_data_dir)

    assert response.status_code == 200
    assert response.content == adapter.audio
    assert response.headers["content-type"] == "audio/wav"
    assert response.headers["x-tts-model"] == ModelID.QWEN_CLONE
    assert response.headers["x-input-chars"] == str(len("새로 합성할 문장"))
    assert adapter.synthesis_requests == [
        InternalSpeechRequest(
            model=ModelID.QWEN_CLONE,
            text="새로 합성할 문장",
            language="Auto",
            voice=None,
            instructions=None,
        )
    ]
    assert len(adapter.build_calls) == 1
    temporary_path, transcript = adapter.build_calls[0]
    assert transcript == "exact text"
    assert adapter.paths_existing_during_build == [True]
    assert temporary_path.name == "reference.wav"
    assert temporary_path.parent.name.startswith("qwen-clone-")
    assert client_filename not in str(temporary_path)
    assert not temporary_path.exists()
    assert registry_before == registry_after == []
    assert files_before == files_after


@pytest.mark.parametrize(
    "transcript",
    [
        pytest.param(None, id="omitted"),
        pytest.param("", id="empty"),
        pytest.param("   ", id="whitespace"),
    ],
)
def test_ephemeral_clone_normalizes_blank_transcript(
    tmp_path: Path,
    transcript: str | None,
) -> None:
    adapter = FakeAdapter()
    app = create_qwen_app(_settings(tmp_path), adapter)
    form = _clone_form()
    if transcript is None:
        form.pop("transcript")
    else:
        form["transcript"] = transcript

    with TestClient(app) as client:
        response = client.post(
            "/internal/v1/speech/clone",
            data=form,
            files={"audio": ("reference.wav", _wav_bytes(), "audio/wav")},
        )

    assert response.status_code == 200
    assert len(adapter.build_calls) == 1
    temporary_path, normalized_transcript = adapter.build_calls[0]
    assert normalized_transcript is None
    assert not temporary_path.exists()


@pytest.mark.parametrize(
    ("form_updates", "audio", "expected_status", "expected_code"),
    [
        pytest.param(
            {"language": "ar"},
            _wav_bytes(),
            422,
            "unsupported_language",
            id="unsupported-language",
        ),
        pytest.param(
            {"consent_confirmed": "false"},
            _wav_bytes(),
            422,
            "consent_required",
            id="false-consent",
        ),
        pytest.param(
            {"response_format": "mp3"},
            _wav_bytes(),
            422,
            "validation_error",
            id="response-format",
        ),
        pytest.param({}, b"not audio", 400, "invalid_audio", id="invalid-audio"),
        pytest.param(
            {},
            b"x" * (20 * 1024 * 1024 + 1),
            413,
            "audio_too_large",
            id="oversized-audio",
        ),
    ],
)
def test_ephemeral_clone_rejects_invalid_requests_before_prompt_build(
    tmp_path: Path,
    form_updates: dict[str, str],
    audio: bytes,
    expected_status: int,
    expected_code: str,
) -> None:
    adapter = FakeAdapter()
    app = create_qwen_app(_settings(tmp_path), adapter)

    with TestClient(app) as client:
        response = client.post(
            "/internal/v1/speech/clone",
            data=_clone_form(**form_updates),
            files={"audio": ("private-name.wav", audio, "audio/wav")},
        )

    assert response.status_code == expected_status
    assert response.json()["error"]["code"] == expected_code
    assert adapter.build_calls == []


def test_ephemeral_clone_requires_consent_before_prompt_build(tmp_path: Path) -> None:
    adapter = FakeAdapter()
    app = create_qwen_app(_settings(tmp_path), adapter)
    form = _clone_form()
    form.pop("consent_confirmed")

    with TestClient(app) as client:
        response = client.post(
            "/internal/v1/speech/clone",
            data=form,
            files={"audio": ("private-name.wav", _wav_bytes(), "audio/wav")},
        )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"
    assert adapter.build_calls == []


@pytest.mark.parametrize(
    "field_name",
    ["voice", "instructions", "speed", "steps", "unknown_private_field"],
)
def test_ephemeral_clone_rejects_unknown_fields_before_prompt_build(
    tmp_path: Path,
    field_name: str,
) -> None:
    adapter = FakeAdapter()
    settings = _settings(tmp_path)
    app = create_qwen_app(settings, adapter)
    parts = _clone_multipart_parts()
    private_value = "PRIVATE_UNKNOWN_FIELD_VALUE"
    parts.append((field_name, (None, private_value)))

    with TestClient(app) as client:
        files_before = _registry_files(settings.voice_data_dir)
        response = client.post("/internal/v1/speech/clone", files=parts)
        files_after = _registry_files(settings.voice_data_dir)

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"
    assert private_value not in response.text
    assert adapter.build_calls == []
    assert files_before == files_after


@pytest.mark.parametrize(
    "field_name",
    [
        "input",
        "audio",
        "consent_confirmed",
        "transcript",
        "language",
        "response_format",
    ],
)
def test_ephemeral_clone_rejects_duplicate_singular_fields_before_prompt_build(
    tmp_path: Path,
    field_name: str,
) -> None:
    adapter = FakeAdapter()
    settings = _settings(tmp_path)
    app = create_qwen_app(settings, adapter)
    parts = _clone_multipart_parts()
    duplicate = next(part for name, part in parts if name == field_name)
    parts.append((field_name, duplicate))

    with TestClient(app) as client:
        files_before = _registry_files(settings.voice_data_dir)
        response = client.post("/internal/v1/speech/clone", files=parts)
        files_after = _registry_files(settings.voice_data_dir)

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"
    assert adapter.build_calls == []
    assert files_before == files_after


@pytest.mark.parametrize("failure_stage", ["build", "synthesis"])
def test_ephemeral_clone_removes_temporary_audio_after_adapter_failure(
    tmp_path: Path,
    failure_stage: str,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    marker = "PRIVATE_CLONE_EXCEPTION_MARKER"
    adapter = FakeAdapter()
    if failure_stage == "build":
        adapter.build_error = RuntimeError(marker)
    else:
        adapter.synthesis_error = RuntimeError(marker)
    app = create_qwen_app(_settings(tmp_path), adapter)
    private_values = (
        "PRIVATE_CLONE_INPUT",
        "PRIVATE_CLONE_TRANSCRIPT",
        "PRIVATE_CLIENT_FILENAME.wav",
        marker,
    )

    with TestClient(app) as client:
        response = client.post(
            "/internal/v1/speech/clone",
            data=_clone_form(
                input=private_values[0],
                transcript=private_values[1],
            ),
            files={"audio": (private_values[2], _wav_bytes(), "audio/wav")},
        )

    assert response.status_code == 500
    assert response.json()["error"]["code"] == "internal_error"
    assert len(adapter.build_calls) == 1
    assert not adapter.build_calls[0][0].exists()
    captured = capsys.readouterr()
    public_output = response.text + caplog.text + captured.out + captured.err
    assert all(value not in public_output for value in private_values)


async def test_ephemeral_clone_cancellation_waits_for_build_before_cleanup(
    tmp_path: Path,
) -> None:
    adapter = FakeAdapter()
    adapter.block_build = True
    app = create_qwen_app(_settings(tmp_path), adapter)
    transport = httpx.ASGITransport(app=app)

    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            request_task = asyncio.create_task(
                client.post(
                    "/internal/v1/speech/clone",
                    data=_clone_form(),
                    files={"audio": ("private.wav", _wav_bytes(), "audio/wav")},
                )
            )
            assert await asyncio.to_thread(adapter.build_started.wait, 1.0)
            temporary_path = adapter.build_calls[0][0]
            assert temporary_path.exists()

            request_task.cancel()
            await asyncio.sleep(0)
            assert not request_task.done()
            assert temporary_path.exists()

            adapter.release_build.set()
            with pytest.raises(asyncio.CancelledError):
                await request_task
            assert not temporary_path.exists()


async def test_ephemeral_clone_cancellation_waits_for_normalization_before_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    normalization_started = threading.Event()
    release_normalization = threading.Event()
    temporary_paths: list[Path] = []

    def blocking_normalizer(_audio: bytes, destination: Path) -> float:
        temporary_paths.append(destination)
        destination.write_bytes(b"normalization in progress")
        normalization_started.set()
        if not release_normalization.wait(timeout=2.0):
            raise AssertionError("test did not release blocked normalization")
        return 4.0

    monkeypatch.setattr(
        "tts_playground.qwen.app.normalize_reference_audio",
        blocking_normalizer,
    )
    app = create_qwen_app(_settings(tmp_path), FakeAdapter())
    transport = httpx.ASGITransport(app=app)

    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            request_task = asyncio.create_task(
                client.post(
                    "/internal/v1/speech/clone",
                    data=_clone_form(),
                    files={"audio": ("private.wav", _wav_bytes(), "audio/wav")},
                )
            )
            assert await asyncio.to_thread(normalization_started.wait, 1.0)
            temporary_path = temporary_paths[0]
            assert temporary_path.exists()

            request_task.cancel()
            await asyncio.sleep(0.05)
            cancellation_waited = not request_task.done()
            path_remained = temporary_path.exists()
            directory_remained = temporary_path.parent.exists()

            release_normalization.set()
            with pytest.raises(asyncio.CancelledError):
                await request_task

            assert cancellation_waited
            assert path_remained
            assert directory_remained
            assert not temporary_path.exists()
            assert not temporary_path.parent.exists()


async def test_ephemeral_clone_cancellation_wins_when_normalization_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    marker = "PRIVATE_CANCELLED_NORMALIZATION_FAILURE"
    normalization_started = threading.Event()
    release_normalization = threading.Event()
    temporary_paths: list[Path] = []

    def failing_normalizer(_audio: bytes, destination: Path) -> float:
        temporary_paths.append(destination)
        destination.write_bytes(b"normalization in progress")
        normalization_started.set()
        if not release_normalization.wait(timeout=2.0):
            raise AssertionError("test did not release blocked normalization")
        raise RuntimeError(marker)

    monkeypatch.setattr(
        "tts_playground.qwen.app.normalize_reference_audio",
        failing_normalizer,
    )
    app = create_qwen_app(_settings(tmp_path), FakeAdapter())
    transport = httpx.ASGITransport(app=app)

    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            request_task = asyncio.create_task(
                client.post(
                    "/internal/v1/speech/clone",
                    data=_clone_form(),
                    files={"audio": ("private.wav", _wav_bytes(), "audio/wav")},
                )
            )
            assert await asyncio.to_thread(normalization_started.wait, 1.0)
            temporary_path = temporary_paths[0]
            assert temporary_path.exists()

            request_task.cancel()
            await asyncio.sleep(0.05)
            assert not request_task.done()
            assert temporary_path.exists()
            assert temporary_path.parent.exists()

            release_normalization.set()
            with pytest.raises(asyncio.CancelledError):
                await request_task

            assert not temporary_path.exists()
            assert not temporary_path.parent.exists()
            metrics = await client.get("/metrics")
            await asyncio.sleep(0)
            assert (
                'tts_qwen_requests_total{model="qwen3-tts-1.7b-clone",status="500"}'
                not in metrics.text
            )
            assert 'tts_qwen_errors_total{code="internal_error"}' not in metrics.text
            assert "Task exception was never retrieved" not in caplog.text
            assert marker not in caplog.text


def test_ephemeral_clone_normalization_failure_without_cancellation_is_sanitized(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    marker = "PRIVATE_NORMALIZATION_FAILURE"
    temporary_paths: list[Path] = []

    def failing_normalizer(_audio: bytes, destination: Path) -> float:
        temporary_paths.append(destination)
        destination.write_bytes(b"normalization failed")
        raise RuntimeError(marker)

    monkeypatch.setattr(
        "tts_playground.qwen.app.normalize_reference_audio",
        failing_normalizer,
    )
    app = create_qwen_app(_settings(tmp_path), FakeAdapter())

    with TestClient(app) as client:
        response = client.post(
            "/internal/v1/speech/clone",
            data=_clone_form(),
            files={"audio": ("private.wav", _wav_bytes(), "audio/wav")},
        )

    assert response.status_code == 500
    assert response.json()["error"]["code"] == "internal_error"
    assert len(temporary_paths) == 1
    assert not temporary_paths[0].exists()
    assert not temporary_paths[0].parent.exists()
    captured = capsys.readouterr()
    public_output = response.text + caplog.text + captured.out + captured.err
    assert marker not in public_output


def test_ephemeral_clone_metrics_record_success_and_sanitized_failure(tmp_path: Path) -> None:
    adapter = FakeAdapter()
    app = create_qwen_app(_settings(tmp_path), adapter)

    with TestClient(app) as client:
        success = client.post(
            "/internal/v1/speech/clone",
            data=_clone_form(),
            files={"audio": ("reference.wav", _wav_bytes(), "audio/wav")},
        )
        adapter.synthesis_error = RuntimeError("PRIVATE_METRICS_MARKER")
        failure = client.post(
            "/internal/v1/speech/clone",
            data=_clone_form(),
            files={"audio": ("reference.wav", _wav_bytes(), "audio/wav")},
        )
        metrics = client.get("/metrics")

    assert success.status_code == 200
    assert failure.status_code == 500
    assert (
        'tts_qwen_requests_total{model="qwen3-tts-1.7b-clone",status="200"} 1.0'
        in metrics.text
    )
    assert (
        'tts_qwen_requests_total{model="qwen3-tts-1.7b-clone",status="500"} 1.0'
        in metrics.text
    )
    assert 'tts_qwen_errors_total{code="internal_error"} 1.0' in metrics.text


def test_voice_registration_list_detail_and_delete(tmp_path: Path) -> None:
    app = create_qwen_app(_settings(tmp_path), FakeAdapter())

    with TestClient(app) as client:
        registered = client.post(
            "/internal/v1/voices",
            data={
                "name": "동의한 목소리",
                "transcript": "정확한 녹취",
                "language": "ko",
                "consent_confirmed": "true",
            },
            files={"audio": ("reference.wav", _wav_bytes(), "audio/wav")},
        )
        voice_id = registered.json()["id"]
        listed = client.get("/internal/v1/voices")
        detail = client.get(f"/internal/v1/voices/{voice_id}")
        deleted = client.delete(f"/internal/v1/voices/{voice_id}")
        missing = client.get(f"/internal/v1/voices/{voice_id}")

    assert registered.status_code == 201
    assert voice_id.startswith("voice_")
    assert listed.json() == [registered.json()]
    assert detail.json() == registered.json()
    assert "transcript" not in detail.json()
    assert "audio_path" not in detail.json()
    assert deleted.status_code == 204
    assert deleted.content == b""
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "voice_not_found"


@pytest.mark.parametrize(
    ("transcript", "expected_transcript"),
    [
        pytest.param(None, None, id="omitted"),
        pytest.param("", None, id="empty"),
        pytest.param("   ", None, id="whitespace"),
        pytest.param("  exact transcript  ", "  exact transcript  ", id="nonblank"),
    ],
)
def test_voice_registration_accepts_optional_transcript(
    tmp_path: Path,
    transcript: str | None,
    expected_transcript: str | None,
) -> None:
    app = create_qwen_app(_settings(tmp_path), FakeAdapter())
    form_data = {
        "name": "optional transcript",
        "language": "en",
        "consent_confirmed": "true",
    }
    if transcript is not None:
        form_data["transcript"] = transcript

    with TestClient(app) as client:
        response = client.post(
            "/internal/v1/voices",
            data=form_data,
            files={"audio": ("reference.wav", _wav_bytes(), "audio/wav")},
        )
        assert response.status_code == 201
        voice_id = response.json()["id"]
        assert app.state.voice_registry.get_internal(voice_id).transcript == (
            expected_transcript
        )

    assert "transcript" not in response.json()
    assert "audio_path" not in response.json()


def test_voice_registration_rejects_more_than_20_mb_before_audio_parsing(
    tmp_path: Path,
) -> None:
    app = create_qwen_app(_settings(tmp_path), FakeAdapter())
    oversized = b"x" * (20 * 1024 * 1024 + 1)

    with TestClient(app) as client:
        response = client.post(
            "/internal/v1/voices",
            headers={"X-Request-ID": "req_upload-limit"},
            data={
                "name": "too large",
                "transcript": "transcript",
                "language": "en",
                "consent_confirmed": "true",
            },
            files={"audio": ("large.wav", oversized, "audio/wav")},
        )

    assert response.status_code == 413
    assert response.json() == {
        "error": {
            "code": "audio_too_large",
            "message": "Reference audio exceeds the 20 MB limit.",
            "request_id": "req_upload-limit",
        }
    }


def test_metrics_expose_qwen_families_and_observed_state(tmp_path: Path) -> None:
    adapter = FakeAdapter()
    app = create_qwen_app(_settings(tmp_path), adapter)

    with TestClient(app) as client:
        speech = client.post("/internal/v1/speech", json=_speech_body())
        metrics = client.get("/metrics")

    assert speech.status_code == 200
    assert metrics.status_code == 200
    assert 'tts_qwen_requests_total{model="qwen3-tts-1.7b-custom",status="200"} 1.0' in metrics.text
    assert 'tts_qwen_request_seconds_count{model="qwen3-tts-1.7b-custom"} 1.0' in metrics.text
    assert 'tts_qwen_audio_seconds_count{model="qwen3-tts-1.7b-custom"} 1.0' in metrics.text
    assert "tts_qwen_queue_depth 0.0" in metrics.text
    assert 'tts_qwen_state{state="ready"} 1.0' in metrics.text
    assert (
        'tts_qwen_loaded_model{model="qwen3-tts-1.7b-custom"} 1.0'
        in metrics.text
    )
    samples = {
        sample.name: sample.value
        for family in text_string_to_metric_families(metrics.text)
        for sample in family.samples
    }
    assert samples["tts_qwen_free_vram_bytes"] == adapter.free_vram


def test_idle_loop_unloads_resident_model(tmp_path: Path) -> None:
    adapter = FakeAdapter()
    app = create_qwen_app(
        _settings(tmp_path, idle_unload_seconds=0.02),
        adapter,
    )

    with TestClient(app) as client:
        response = client.post("/internal/v1/speech", json=_speech_body())
        assert response.status_code == 200
        assert adapter.device_memory_cleaned.wait(timeout=1.0)
        assert client.get("/internal/v1/status").json()["state"] == "unloaded"


def test_lifespan_shutdown_runs_when_idle_task_has_failed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = FakeAdapter()

    async def fail_idle_task(*_args: object) -> None:
        raise RuntimeError("idle task failed")

    monkeypatch.setattr(
        "tts_playground.qwen.app._idle_unload_loop",
        fail_idle_task,
    )
    app = create_qwen_app(_settings(tmp_path), adapter)

    with pytest.raises(RuntimeError, match="idle task failed"):
        with TestClient(app) as client:
            response = client.post("/internal/v1/speech", json=_speech_body())
            assert response.status_code == 200

    assert adapter.device_memory_cleaned.is_set()
