from __future__ import annotations

import io
import re
import threading
import time
import warnings
from pathlib import Path

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
        self.synthesis_error: BaseException | None = None
        self.synthesis_requests: list[InternalSpeechRequest] = []
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
        transcript: str,
    ) -> object:
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


def test_unexpected_worker_error_uses_safe_internal_envelope(tmp_path: Path) -> None:
    adapter = FakeAdapter()
    adapter.synthesis_error = RuntimeError("private inference detail")
    app = create_qwen_app(_settings(tmp_path), adapter)

    with TestClient(app, raise_server_exceptions=False) as client:
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
    assert "private inference detail" not in response.text


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
