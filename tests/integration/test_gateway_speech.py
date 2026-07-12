import json
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest
import respx

from tts_playground.common.settings import GatewaySettings
from tts_playground.gateway.app import create_gateway_app
from tts_playground.gateway.backends import BackendRouter

VALID_WAV = (
    b"RIFF\x24\x00\x00\x00WAVE"
    b"fmt \x10\x00\x00\x00\x01\x00\x01\x00\x40\x1f\x00\x00"
    b"\x80\x3e\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
)


@pytest.fixture
async def app_client(tmp_path: Path) -> AsyncIterator[httpx.AsyncClient]:
    key_file = tmp_path / "api-key"
    key_file.write_text("secret-key\n", encoding="utf-8")
    settings = GatewaySettings(api_key_file=key_file)
    async with httpx.AsyncClient() as backend_client:
        router = BackendRouter(settings, backend_client)
        app = create_gateway_app(settings, router)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://gateway") as client:
            yield client


def _authorization(request_id: str = "req_gateway-test") -> dict[str, str]:
    return {
        "Authorization": "Bearer secret-key",
        "X-Request-ID": request_id,
    }


@respx.mock
async def test_routes_supertonic_without_leaking_text_to_logs(
    app_client: httpx.AsyncClient,
    capsys: pytest.CaptureFixture[str],
) -> None:
    route = respx.post("http://supertonic-worker:7788/v1/tts").mock(
        return_value=httpx.Response(
            200,
            content=VALID_WAV,
            headers={
                "content-type": "audio/wav; codec=pcm",
                "x-audio-duration": "1.2",
                "x-queue-ms": "4",
                "x-processing-ms": "81",
            },
        )
    )

    response = await app_client.post(
        "/v1/audio/speech",
        headers=_authorization(),
        json={
            "model": "supertonic-3",
            "input": "민감한 원문",
            "voice": "F1",
            "language": "ko",
            "speed": 1.1,
            "steps": 8,
            "response_format": "wav",
        },
    )

    assert response.status_code == 200
    assert response.content == VALID_WAV
    assert response.headers["content-type"] == "audio/wav"
    assert response.headers["x-request-id"] == "req_gateway-test"
    assert response.headers["x-tts-model"] == "supertonic-3"
    assert response.headers["x-audio-duration"] == "1.2"
    assert response.headers["x-queue-ms"] == "4"
    assert response.headers["x-processing-ms"] == "81"
    assert route.calls.last.request.headers["x-request-id"] == "req_gateway-test"
    assert json.loads(route.calls.last.request.content) == {
        "text": "민감한 원문",
        "voice": "F1",
        "lang": "ko",
        "speed": 1.1,
        "steps": 8,
        "response_format": "wav",
    }
    assert "민감한 원문" not in capsys.readouterr().out


@respx.mock
async def test_routes_qwen_with_public_model_and_normalized_language(
    app_client: httpx.AsyncClient,
) -> None:
    route = respx.post("http://qwen-worker:8001/internal/v1/speech").mock(
        return_value=httpx.Response(
            200,
            content=VALID_WAV,
            headers={
                "content-type": "audio/wav",
                "x-audio-duration": "2.5",
                "x-queue-ms": "12",
                "x-processing-ms": "413",
            },
        )
    )

    response = await app_client.post(
        "/v1/audio/speech",
        headers=_authorization("req_qwen-route"),
        json={
            "model": "qwen3-tts-1.7b-custom",
            "input": "안녕하세요",
            "voice": "Sohee",
            "language": "ko",
            "instructions": "차분하게",
        },
    )

    assert response.status_code == 200
    assert response.content == VALID_WAV
    assert response.headers["x-tts-model"] == "qwen3-tts-1.7b-custom"
    assert route.calls.last.request.headers["x-request-id"] == "req_qwen-route"
    assert json.loads(route.calls.last.request.content) == {
        "model": "qwen3-tts-1.7b-custom",
        "text": "안녕하세요",
        "language": "Korean",
        "voice": "Sohee",
        "instructions": "차분하게",
    }


@respx.mock
async def test_preserves_backend_error_but_replaces_request_id(
    app_client: httpx.AsyncClient,
    capsys: pytest.CaptureFixture[str],
) -> None:
    marker = "PRIVATE_QWEN_UPSTREAM_MESSAGE_MARKER"
    respx.post("http://qwen-worker:8001/internal/v1/speech").mock(
        return_value=httpx.Response(
            503,
            json={
                "error": {
                    "code": "gpu_memory_unavailable",
                    "message": marker,
                    "request_id": "req_backend-private",
                }
            },
        )
    )

    response = await app_client.post(
        "/v1/audio/speech",
        headers=_authorization("req_gateway-error"),
        json={
            "model": "qwen3-tts-1.7b-custom",
            "input": "test",
            "voice": "Sohee",
            "language": "en",
        },
    )

    assert response.status_code == 503
    assert response.headers["x-request-id"] == "req_gateway-error"
    assert response.json() == {
        "error": {
            "code": "gpu_memory_unavailable",
            "message": "Backend request failed.",
            "request_id": "req_gateway-error",
        }
    }
    assert marker not in response.text
    assert marker not in capsys.readouterr().out

    metrics = await app_client.get("/metrics", headers=_authorization())
    assert (
        'tts_gateway_backend_errors_total{backend="qwen",code="gpu_memory_unavailable"} 1.0'
        in metrics.text
    )


@respx.mock
async def test_does_not_expose_non_json_backend_error_body(
    app_client: httpx.AsyncClient,
) -> None:
    respx.post("http://supertonic-worker:7788/v1/tts").mock(
        return_value=httpx.Response(503, text="<html>private worker details</html>")
    )

    response = await app_client.post(
        "/v1/audio/speech",
        headers=_authorization("req_html-error"),
        json={
            "model": "supertonic-3",
            "input": "test",
            "voice": "F1",
            "language": "en",
        },
    )

    assert response.status_code == 503
    assert response.json() == {
        "error": {
            "code": "backend_error",
            "message": "Backend request failed.",
            "request_id": "req_html-error",
        }
    }
    assert "private worker details" not in response.text


@respx.mock
@pytest.mark.parametrize(
    ("content_type", "content"),
    [
        ("text/html", b"<html>sensitive success marker</html>"),
        ("audio/mpeg", b"ID3 sensitive success marker"),
        ("audio/wav", b"NOPE....WAVE sensitive success marker"),
        ("audio/wav", b"RIFFxxxxNOPE sensitive success marker"),
    ],
)
async def test_rejects_non_wav_success_responses(
    app_client: httpx.AsyncClient,
    capsys: pytest.CaptureFixture[str],
    content_type: str,
    content: bytes,
) -> None:
    respx.post("http://supertonic-worker:7788/v1/tts").mock(
        return_value=httpx.Response(
            200,
            content=content,
            headers={"content-type": content_type},
        )
    )

    response = await app_client.post(
        "/v1/audio/speech",
        headers=_authorization("req_invalid-audio"),
        json={
            "model": "supertonic-3",
            "input": "test",
            "voice": "F1",
            "language": "en",
        },
    )

    assert response.status_code == 503
    assert response.json() == {
        "error": {
            "code": "backend_invalid_response",
            "message": "Backend returned an invalid audio response.",
            "request_id": "req_invalid-audio",
        }
    }
    assert "sensitive success marker" not in response.text
    assert "sensitive success marker" not in capsys.readouterr().out


@respx.mock
async def test_invalid_upstream_error_code_uses_safe_fallback_in_telemetry(
    app_client: httpx.AsyncClient,
    capsys: pytest.CaptureFixture[str],
) -> None:
    marker = "SENSITIVE Error Code Marker!"
    respx.post("http://qwen-worker:8001/internal/v1/speech").mock(
        return_value=httpx.Response(
            503,
            json={
                "error": {
                    "code": marker,
                    "message": "Backend request failed safely.",
                    "request_id": "req_backend-private",
                }
            },
        )
    )

    response = await app_client.post(
        "/v1/audio/speech",
        headers=_authorization("req_invalid-code"),
        json={
            "model": "qwen3-tts-1.7b-custom",
            "input": "test",
            "voice": "Sohee",
            "language": "en",
        },
    )
    metrics = await app_client.get("/metrics", headers=_authorization())

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "backend_error"
    assert marker not in metrics.text
    assert marker not in capsys.readouterr().out
    assert 'code="backend_error"' in metrics.text


@respx.mock
async def test_invalid_timing_headers_are_absent_from_response_logs_and_metrics(
    app_client: httpx.AsyncClient,
    capsys: pytest.CaptureFixture[str],
) -> None:
    queue_marker = "sensitive queue marker"
    respx.post("http://supertonic-worker:7788/v1/tts").mock(
        return_value=httpx.Response(
            200,
            content=VALID_WAV,
            headers={
                "content-type": "audio/wav",
                "x-audio-duration": "NaN",
                "x-queue-ms": queue_marker,
                "x-processing-ms": "999999999999999",
            },
        )
    )

    response = await app_client.post(
        "/v1/audio/speech",
        headers=_authorization("req_invalid-timing"),
        json={
            "model": "supertonic-3",
            "input": "test",
            "voice": "F1",
            "language": "en",
        },
    )
    metrics = await app_client.get("/metrics", headers=_authorization())

    assert response.status_code == 200
    assert "x-audio-duration" not in response.headers
    assert "x-queue-ms" not in response.headers
    assert "x-processing-ms" not in response.headers
    assert 'tts_gateway_audio_seconds_count{model="supertonic-3"}' not in metrics.text
    assert queue_marker not in capsys.readouterr().out


@respx.mock
@pytest.mark.parametrize("duration", ["NaN", "Inf"])
async def test_nonfinite_audio_duration_does_not_poison_metrics(
    app_client: httpx.AsyncClient,
    duration: str,
) -> None:
    respx.post("http://supertonic-worker:7788/v1/tts").mock(
        return_value=httpx.Response(
            200,
            content=VALID_WAV,
            headers={
                "content-type": "audio/wav",
                "x-audio-duration": duration,
            },
        )
    )

    response = await app_client.post(
        "/v1/audio/speech",
        headers=_authorization("req_nonfinite-duration"),
        json={
            "model": "supertonic-3",
            "input": "test",
            "voice": "F1",
            "language": "en",
        },
    )
    metrics = await app_client.get("/metrics", headers=_authorization())

    assert response.status_code == 200
    assert "x-audio-duration" not in response.headers
    assert 'tts_gateway_audio_seconds_count{model="supertonic-3"}' not in metrics.text
    assert f'tts_gateway_audio_seconds_sum{{model="supertonic-3"}} {duration}' not in metrics.text


@respx.mock
@pytest.mark.parametrize(
    ("exception", "expected_status", "expected_code"),
    [
        (httpx.ConnectError("offline"), 503, "backend_unavailable"),
        (httpx.ReadTimeout("slow"), 504, "backend_timeout"),
    ],
)
async def test_maps_backend_transport_errors(
    app_client: httpx.AsyncClient,
    exception: httpx.RequestError,
    expected_status: int,
    expected_code: str,
) -> None:
    respx.post("http://supertonic-worker:7788/v1/tts").mock(side_effect=exception)

    response = await app_client.post(
        "/v1/audio/speech",
        headers=_authorization("req_transport-error"),
        json={
            "model": "supertonic-3",
            "input": "test",
            "voice": "F1",
            "language": "en",
        },
    )

    assert response.status_code == expected_status
    assert response.json()["error"]["code"] == expected_code
    assert "offline" not in response.text
    assert "slow" not in response.text


async def test_local_validation_failure_is_not_counted_as_backend_error(
    app_client: httpx.AsyncClient,
) -> None:
    response = await app_client.post(
        "/v1/audio/speech",
        headers=_authorization("req_local-language-error"),
        json={
            "model": "supertonic-3",
            "input": "test",
            "voice": "F1",
            "language": "zz",
        },
    )
    metrics = await app_client.get("/metrics", headers=_authorization())

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "unsupported_language"
    assert "tts_gateway_backend_errors_total{" not in metrics.text


@respx.mock
async def test_exposes_gateway_metrics_from_an_app_local_registry(
    app_client: httpx.AsyncClient,
) -> None:
    respx.post("http://supertonic-worker:7788/v1/tts").mock(
        return_value=httpx.Response(
            200,
            content=VALID_WAV,
            headers={
                "content-type": "audio/wav",
                "x-audio-duration": "1.2",
            },
        )
    )
    await app_client.post(
        "/v1/audio/speech",
        headers=_authorization(),
        json={
            "model": "supertonic-3",
            "input": "test",
            "voice": "F1",
            "language": "en",
        },
    )

    response = await app_client.get("/metrics", headers=_authorization())

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert 'tts_gateway_requests_total{model="supertonic-3",status="200"} 1.0' in response.text
    assert "tts_gateway_request_seconds_count" in response.text
    assert "tts_gateway_audio_seconds_count" in response.text


async def test_metrics_requires_authentication(app_client: httpx.AsyncClient) -> None:
    response = await app_client.get("/metrics", headers={"X-Request-ID": "req_metrics-auth"})

    assert response.status_code == 401
    assert response.json() == {
        "error": {
            "code": "unauthorized",
            "message": "Invalid or missing API key.",
            "request_id": "req_metrics-auth",
        }
    }
