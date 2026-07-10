import json
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest
import respx

from tts_playground.common.settings import GatewaySettings
from tts_playground.gateway.app import create_gateway_app
from tts_playground.gateway.backends import BackendRouter


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
            content=b"RIFFaudio",
            headers={
                "content-type": "audio/wav",
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
    assert response.content == b"RIFFaudio"
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
            content=b"RIFFqwen",
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
    assert response.content == b"RIFFqwen"
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
) -> None:
    respx.post("http://qwen-worker:8001/internal/v1/speech").mock(
        return_value=httpx.Response(
            503,
            json={
                "error": {
                    "code": "gpu_memory_unavailable",
                    "message": "Not enough free GPU memory.",
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
            "message": "Not enough free GPU memory.",
            "request_id": "req_gateway-error",
        }
    }

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


@respx.mock
async def test_exposes_gateway_metrics_from_an_app_local_registry(
    app_client: httpx.AsyncClient,
) -> None:
    respx.post("http://supertonic-worker:7788/v1/tts").mock(
        return_value=httpx.Response(
            200,
            content=b"RIFFaudio",
            headers={"x-audio-duration": "1.2"},
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
