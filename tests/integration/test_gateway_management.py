import json
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest
import respx

from tts_playground.common.models import QWEN_CUSTOM_SPEAKERS
from tts_playground.common.settings import GatewaySettings
from tts_playground.gateway.app import create_gateway_app
from tts_playground.gateway.backends import BackendRouter


@pytest.fixture
async def app_client(tmp_path: Path) -> AsyncIterator[httpx.AsyncClient]:
    async with _gateway_client(tmp_path) as client:
        yield client


def _gateway_client(
    tmp_path: Path,
    *,
    raise_app_exceptions: bool = True,
    **setting_overrides: object,
) -> httpx.AsyncClient:
    key_file = tmp_path / "api-key"
    key_file.write_text("secret-key\n", encoding="utf-8")
    settings = GatewaySettings(api_key_file=key_file, **setting_overrides)
    backend_client = httpx.AsyncClient()
    router = BackendRouter(settings, backend_client)
    app = create_gateway_app(settings, router)
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(
            app=app,
            raise_app_exceptions=raise_app_exceptions,
        ),
        base_url="http://gateway",
    )

    original_aclose = client.aclose

    async def close_clients() -> None:
        await original_aclose()
        await backend_client.aclose()

    client.aclose = close_clients  # type: ignore[method-assign]
    return client


def _authorization(request_id: str = "req_management-test") -> dict[str, str]:
    return {
        "Authorization": "Bearer secret-key",
        "X-Request-ID": request_id,
    }


def _status(*, state: str = "ready", loaded_model: str | None = None) -> dict[str, object]:
    return {
        "state": state,
        "loaded_model": loaded_model,
        "queue_depth": 0,
        "last_error_code": None,
    }


def _public_voice() -> dict[str, object]:
    return {
        "id": "voice_0123456789abcdef0123456789abcdef",
        "name": "Consented voice",
        "language": "ko",
        "duration_seconds": 3.25,
        "consent_confirmed": True,
        "created_at": "2026-07-12T00:00:00+00:00",
    }


async def _chunks(*chunks: bytes) -> AsyncIterator[bytes]:
    for chunk in chunks:
        yield chunk


@respx.mock
async def test_models_expose_stable_ids_voices_capabilities_and_worker_state(
    app_client: httpx.AsyncClient,
) -> None:
    respx.get("http://supertonic-worker:7788/v1/styles").mock(
        return_value=httpx.Response(
            200,
            json={
                "styles": [
                    {"name": "M1", "kind": "builtin"},
                    {"name": "F1", "kind": "builtin"},
                ]
            },
        )
    )
    respx.get("http://qwen-worker:8001/internal/v1/status").mock(
        return_value=httpx.Response(
            200,
            json=_status(loaded_model="qwen3-tts-1.7b-custom"),
        )
    )

    response = await app_client.get("/v1/models", headers=_authorization("req_models-test"))

    assert response.status_code == 200
    assert response.headers["x-request-id"] == "req_models-test"
    payload = response.json()
    assert payload["object"] == "list"
    models = {model["id"]: model for model in payload["data"]}
    assert set(models) == {
        "supertonic-3",
        "qwen3-tts-1.7b-custom",
        "qwen3-tts-1.7b-design",
        "qwen3-tts-1.7b-clone",
    }
    assert models["supertonic-3"]["voices"] == ["M1", "F1"]
    assert models["supertonic-3"]["capabilities"] == {
        "instructions": False,
        "speed": True,
        "steps": True,
        "voice": True,
        "voice_cloning": False,
        "voice_design": False,
    }
    assert models["supertonic-3"]["worker_state"] == "ready"
    assert models["supertonic-3"]["loaded"] is True
    assert models["qwen3-tts-1.7b-custom"]["voices"] == list(QWEN_CUSTOM_SPEAKERS)
    assert models["qwen3-tts-1.7b-custom"]["worker_state"] == "ready"
    assert models["qwen3-tts-1.7b-custom"]["loaded"] is True
    assert models["qwen3-tts-1.7b-design"]["loaded"] is False
    assert models["qwen3-tts-1.7b-clone"]["loaded"] is False
    assert models["qwen3-tts-1.7b-design"]["capabilities"]["voice_design"] is True
    assert models["qwen3-tts-1.7b-clone"]["capabilities"] == {
        "instructions": False,
        "speed": False,
        "steps": False,
        "voice": True,
        "voice_cloning": True,
        "voice_design": False,
        "registered_reference": True,
        "ephemeral_reference": True,
        "reference_text_optional": True,
    }


@respx.mock
async def test_voice_registration_streams_multipart_fields_and_audio_unchanged(
    app_client: httpx.AsyncClient,
) -> None:
    captured: dict[str, object] = {}
    public_voice = _public_voice()

    async def capture_upload(request: httpx.Request) -> httpx.Response:
        captured["body"] = await request.aread()
        captured["content_type"] = request.headers["content-type"]
        captured["request_id"] = request.headers["x-request-id"]
        return httpx.Response(201, json=public_voice)

    route = respx.post("http://qwen-worker:8001/internal/v1/voices").mock(
        side_effect=capture_upload
    )
    audio = b"RIFF-reference-audio-WAVE"

    response = await app_client.post(
        "/v1/audio/voices",
        headers=_authorization("req_voice-upload"),
        data={
            "name": "Consented voice",
            "transcript": "exact transcript",
            "language": "ko",
            "consent_confirmed": "true",
        },
        files={"audio": ("reference.wav", audio, "audio/wav")},
    )

    assert response.status_code == 201
    assert response.json() == public_voice
    assert route.called
    body = captured["body"]
    assert isinstance(body, bytes)
    assert b'name="name"' in body and b"Consented voice" in body
    assert b'name="transcript"' in body and b"exact transcript" in body
    assert b'name="language"' in body and b"ko" in body
    assert b'name="consent_confirmed"' in body and b"true" in body
    assert b'name="audio"; filename="reference.wav"' in body
    assert b"Content-Type: audio/wav" in body
    assert audio in body
    assert str(captured["content_type"]).startswith("multipart/form-data; boundary=")
    assert captured["request_id"] == "req_voice-upload"


@respx.mock
async def test_voice_list_and_detail_only_return_public_metadata(
    app_client: httpx.AsyncClient,
) -> None:
    voice_id = "voice_0123456789abcdef0123456789abcdef"
    upstream_voice = {
        "id": voice_id,
        "name": "Private source voice",
        "language": "en",
        "duration_seconds": 4.0,
        "consent_confirmed": True,
        "created_at": "2026-07-12T00:00:00+00:00",
        "transcript": "must not escape",
        "audio_path": "/data/voices/private/reference.wav",
        "path": "/another/private/path",
    }
    respx.get("http://qwen-worker:8001/internal/v1/voices").mock(
        return_value=httpx.Response(200, json=[upstream_voice])
    )
    respx.get(f"http://qwen-worker:8001/internal/v1/voices/{voice_id}").mock(
        return_value=httpx.Response(200, json=upstream_voice)
    )

    listed = await app_client.get("/v1/audio/voices", headers=_authorization())
    detail = await app_client.get(
        f"/v1/audio/voices/{voice_id}",
        headers=_authorization(),
    )

    expected = {
        "id": voice_id,
        "name": "Private source voice",
        "language": "en",
        "duration_seconds": 4.0,
        "consent_confirmed": True,
        "created_at": "2026-07-12T00:00:00+00:00",
    }
    assert listed.status_code == 200
    assert listed.json() == [expected]
    assert detail.status_code == 200
    assert detail.json() == expected
    assert "transcript" not in listed.text
    assert "path" not in listed.text
    assert "transcript" not in detail.text
    assert "path" not in detail.text


@respx.mock
async def test_voice_delete_returns_empty_204(app_client: httpx.AsyncClient) -> None:
    voice_id = "voice_0123456789abcdef0123456789abcdef"
    route = respx.delete(
        f"http://qwen-worker:8001/internal/v1/voices/{voice_id}"
    ).mock(return_value=httpx.Response(204))

    response = await app_client.delete(
        f"/v1/audio/voices/{voice_id}",
        headers=_authorization("req_voice-delete"),
    )

    assert response.status_code == 204
    assert response.content == b""
    assert route.calls.last.request.headers["x-request-id"] == "req_voice-delete"


@respx.mock
async def test_voice_proxy_preserves_error_and_replaces_private_request_id(
    app_client: httpx.AsyncClient,
) -> None:
    voice_id = "voice_0123456789abcdef0123456789abcdef"
    respx.get(f"http://qwen-worker:8001/internal/v1/voices/{voice_id}").mock(
        return_value=httpx.Response(
            404,
            json={
                "error": {
                    "code": "voice_not_found",
                    "message": "Voice was not found.",
                    "request_id": "req_private-worker",
                }
            },
        )
    )

    response = await app_client.get(
        f"/v1/audio/voices/{voice_id}",
        headers=_authorization("req_public-error"),
    )

    assert response.status_code == 404
    assert response.json() == {
        "error": {
            "code": "voice_not_found",
            "message": "Backend request failed.",
            "request_id": "req_public-error",
        }
    }


@respx.mock
async def test_voice_registration_enforces_gateway_audio_limit(tmp_path: Path) -> None:
    async with _gateway_client(tmp_path, max_upload_bytes=4) as client:
        response = await client.post(
            "/v1/audio/voices",
            headers=_authorization("req_gateway-upload-limit"),
            data={
                "name": "oversized",
                "transcript": "transcript",
                "language": "en",
                "consent_confirmed": "true",
            },
            files={"audio": ("reference.wav", b"12345", "audio/wav")},
        )

    assert response.status_code == 413
    assert response.json() == {
        "error": {
            "code": "audio_too_large",
            "message": "Reference audio exceeds the 20 MB limit.",
            "request_id": "req_gateway-upload-limit",
        }
    }


@respx.mock
async def test_voice_registration_accepts_exact_audio_limit_with_maximum_fields(
    tmp_path: Path,
) -> None:
    route = respx.post("http://qwen-worker:8001/internal/v1/voices").mock(
        return_value=httpx.Response(201, json=_public_voice())
    )
    audio = b"x" * (20 * 1024 * 1024)

    async with _gateway_client(tmp_path) as client:
        response = await client.post(
            "/v1/audio/voices",
            headers=_authorization("req_exact-upload-limit"),
            data={
                "name": "😀" * 100,
                "transcript": "😀" * 4000,
                "language": "😀" * 10,
                "consent_confirmed": "true",
            },
            files={"audio": ("reference.wav", audio, "audio/wav")},
        )

    assert response.status_code == 201
    assert route.called


@respx.mock
async def test_malformed_chunked_multipart_uses_controlled_error_envelope(
    tmp_path: Path,
) -> None:
    request_id = "req_malformed-multipart"
    async with _gateway_client(tmp_path, raise_app_exceptions=False) as client:
        response = await client.post(
            "/v1/audio/voices",
            headers={
                **_authorization(request_id),
                "Content-Type": "multipart/form-data; boundary=boundary",
            },
            content=_chunks(b"not-a-multipart-body"),
        )

    assert response.status_code == 422
    assert response.headers["x-request-id"] == request_id
    assert response.json() == {
        "error": {
            "code": "invalid_multipart",
            "message": "Multipart request is invalid.",
            "request_id": request_id,
        }
    }


@respx.mock
async def test_unterminated_chunked_multipart_uses_controlled_error_envelope(
    tmp_path: Path,
) -> None:
    respx.post("http://qwen-worker:8001/internal/v1/voices").mock(
        return_value=httpx.Response(201, json=_public_voice())
    )
    request_id = "req_unterminated-multipart"
    body = (
        b"--boundary\r\n"
        b'Content-Disposition: form-data; name="audio"; filename="reference.wav"\r\n'
        b"Content-Type: audio/wav\r\n\r\n"
        b"RIFF-partial-WAVE"
    )
    async with _gateway_client(tmp_path, raise_app_exceptions=False) as client:
        response = await client.post(
            "/v1/audio/voices",
            headers={
                **_authorization(request_id),
                "Content-Type": "multipart/form-data; boundary=boundary",
            },
            content=_chunks(body[:31], body[31:]),
        )

    assert response.status_code == 422
    assert response.headers["x-request-id"] == request_id
    assert response.json()["error"] == {
        "code": "invalid_multipart",
        "message": "Multipart request is invalid.",
        "request_id": request_id,
    }


@respx.mock
async def test_oversized_multipart_header_uses_controlled_error_envelope(
    tmp_path: Path,
) -> None:
    request_id = "req_oversized-header"
    body = (
        b"--boundary\r\n"
        b"X-Oversized: "
        + b"x" * 5000
        + b"\r\n\r\ncontent\r\n--boundary--\r\n"
    )
    async with _gateway_client(tmp_path, raise_app_exceptions=False) as client:
        response = await client.post(
            "/v1/audio/voices",
            headers={
                **_authorization(request_id),
                "Content-Type": "multipart/form-data; boundary=boundary",
            },
            content=_chunks(body),
        )

    assert response.status_code == 422
    assert response.headers["x-request-id"] == request_id
    assert response.json()["error"]["code"] == "invalid_multipart"


@respx.mock
async def test_ready_requires_both_workers_reachable(app_client: httpx.AsyncClient) -> None:
    respx.get("http://supertonic-worker:7788/v1/styles").mock(
        return_value=httpx.Response(200, json={"styles": []})
    )
    respx.get("http://qwen-worker:8001/readyz").mock(
        return_value=httpx.Response(200, json={"status": "ready", "state": "unloaded"})
    )

    response = await app_client.get("/readyz", headers=_authorization("req_ready"))

    assert response.status_code == 200
    assert response.json() == {"status": "ready"}


@respx.mock
@pytest.mark.parametrize("unavailable", ["supertonic", "qwen"])
async def test_ready_is_503_when_a_worker_is_unavailable_or_in_cooldown(
    app_client: httpx.AsyncClient,
    unavailable: str,
) -> None:
    supertonic_status = 503 if unavailable == "supertonic" else 200
    qwen_status = 503 if unavailable == "qwen" else 200
    respx.get("http://supertonic-worker:7788/v1/styles").mock(
        return_value=httpx.Response(supertonic_status, json={"styles": []})
    )
    respx.get("http://qwen-worker:8001/readyz").mock(
        return_value=httpx.Response(
            qwen_status,
            json=(
                {
                    "error": {
                        "code": "model_unavailable",
                        "message": "Qwen model is recovering from an error.",
                        "request_id": "req_private-worker",
                    }
                }
                if qwen_status == 503
                else {"status": "ready", "state": "unloaded"}
            ),
        )
    )

    response = await app_client.get("/readyz", headers=_authorization("req_not-ready"))

    assert response.status_code == 503
    assert response.json()["error"]["request_id"] == "req_not-ready"


@respx.mock
@pytest.mark.parametrize("qwen_status", [500, 502, 503, 504])
async def test_ready_normalizes_every_non_ready_qwen_status(
    app_client: httpx.AsyncClient,
    qwen_status: int,
) -> None:
    respx.get("http://supertonic-worker:7788/v1/styles").mock(
        return_value=httpx.Response(200, json={"styles": []})
    )
    respx.get("http://qwen-worker:8001/readyz").mock(
        return_value=httpx.Response(
            qwen_status,
            json={
                "error": {
                    "code": "private_worker_error",
                    "message": "private readiness detail",
                    "request_id": "req_private-worker",
                }
            },
        )
    )

    response = await app_client.get(
        "/readyz",
        headers=_authorization("req_normalized-ready"),
    )

    assert response.status_code == 503
    assert response.json() == {
        "error": {
            "code": "backend_unavailable",
            "message": "Required backend is unavailable.",
            "request_id": "req_normalized-ready",
        }
    }


@respx.mock
@pytest.mark.parametrize(
    "transport_error",
    [httpx.ConnectError("offline"), httpx.ReadTimeout("slow")],
)
async def test_ready_normalizes_qwen_transport_failures(
    app_client: httpx.AsyncClient,
    transport_error: httpx.RequestError,
) -> None:
    respx.get("http://supertonic-worker:7788/v1/styles").mock(
        return_value=httpx.Response(200, json={"styles": []})
    )
    respx.get("http://qwen-worker:8001/readyz").mock(side_effect=transport_error)

    response = await app_client.get(
        "/readyz",
        headers=_authorization("req_transport-ready"),
    )

    assert response.status_code == 503
    assert response.json()["error"] == {
        "code": "backend_unavailable",
        "message": "Required backend is unavailable.",
        "request_id": "req_transport-ready",
    }


@respx.mock
async def test_metrics_aggregate_gateway_and_qwen_families(
    app_client: httpx.AsyncClient,
) -> None:
    respx.get("http://qwen-worker:8001/metrics").mock(
        return_value=httpx.Response(
            200,
            text="# HELP tts_qwen_queue_depth Queue depth.\n"
            "# TYPE tts_qwen_queue_depth gauge\n"
            "tts_qwen_queue_depth 0.0\n",
            headers={"content-type": "text/plain; version=0.0.4; charset=utf-8"},
        )
    )

    response = await app_client.get("/metrics", headers=_authorization())

    assert response.status_code == 200
    assert "tts_gateway_requests_total" in response.text
    assert "tts_qwen_queue_depth 0.0" in response.text


@respx.mock
@pytest.mark.parametrize(
    "upstream_response",
    [
        httpx.Response(503, text="private failure"),
        httpx.Response(200, text="private json", headers={"content-type": "application/json"}),
    ],
)
async def test_metrics_degrade_to_gateway_metrics_for_unusable_qwen_response(
    app_client: httpx.AsyncClient,
    upstream_response: httpx.Response,
) -> None:
    respx.get("http://qwen-worker:8001/metrics").mock(return_value=upstream_response)

    response = await app_client.get("/metrics", headers=_authorization())

    assert response.status_code == 200
    assert "tts_gateway_requests_total" in response.text
    assert "private" not in response.text
    assert "tts_qwen_" not in response.text


@respx.mock
@pytest.mark.parametrize(
    ("body", "forbidden"),
    [
        (b"malformed prometheus marker", b"malformed prometheus marker"),
        (
            b"# TYPE unrelated_metric gauge\nunrelated_metric 1\n",
            b"unrelated_metric",
        ),
        (
            b"# TYPE tts_gateway_collision gauge\ntts_gateway_collision 1\n",
            b"tts_gateway_collision",
        ),
        (
            b"# TYPE tts_qwen_valid gauge\ntts_qwen_valid 1\n"
            b"# TYPE unrelated_mixed gauge\nunrelated_mixed 1\n",
            b"tts_qwen_valid",
        ),
        (b"\xff\xfe", b"\xff\xfe"),
    ],
)
async def test_metrics_reject_invalid_or_unprefixed_qwen_exposition(
    app_client: httpx.AsyncClient,
    body: bytes,
    forbidden: bytes,
) -> None:
    respx.get("http://qwen-worker:8001/metrics").mock(
        return_value=httpx.Response(
            200,
            content=body,
            headers={"content-type": "text/plain; version=0.0.4; charset=utf-8"},
        )
    )

    response = await app_client.get("/metrics", headers=_authorization())

    assert response.status_code == 200
    assert b"tts_gateway_requests_total" in response.content
    assert forbidden not in response.content


@respx.mock
@pytest.mark.parametrize(
    ("field", "invalid_value", "private_marker"),
    [
        ("duration_seconds", "3.25", "3.25"),
        ("duration_seconds", float("nan"), "nan"),
        ("duration_seconds", 10**400, "10000000000000000000"),
        ("name", {"transcript": "nested private transcript"}, "nested private transcript"),
        ("created_at", {"path": "/private/voice/path"}, "/private/voice/path"),
    ],
)
async def test_voice_detail_rejects_invalid_or_nested_public_metadata(
    app_client: httpx.AsyncClient,
    field: str,
    invalid_value: object,
    private_marker: str,
) -> None:
    voice = _public_voice()
    voice[field] = invalid_value
    voice_id = str(voice["id"])
    respx.get(f"http://qwen-worker:8001/internal/v1/voices/{voice_id}").mock(
        return_value=httpx.Response(
            200,
            content=json.dumps(voice, allow_nan=True).encode(),
            headers={"content-type": "application/json"},
        )
    )

    response = await app_client.get(
        f"/v1/audio/voices/{voice_id}",
        headers=_authorization("req_invalid-voice-metadata"),
    )

    assert response.status_code == 503
    assert response.json() == {
        "error": {
            "code": "backend_invalid_response",
            "message": "Backend returned an invalid response.",
            "request_id": "req_invalid-voice-metadata",
        }
    }
    assert private_marker not in response.text


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("GET", "/v1/models"),
        ("POST", "/v1/audio/voices"),
        ("GET", "/v1/audio/voices"),
        ("GET", "/v1/audio/voices/voice_0123456789abcdef0123456789abcdef"),
        ("DELETE", "/v1/audio/voices/voice_0123456789abcdef0123456789abcdef"),
        ("GET", "/readyz"),
        ("GET", "/metrics"),
    ],
)
async def test_management_endpoints_require_authentication(
    app_client: httpx.AsyncClient,
    method: str,
    path: str,
) -> None:
    response = await app_client.request(
        method,
        path,
        headers={"X-Request-ID": "req_management-auth"},
    )

    assert response.status_code == 401
    assert response.json() == {
        "error": {
            "code": "unauthorized",
            "message": "Invalid or missing API key.",
            "request_id": "req_management-auth",
        }
    }
