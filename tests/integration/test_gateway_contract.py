import re
import warnings
from pathlib import Path
from typing import Any

import pytest
from prometheus_client import generate_latest
from starlette.exceptions import StarletteDeprecationWarning

with warnings.catch_warnings():
    warnings.simplefilter("ignore", StarletteDeprecationWarning)
    from starlette.testclient import TestClient

from tts_playground.common.errors import ServiceError
from tts_playground.common.settings import GatewaySettings
from tts_playground.common.validation import ValidationIssue
from tts_playground.gateway.app import create_gateway_app
from tts_playground.gateway.backends import AudioResult


def _create_client(
    key_file: Path,
    backend_router: object | None = None,
    *,
    raise_server_exceptions: bool = True,
) -> TestClient:
    app = create_gateway_app(
        GatewaySettings(api_key_file=key_file),
        backend_router=backend_router,
    )
    return TestClient(app, raise_server_exceptions=raise_server_exceptions)


def _speech_body() -> dict[str, object]:
    return {
        "model": "supertonic-3",
        "input": "test",
        "voice": "F1",
        "language": "en",
    }


class _SynthesizeOnlyRouter:
    async def synthesize(self, _request: object, _request_id: str) -> AudioResult:
        return AudioResult(
            content=b"RIFF\x00\x00\x00\x00WAVE",
            content_type="audio/wav",
            duration_seconds="0",
            queue_ms=None,
            processing_ms=None,
        )


class _UnexpectedRouter:
    def __init__(self, marker: str) -> None:
        self.marker = marker

    async def synthesize(self, _request: object, _request_id: str) -> AudioResult:
        raise RuntimeError(self.marker)

    async def synthesize_ephemeral(self, **_kwargs: object) -> AudioResult:
        raise RuntimeError(self.marker)


class _FatalRouter:
    async def synthesize(self, _request: object, _request_id: str) -> AudioResult:
        raise BaseException("fatal marker")


class _ValidationDetailRouter:
    async def synthesize(self, _request: object, _request_id: str) -> AudioResult:
        raise ServiceError(
            status_code=422,
            code="validation_error",
            message="Request validation failed.",
            details=(ValidationIssue("voice", "invalid"),),
        )


class _NonValidationDetailRouter:
    async def synthesize(self, _request: object, _request_id: str) -> AudioResult:
        raise ServiceError(
            status_code=422,
            code="clone_rejected",
            message="Backend request failed.",
            details=(ValidationIssue("voice", "invalid"),),
        )


def test_health_is_public_and_speech_requires_auth(tmp_path: Path) -> None:
    key = tmp_path / "key"
    key.write_text("secret-key\n")
    client = _create_client(key)
    request_id = "req_unauthorized"

    assert client.get("/healthz").status_code == 200
    response = client.post(
        "/v1/audio/speech",
        headers={"X-Request-ID": request_id},
        json=_speech_body(),
    )

    assert response.status_code == 401
    assert response.headers["X-Request-ID"] == request_id
    assert response.json() == {
        "error": {
            "code": "unauthorized",
            "message": "Invalid or missing API key.",
            "request_id": request_id,
        }
    }


@pytest.mark.parametrize(
    "authorization",
    ["Basic secret-key", "Bearer", "Bearer wrong-key", "Bearer secret-key extra"],
)
def test_speech_rejects_invalid_authorization(
    tmp_path: Path,
    authorization: str,
) -> None:
    key = tmp_path / "key"
    key.write_text("secret-key\n")
    client = _create_client(key)

    response = client.post(
        "/v1/audio/speech",
        headers={"Authorization": authorization},
        json=_speech_body(),
    )

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "unauthorized"


@pytest.mark.asyncio
@pytest.mark.parametrize("authorization", [None, "Bearer wrong-key"])
async def test_clone_authentication_rejects_before_reading_multipart_body(
    tmp_path: Path,
    authorization: str | None,
) -> None:
    key = tmp_path / "key"
    key.write_text("secret-key\n")
    app = create_gateway_app(
        GatewaySettings(api_key_file=key, max_synthesis_body_bytes=1),
        backend_router=None,
    )
    receive_calls = 0
    sent: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        nonlocal receive_calls
        receive_calls += 1
        return {
            "type": "http.request",
            "body": b"PRIVATE_MULTIPART_MARKER",
            "more_body": False,
        }

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    headers = [
        (b"content-type", b"multipart/form-data; boundary=boundary"),
        (b"content-length", b"99999999"),
        (b"x-request-id", b"req_clone-auth-before-read"),
    ]
    if authorization is not None:
        headers.append((b"authorization", authorization.encode("ascii")))

    await app(
        {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/v1/audio/speech/clone",
            "raw_path": b"/v1/audio/speech/clone",
            "query_string": b"",
            "root_path": "",
            "headers": headers,
            "client": ("127.0.0.1", 1234),
            "server": ("gateway", 80),
            "state": {},
        },
        receive,
        send,
    )

    assert receive_calls == 0
    assert sent[0]["status"] == 401
    assert b'"code":"unauthorized"' in sent[1]["body"]


def test_authenticated_speech_reports_unconfigured_backend(tmp_path: Path) -> None:
    key = tmp_path / "key"
    key.write_text("secret-key\n")
    client = _create_client(key)
    request_id = "req_0123456789abcdef"

    response = client.post(
        "/v1/audio/speech",
        headers={
            "Authorization": "Bearer secret-key",
            "X-Request-ID": request_id,
        },
        json=_speech_body(),
    )

    assert response.status_code == 503
    assert response.headers["X-Request-ID"] == request_id
    assert response.json() == {
        "error": {
            "code": "backend_not_configured",
            "message": "Backend router is not configured.",
            "request_id": request_id,
        }
    }


def test_synthesize_only_router_keeps_json_route_compatible_and_clone_unconfigured(
    tmp_path: Path,
) -> None:
    key = tmp_path / "key"
    key.write_text("secret-key\n")
    client = _create_client(key, _SynthesizeOnlyRouter())
    headers = {
        "Authorization": "Bearer secret-key",
        "X-Request-ID": "req_synthesize-only",
    }

    speech = client.post("/v1/audio/speech", headers=headers, json=_speech_body())
    clone = client.post(
        "/v1/audio/speech/clone",
        headers=headers,
        data={"input": "test", "consent_confirmed": "true"},
        files={"audio": ("reference.wav", b"audio", "audio/wav")},
    )

    assert speech.status_code == 200
    assert speech.headers["content-type"] == "audio/wav"
    assert clone.status_code == 503
    assert clone.json()["error"]["code"] == "backend_not_configured"


@pytest.mark.parametrize("route", ["json", "ephemeral"])
def test_unexpected_synthesis_errors_are_sanitized_recorded_and_not_backend_failures(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    route: str,
) -> None:
    marker = f"PRIVATE_GATEWAY_{route.upper()}_EXCEPTION"
    key = tmp_path / "key"
    key.write_text("secret-key\n")
    app = create_gateway_app(
        GatewaySettings(api_key_file=key),
        backend_router=_UnexpectedRouter(marker),
    )
    client = TestClient(app, raise_server_exceptions=False)
    headers = {
        "Authorization": "Bearer secret-key",
        "X-Request-ID": f"req_unexpected-{route}",
    }

    if route == "json":
        response = client.post("/v1/audio/speech", headers=headers, json=_speech_body())
        model = "supertonic-3"
    else:
        response = client.post(
            "/v1/audio/speech/clone",
            headers=headers,
            data={"input": marker, "consent_confirmed": "true"},
            files={"audio": (f"{marker}.wav", marker.encode(), "audio/wav")},
        )
        model = "qwen3-tts-1.7b-clone"

    metrics = generate_latest(app.state.gateway_metrics.registry).decode()
    captured = capsys.readouterr()
    public_output = response.text + captured.out + captured.err

    assert response.status_code == 500
    assert response.json()["error"] == {
        "code": "internal_error",
        "message": "Internal gateway error.",
        "request_id": f"req_unexpected-{route}",
    }
    assert f'tts_gateway_requests_total{{model="{model}",status="500"}} 1.0' in metrics
    assert "tts_gateway_backend_errors_total{" not in metrics
    assert '"status": 500' in captured.out
    assert '"error_code": "internal_error"' in captured.out
    assert marker not in public_output


def test_synthesis_does_not_catch_base_exception(tmp_path: Path) -> None:
    key = tmp_path / "key"
    key.write_text("secret-key\n")
    client = _create_client(key, _FatalRouter())

    with pytest.raises(BaseException) as raised:
        client.post(
            "/v1/audio/speech",
            headers={"Authorization": "Bearer secret-key"},
            json=_speech_body(),
        )
    assert "fatal marker" in repr(raised.value)


def test_non_null_backend_is_controlled_until_routing_is_implemented(tmp_path: Path) -> None:
    key = tmp_path / "key"
    key.write_text("secret-key\n")
    client = _create_client(key, object(), raise_server_exceptions=False)
    request_id = "req_router-pending"

    response = client.post(
        "/v1/audio/speech",
        headers={
            "Authorization": "Bearer secret-key",
            "X-Request-ID": request_id,
        },
        json=_speech_body(),
    )

    assert response.status_code == 503
    assert response.headers["X-Request-ID"] == request_id
    assert response.json() == {
        "error": {
            "code": "backend_not_configured",
            "message": "Backend router is not configured.",
            "request_id": request_id,
        }
    }


def test_unknown_path_uses_public_error_envelope(tmp_path: Path) -> None:
    key = tmp_path / "key"
    key.write_text("secret-key\n")
    client = _create_client(key)
    request_id = "req_unknown-path"

    response = client.get("/missing", headers={"X-Request-ID": request_id})

    assert response.status_code == 404
    assert response.headers["X-Request-ID"] == request_id
    assert response.json() == {
        "error": {
            "code": "not_found",
            "message": "Resource not found.",
            "request_id": request_id,
        }
    }


def test_wrong_method_uses_public_error_envelope(tmp_path: Path) -> None:
    key = tmp_path / "key"
    key.write_text("secret-key\n")
    client = _create_client(key)
    request_id = "req_wrong-method"

    response = client.get("/v1/audio/speech", headers={"X-Request-ID": request_id})

    assert response.status_code == 405
    assert response.headers["Allow"] == "POST"
    assert response.headers["X-Request-ID"] == request_id
    assert response.json() == {
        "error": {
            "code": "method_not_allowed",
            "message": "Method not allowed.",
            "request_id": request_id,
        }
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("authenticated", [False, True])
async def test_oversized_speech_content_length_is_rejected_before_body_read(
    tmp_path: Path,
    authenticated: bool,
) -> None:
    key = tmp_path / "key"
    key.write_text("secret-key\n")
    app = create_gateway_app(
        GatewaySettings(api_key_file=key, max_synthesis_body_bytes=64),
        backend_router=None,
    )
    receive_calls = 0
    sent: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        nonlocal receive_calls
        receive_calls += 1
        return {"type": "http.request", "body": b"must not be read", "more_body": False}

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    headers = [
        (b"content-type", b"application/json"),
        (b"content-length", b"65"),
        (b"x-request-id", b"req_content-length-limit"),
    ]
    if authenticated:
        headers.append((b"authorization", b"Bearer secret-key"))

    await app(
        {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/v1/audio/speech",
            "raw_path": b"/v1/audio/speech",
            "query_string": b"",
            "root_path": "",
            "headers": headers,
            "client": ("127.0.0.1", 1234),
            "server": ("gateway", 80),
            "state": {},
        },
        receive,
        send,
    )

    assert receive_calls == 0
    assert sent[0]["status"] == 413
    response_headers = dict(sent[0]["headers"])
    assert response_headers[b"x-request-id"] == b"req_content-length-limit"
    assert b'"code":"request_too_large"' in sent[1]["body"]
    assert b'"request_id":"req_content-length-limit"' in sent[1]["body"]


@pytest.mark.asyncio
@pytest.mark.parametrize("authenticated", [False, True])
async def test_oversized_chunked_speech_stops_reading_at_limit(
    tmp_path: Path,
    authenticated: bool,
) -> None:
    key = tmp_path / "key"
    key.write_text("secret-key\n")
    app = create_gateway_app(
        GatewaySettings(api_key_file=key, max_synthesis_body_bytes=64),
        backend_router=None,
    )
    chunks = [b"x" * 40, b"y" * 25, b"private marker beyond limit"]
    receive_calls = 0
    sent: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        nonlocal receive_calls
        chunk = chunks[receive_calls]
        receive_calls += 1
        return {"type": "http.request", "body": chunk, "more_body": True}

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    headers = [
        (b"content-type", b"application/json"),
        (b"x-request-id", b"req_chunked-limit"),
    ]
    if authenticated:
        headers.append((b"authorization", b"Bearer secret-key"))

    await app(
        {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/v1/audio/speech",
            "raw_path": b"/v1/audio/speech",
            "query_string": b"",
            "root_path": "",
            "headers": headers,
            "client": ("127.0.0.1", 1234),
            "server": ("gateway", 80),
            "state": {},
        },
        receive,
        send,
    )

    assert receive_calls == 2
    assert sent[0]["status"] == 413
    assert b'"code":"request_too_large"' in sent[1]["body"]
    assert b"private marker beyond limit" not in sent[1]["body"]


def test_validation_errors_use_public_envelope(tmp_path: Path) -> None:
    key = tmp_path / "key"
    key.write_text("secret-key\n")
    client = _create_client(key)
    request_id = "req_validation-id"

    response = client.post(
        "/v1/audio/speech",
        headers={
            "Authorization": "Bearer secret-key",
            "X-Request-ID": request_id,
        },
        json={**_speech_body(), "voice": "invalid"},
    )

    assert response.status_code == 422
    assert response.headers["X-Request-ID"] == request_id
    assert response.json() == {
        "error": {
            "code": "validation_error",
            "message": "Request validation failed.",
            "request_id": request_id,
        }
    }


def test_service_validation_details_use_public_envelope(tmp_path: Path) -> None:
    key = tmp_path / "key"
    key.write_text("secret-key\n")
    client = _create_client(key, _ValidationDetailRouter())
    request_id = "req_service-validation"

    response = client.post(
        "/v1/audio/speech",
        headers={
            "Authorization": "Bearer secret-key",
            "X-Request-ID": request_id,
        },
        json=_speech_body(),
    )

    assert response.status_code == 422
    assert response.json() == {
        "error": {
            "code": "validation_error",
            "message": "Request validation failed.",
            "request_id": request_id,
            "details": [{"field": "voice", "reason": "invalid"}],
        }
    }


def test_service_non_validation_details_are_omitted(tmp_path: Path) -> None:
    key = tmp_path / "key"
    key.write_text("secret-key\n")
    client = _create_client(key, _NonValidationDetailRouter())
    request_id = "req_service-non-validation"

    response = client.post(
        "/v1/audio/speech",
        headers={
            "Authorization": "Bearer secret-key",
            "X-Request-ID": request_id,
        },
        json=_speech_body(),
    )

    assert response.status_code == 422
    assert response.json() == {
        "error": {
            "code": "clone_rejected",
            "message": "Backend request failed.",
            "request_id": request_id,
        }
    }


def test_invalid_request_id_is_replaced_everywhere(tmp_path: Path) -> None:
    key = tmp_path / "key"
    key.write_text("secret-key\n")
    client = _create_client(key)

    response = client.post(
        "/v1/audio/speech",
        headers={
            "Authorization": "Bearer secret-key",
            "X-Request-ID": "not-a-request-id",
        },
        json=_speech_body(),
    )

    generated_request_id = response.headers["X-Request-ID"]
    assert re.fullmatch(r"req_[0-9a-f]{32}", generated_request_id)
    assert response.json()["error"]["request_id"] == generated_request_id


def test_api_key_is_read_and_stripped_once_at_app_creation(tmp_path: Path) -> None:
    key = tmp_path / "key"
    key.write_text("  secret-key\n")
    client = _create_client(key)
    key.write_text("changed-key")

    original_key_response = client.post(
        "/v1/audio/speech",
        headers={"Authorization": "Bearer secret-key"},
        json=_speech_body(),
    )
    changed_key_response = client.post(
        "/v1/audio/speech",
        headers={"Authorization": "Bearer changed-key"},
        json=_speech_body(),
    )

    assert original_key_response.status_code == 503
    assert changed_key_response.status_code == 401


def test_empty_api_key_file_is_rejected_at_app_creation(tmp_path: Path) -> None:
    key = tmp_path / "key"
    key.write_text(" \n")

    with pytest.raises(ValueError, match="API key file must not be empty"):
        create_gateway_app(GatewaySettings(api_key_file=key), backend_router=None)
