import json
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest
import respx

from tts_playground.common.settings import GatewaySettings, QwenSettings
from tts_playground.gateway.app import create_gateway_app
from tts_playground.gateway.backends import BackendRouter
from tts_playground.qwen.adapter import InternalSpeechRequest, SynthesisResult
from tts_playground.qwen.app import create_qwen_app

VALID_WAV = (
    b"RIFF\x24\x00\x00\x00WAVE"
    b"fmt \x10\x00\x00\x00\x01\x00\x01\x00\x40\x1f\x00\x00"
    b"\x80\x3e\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
)


class _UnusedQwenAdapter:
    def free_vram_bytes(self) -> int:
        raise AssertionError("field validation must precede GPU access")

    def load(self, repository: str, attention_implementation: str) -> object:
        raise AssertionError("field validation must precede model loading")

    def cleanup_device_memory(self) -> None:
        raise AssertionError("field validation must precede device cleanup")

    def build_clone_prompt(
        self,
        model: object,
        audio_path: Path,
        transcript: str | None,
    ) -> object:
        raise AssertionError("field validation must precede prompt building")

    def synthesize(
        self,
        model: object,
        request: InternalSpeechRequest,
        clone_prompt: object | None,
    ) -> SynthesisResult:
        raise AssertionError("field validation must precede synthesis")


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


def _authorization(request_id: str = "req_gateway-test") -> dict[str, str]:
    return {
        "Authorization": "Bearer secret-key",
        "X-Request-ID": request_id,
    }


async def _chunks(*chunks: bytes) -> AsyncIterator[bytes]:
    for chunk in chunks:
        yield chunk


def _clone_body(
    *,
    boundary: bytes = b"boundary",
    input_text: bytes = b"hello",
    audio: bytes = b"RIFF-reference-WAVE",
    filename: bytes = b"reference.wav",
    final_boundary: bool = True,
) -> bytes:
    body = (
        b"--"
        + boundary
        + b'\r\nContent-Disposition: form-data; name="input"\r\n\r\n'
        + input_text
        + b"\r\n--"
        + boundary
        + b'\r\nContent-Disposition: form-data; name="audio"; filename="'
        + filename
        + b'"\r\nContent-Type: audio/wav\r\n\r\n'
        + audio
        + b"\r\n--"
        + boundary
        + b'\r\nContent-Disposition: form-data; name="consent_confirmed"\r\n\r\ntrue'
    )
    if final_boundary:
        body += b"\r\n--" + boundary + b"--\r\n"
    return body


@respx.mock
async def test_routes_ephemeral_clone_streams_multipart_and_safe_headers(
    app_client: httpx.AsyncClient,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured: dict[str, object] = {}

    async def capture_upload(request: httpx.Request) -> httpx.Response:
        captured["body"] = await request.aread()
        captured["content_type"] = request.headers["content-type"]
        captured["content_length"] = request.headers["content-length"]
        captured["request_id"] = request.headers["x-request-id"]
        return httpx.Response(
            200,
            content=VALID_WAV,
            headers={
                "content-type": "audio/wav",
                "x-audio-duration": "1.5",
                "x-queue-ms": "3",
                "x-processing-ms": "40",
                "x-input-chars": "14",
            },
        )

    route = respx.post("http://qwen-worker:8001/internal/v1/speech/clone").mock(
        side_effect=capture_upload
    )
    input_marker = "PRIVATE_CLONE_INPUT"
    transcript_marker = "PRIVATE_CLONE_TRANSCRIPT"
    filename_marker = "PRIVATE_CLONE_FILENAME.wav"
    audio_marker = b"PRIVATE_CLONE_AUDIO"

    response = await app_client.post(
        "/v1/audio/speech/clone",
        headers=_authorization("req_ephemeral-clone"),
        data={
            "input": input_marker,
            "transcript": transcript_marker,
            "language": "en",
            "consent_confirmed": "true",
        },
        files={"audio": (filename_marker, audio_marker, "audio/wav")},
    )

    assert response.status_code == 200
    assert response.content == VALID_WAV
    assert response.headers["content-type"] == "audio/wav"
    assert response.headers["x-request-id"] == "req_ephemeral-clone"
    assert response.headers["x-tts-model"] == "qwen3-tts-1.7b-clone"
    assert response.headers["x-audio-duration"] == "1.5"
    assert response.headers["x-queue-ms"] == "3"
    assert response.headers["x-processing-ms"] == "40"
    assert "x-input-chars" not in response.headers
    assert route.call_count == 1
    assert captured["request_id"] == "req_ephemeral-clone"
    assert str(captured["content_type"]).startswith("multipart/form-data; boundary=")
    assert int(str(captured["content_length"])) == len(captured["body"])  # type: ignore[arg-type]
    body = captured["body"]
    assert isinstance(body, bytes)
    for marker in (
        input_marker.encode(),
        transcript_marker.encode(),
        filename_marker.encode(),
        audio_marker,
    ):
        assert marker in body

    metrics = await app_client.get("/metrics", headers=_authorization())
    logs = capsys.readouterr().out
    assert (
        'tts_gateway_requests_total{model="qwen3-tts-1.7b-clone",status="200"} 1.0'
        in metrics.text
    )
    assert '"input_chars": 14' in logs
    for marker in (input_marker, transcript_marker, filename_marker, audio_marker.decode()):
        assert marker not in logs


@respx.mock
async def test_ephemeral_clone_rejects_oversized_content_length_before_forwarding(
    tmp_path: Path,
) -> None:
    route = respx.post("http://qwen-worker:8001/internal/v1/speech/clone").mock(
        return_value=httpx.Response(200, content=VALID_WAV, headers={"content-type": "audio/wav"})
    )
    maximum_request_bytes = 20 * 1024 * 1024 + 1024 * 1024
    async with _gateway_client(tmp_path) as client:
        response = await client.post(
            "/v1/audio/speech/clone",
            headers={
                **_authorization("req_clone-content-length"),
                "Content-Type": "multipart/form-data; boundary=boundary",
                "Content-Length": str(maximum_request_bytes + 1),
            },
            content=_chunks(b"must-not-be-forwarded"),
        )

    assert response.status_code == 413
    assert response.json()["error"] == {
        "code": "request_too_large",
        "message": "Multipart request exceeds the size limit.",
        "request_id": "req_clone-content-length",
    }
    assert not route.called


@respx.mock
async def test_ephemeral_clone_rejects_chunked_audio_over_20_mb(tmp_path: Path) -> None:
    respx.post("http://qwen-worker:8001/internal/v1/speech/clone").mock(
        return_value=httpx.Response(200, content=VALID_WAV, headers={"content-type": "audio/wav"})
    )
    body = _clone_body(audio=b"x" * (20 * 1024 * 1024 + 1))
    async with _gateway_client(tmp_path, raise_app_exceptions=False) as client:
        response = await client.post(
            "/v1/audio/speech/clone",
            headers={
                **_authorization("req_clone-audio-limit"),
                "Content-Type": "multipart/form-data; boundary=boundary",
            },
            content=_chunks(body[:1024], body[1024:]),
        )

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "audio_too_large"


@respx.mock
async def test_ephemeral_clone_rejects_oversized_non_audio_multipart(tmp_path: Path) -> None:
    respx.post("http://qwen-worker:8001/internal/v1/speech/clone").mock(
        return_value=httpx.Response(200, content=VALID_WAV, headers={"content-type": "audio/wav"})
    )
    body = _clone_body(input_text=b"x" * (1024 * 1024 + 1), audio=b"1234")
    async with _gateway_client(
        tmp_path,
        max_upload_bytes=4,
        raise_app_exceptions=False,
    ) as client:
        response = await client.post(
            "/v1/audio/speech/clone",
            headers={
                **_authorization("req_clone-request-limit"),
                "Content-Type": "multipart/form-data; boundary=boundary",
            },
            content=_chunks(body),
        )

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "request_too_large"


@respx.mock
@pytest.mark.parametrize(
    ("content_type", "body"),
    [
        ("multipart/form-data", b"not-multipart"),
        (
            "multipart/form-data; boundary=boundary",
            b"--boundary\r\n" + b"X-Header: value\r\n" * 9 + b"\r\nx\r\n--boundary--\r\n",
        ),
        (
            "multipart/form-data; boundary=boundary",
            b"--boundary\r\nX-Large: " + b"x" * 5000 + b"\r\n\r\nx\r\n--boundary--\r\n",
        ),
        (
            "multipart/form-data; boundary=boundary",
            _clone_body(final_boundary=False),
        ),
    ],
)
async def test_ephemeral_clone_rejects_malformed_multipart(
    tmp_path: Path,
    content_type: str,
    body: bytes,
) -> None:
    route = respx.post("http://qwen-worker:8001/internal/v1/speech/clone").mock(
        return_value=httpx.Response(200, content=VALID_WAV, headers={"content-type": "audio/wav"})
    )
    async with _gateway_client(tmp_path, raise_app_exceptions=False) as client:
        response = await client.post(
            "/v1/audio/speech/clone",
            headers={
                **_authorization("req_clone-invalid-multipart"),
                "Content-Type": content_type,
            },
            content=_chunks(body),
        )

    assert response.status_code == 422
    assert response.json()["error"] == {
        "code": "invalid_multipart",
        "message": "Multipart request is invalid.",
        "request_id": "req_clone-invalid-multipart",
    }
    if content_type == "multipart/form-data":
        assert not route.called


@respx.mock
@pytest.mark.parametrize("upstream_status", [422, 503])
async def test_ephemeral_clone_sanitizes_upstream_errors_and_records_metrics(
    app_client: httpx.AsyncClient,
    capsys: pytest.CaptureFixture[str],
    upstream_status: int,
) -> None:
    private_message = "PRIVATE_UPSTREAM_CLONE_MESSAGE"
    private_request_id = "req_private-clone-worker"
    respx.post("http://qwen-worker:8001/internal/v1/speech/clone").mock(
        return_value=httpx.Response(
            upstream_status,
            json={
                "error": {
                    "code": "clone_rejected",
                    "message": private_message,
                    "request_id": private_request_id,
                }
            },
        )
    )

    response = await app_client.post(
        "/v1/audio/speech/clone",
        headers=_authorization("req_public-clone-error"),
        data={"input": "PRIVATE_FAILED_INPUT", "consent_confirmed": "true"},
        files={"audio": ("PRIVATE_FAILED.wav", b"PRIVATE_FAILED_AUDIO", "audio/wav")},
    )
    metrics = await app_client.get("/metrics", headers=_authorization())
    logs = capsys.readouterr().out

    assert response.status_code == upstream_status
    assert response.json()["error"] == {
        "code": "clone_rejected",
        "message": "Backend request failed.",
        "request_id": "req_public-clone-error",
    }
    assert private_message not in response.text
    assert private_request_id not in response.text
    assert (
        f'tts_gateway_requests_total{{model="qwen3-tts-1.7b-clone",status="{upstream_status}"}} 1.0'
        in metrics.text
    )
    assert (
        'tts_gateway_backend_errors_total{backend="qwen",code="clone_rejected"} 1.0'
        in metrics.text
    )
    for marker in (
        private_message,
        private_request_id,
        "PRIVATE_FAILED_INPUT",
        "PRIVATE_FAILED.wav",
        "PRIVATE_FAILED_AUDIO",
    ):
        assert marker not in logs


@respx.mock
@pytest.mark.parametrize(
    ("exception", "status", "code"),
    [
        (httpx.ConnectError("PRIVATE_CLONE_OFFLINE"), 503, "backend_unavailable"),
        (httpx.ReadTimeout("PRIVATE_CLONE_SLOW"), 504, "backend_timeout"),
    ],
)
async def test_ephemeral_clone_transport_errors_increment_qwen_metrics(
    app_client: httpx.AsyncClient,
    exception: httpx.RequestError,
    status: int,
    code: str,
) -> None:
    respx.post("http://qwen-worker:8001/internal/v1/speech/clone").mock(
        side_effect=exception
    )

    response = await app_client.post(
        "/v1/audio/speech/clone",
        headers=_authorization("req_clone-transport"),
        data={"input": "test", "consent_confirmed": "true"},
        files={"audio": ("reference.wav", b"audio", "audio/wav")},
    )
    metrics = await app_client.get("/metrics", headers=_authorization())

    assert response.status_code == status
    assert response.json()["error"]["code"] == code
    assert f'tts_gateway_backend_errors_total{{backend="qwen",code="{code}"}} 1.0' in metrics.text


@pytest.mark.parametrize(
    ("case", "field_name", "reason"),
    [
        pytest.param("missing", "input", "required", id="missing"),
        pytest.param("invalid", "language", "invalid", id="invalid"),
        pytest.param("forbidden", "voice", "unknown", id="forbidden-voice"),
        pytest.param(
            "forbidden",
            "instructions",
            "unknown",
            id="forbidden-instructions",
        ),
        pytest.param("forbidden", "speed", "unknown", id="forbidden-speed"),
        pytest.param("forbidden", "steps", "unknown", id="forbidden-steps"),
        pytest.param(
            "unknown",
            "unknown_private_field",
            "unknown",
            id="unknown",
        ),
        pytest.param("duplicate", "input", "duplicate", id="duplicate-input"),
        pytest.param("duplicate", "audio", "duplicate", id="duplicate-audio"),
        pytest.param(
            "duplicate",
            "consent_confirmed",
            "duplicate",
            id="duplicate-consent",
        ),
        pytest.param(
            "duplicate",
            "transcript",
            "duplicate",
            id="duplicate-transcript",
        ),
        pytest.param(
            "duplicate",
            "language",
            "duplicate",
            id="duplicate-language",
        ),
        pytest.param(
            "duplicate",
            "response_format",
            "duplicate",
            id="duplicate-response-format",
        ),
    ],
)
async def test_public_clone_forwards_real_worker_validation_details(
    tmp_path: Path,
    case: str,
    field_name: str,
    reason: str,
) -> None:
    key_file = tmp_path / "api-key"
    key_file.write_text("secret-key\n", encoding="utf-8")
    gateway_settings = GatewaySettings(api_key_file=key_file)
    qwen_settings = QwenSettings(
        voice_data_dir=tmp_path / "voices",
        min_free_vram_bytes=0,
    )
    qwen_app = create_qwen_app(qwen_settings, _UnusedQwenAdapter())
    parts: list[tuple[str, tuple[None, str] | tuple[str, bytes, str]]] = [
        ("input", (None, "test")),
        ("consent_confirmed", (None, "true")),
        ("transcript", (None, "exact text")),
        ("language", (None, "auto")),
        ("response_format", (None, "wav")),
        ("audio", ("reference.wav", b"PRIVATE_INVALID_AUDIO", "audio/wav")),
    ]
    private_value = "PRIVATE_WORKER_VALIDATION_VALUE"
    if case == "missing":
        parts = [part for part in parts if part[0] != field_name]
    elif case == "invalid":
        parts = [
            (name, (None, private_value) if name == field_name else part)
            for name, part in parts
        ]
    elif case == "duplicate":
        duplicate_part = next(part for name, part in parts if name == field_name)
        parts.append((field_name, duplicate_part))
    else:
        parts.append((field_name, (None, private_value)))

    async with qwen_app.router.lifespan_context(qwen_app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=qwen_app),
        ) as backend_client:
            router = BackendRouter(gateway_settings, backend_client)
            gateway_app = create_gateway_app(gateway_settings, router)
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=gateway_app),
                base_url="http://gateway",
            ) as client:
                response = await client.post(
                    "/v1/audio/speech/clone",
                    headers=_authorization("req_public-field-validation"),
                    files=parts,
                )

    assert response.status_code == 422
    assert response.json()["error"] == {
        "code": "validation_error",
        "message": "Request validation failed.",
        "request_id": "req_public-field-validation",
        "details": [{"field": field_name, "reason": reason}],
    }
    assert private_value not in response.text


@respx.mock
async def test_worker_validation_details_are_forwarded_with_public_request_id(
    app_client: httpx.AsyncClient,
    capsys: pytest.CaptureFixture[str],
) -> None:
    private_message = "PRIVATE_VALIDATION_MESSAGE"
    private_request_id = "req_private-validation-worker"
    private_value = "PRIVATE_VALIDATION_VALUE"
    private_context = "PRIVATE_VALIDATION_CONTEXT"
    safe_detail_marker = "private_safe_field"
    respx.post("http://qwen-worker:8001/internal/v1/speech/clone").mock(
        return_value=httpx.Response(
            422,
            json={
                "error": {
                    "code": "validation_error",
                    "message": private_message,
                    "request_id": private_request_id,
                    "value": private_value,
                    "context": {"marker": private_context},
                    "details": [
                        {"field": safe_detail_marker, "reason": "unknown"}
                    ],
                }
            },
        )
    )

    response = await app_client.post(
        "/v1/audio/speech/clone",
        headers=_authorization("req_public-validation"),
        data={"input": "test", "consent_confirmed": "true"},
        files={"audio": ("reference.wav", b"audio", "audio/wav")},
    )
    metrics = await app_client.get("/metrics", headers=_authorization())
    logs = capsys.readouterr().out

    assert response.status_code == 422
    assert response.json()["error"] == {
        "code": "validation_error",
        "message": "Request validation failed.",
        "request_id": "req_public-validation",
        "details": [{"field": safe_detail_marker, "reason": "unknown"}],
    }
    assert private_message not in response.text
    assert private_request_id not in response.text
    assert private_value not in response.text
    assert private_context not in response.text
    assert safe_detail_marker not in metrics.text
    assert safe_detail_marker not in logs


@respx.mock
async def test_worker_validation_details_are_discarded_for_other_error_codes(
    app_client: httpx.AsyncClient,
) -> None:
    respx.post("http://qwen-worker:8001/internal/v1/speech/clone").mock(
        return_value=httpx.Response(
            422,
            json={
                "error": {
                    "code": "clone_rejected",
                    "message": "PRIVATE_OTHER_ERROR_MESSAGE",
                    "request_id": "req_private-other-worker",
                    "details": [{"field": "audio", "reason": "required"}],
                }
            },
        )
    )

    response = await app_client.post(
        "/v1/audio/speech/clone",
        headers=_authorization("req_public-other-error"),
        data={"input": "test", "consent_confirmed": "true"},
        files={"audio": ("reference.wav", b"audio", "audio/wav")},
    )

    assert response.status_code == 422
    assert response.json()["error"] == {
        "code": "clone_rejected",
        "message": "Backend request failed.",
        "request_id": "req_public-other-error",
    }


@respx.mock
@pytest.mark.parametrize(
    "details",
    [
        pytest.param(
            [{"field": "audio", "reason": "required"}] * 17,
            id="seventeen-issues",
        ),
        pytest.param(
            [{"field": "audio", "reason": "PRIVATE_REASON"}],
            id="bad-reason",
        ),
        pytest.param(
            [{"field": "unsafe-field", "reason": "required"}],
            id="unsafe-field",
        ),
        pytest.param(
            [
                {
                    "field": "audio",
                    "reason": "required",
                    "value": "PRIVATE_DETAIL_VALUE",
                    "context": {"marker": "PRIVATE_DETAIL_CONTEXT"},
                }
            ],
            id="extra-keys",
        ),
        pytest.param({"field": "audio", "reason": "required"}, id="non-list"),
    ],
)
async def test_malicious_validation_details_are_discarded_all_or_nothing(
    app_client: httpx.AsyncClient,
    capsys: pytest.CaptureFixture[str],
    details: object,
) -> None:
    markers = (
        "PRIVATE_MALICIOUS_MESSAGE",
        "req_private-malicious-worker",
        "PRIVATE_DETAIL_VALUE",
        "PRIVATE_DETAIL_CONTEXT",
        "PRIVATE_REASON",
        "unsafe-field",
    )
    respx.post("http://qwen-worker:8001/internal/v1/speech/clone").mock(
        return_value=httpx.Response(
            422,
            json={
                "error": {
                    "code": "validation_error",
                    "message": markers[0],
                    "request_id": markers[1],
                    "details": details,
                }
            },
        )
    )

    response = await app_client.post(
        "/v1/audio/speech/clone",
        headers=_authorization("req_public-malicious"),
        data={"input": "PRIVATE_FAILED_INPUT", "consent_confirmed": "true"},
        files={"audio": ("PRIVATE_FAILED.wav", b"PRIVATE_FAILED_AUDIO", "audio/wav")},
    )
    metrics = await app_client.get("/metrics", headers=_authorization())
    logs = capsys.readouterr().out

    assert response.status_code == 422
    assert response.json()["error"] == {
        "code": "validation_error",
        "message": "Backend request failed.",
        "request_id": "req_public-malicious",
    }
    public_output = response.text + metrics.text + logs
    assert all(marker not in public_output for marker in markers)
    assert "PRIVATE_FAILED_INPUT" not in logs
    assert "PRIVATE_FAILED.wav" not in logs
    assert "PRIVATE_FAILED_AUDIO" not in logs


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
