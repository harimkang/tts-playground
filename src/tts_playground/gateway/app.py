import json
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from time import perf_counter
from typing import Annotated, Protocol, cast, runtime_checkable

import httpx
from fastapi import Depends, FastAPI, Header, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException
from starlette.responses import Response
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from tts_playground.common.errors import ServiceError
from tts_playground.common.logging import (
    MAX_AUDIO_DURATION_SECONDS,
    configure_logging,
    log_gateway_request,
    normalize_bounded_nonnegative,
    normalize_error_code,
)
from tts_playground.common.models import ModelID
from tts_playground.common.request_id import resolve_request_id
from tts_playground.common.settings import GatewaySettings
from tts_playground.common.validation import (
    ValidationIssue,
    validation_issues_to_public,
)
from tts_playground.gateway.auth import read_api_key, require_api_key
from tts_playground.gateway.backends import (
    AudioResult,
    BackendRouter,
    DiscoveryResult,
    ProxyResult,
)
from tts_playground.gateway.metrics import GatewayMetrics
from tts_playground.gateway.routes import create_management_router
from tts_playground.gateway.schemas import SpeechRequest

RequestHandler = Callable[[Request], Awaitable[Response]]
Authenticator = Callable[[str | None], None]
AppLifespan = Callable[[FastAPI], AbstractAsyncContextManager[None]]


@runtime_checkable
class SpeechSynthesisRouter(Protocol):
    async def synthesize(self, request: SpeechRequest, request_id: str) -> AudioResult: ...


@runtime_checkable
class EphemeralSynthesisRouter(Protocol):
    async def synthesize_ephemeral(
        self,
        *,
        content: AsyncIterator[bytes],
        content_type: str | None,
        content_length: int | None,
        request_id: str,
    ) -> AudioResult: ...


def create_gateway_app(
    settings: GatewaySettings,
    backend_router: object | None,
    *,
    authenticate: Authenticator | None = None,
    lifespan: AppLifespan | None = None,
) -> FastAPI:
    if authenticate is None:
        authenticate = require_api_key(read_api_key(settings.api_key_file))
    configure_logging()
    metrics = GatewayMetrics.create()
    app = FastAPI(lifespan=lifespan)
    app.state.gateway_metrics = metrics

    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next: RequestHandler) -> Response:
        request_id = resolve_request_id(request.headers.get("X-Request-ID"))
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

    @app.exception_handler(ServiceError)
    async def service_error_handler(request: Request, error: ServiceError) -> JSONResponse:
        return _error_response(
            status_code=error.status_code,
            code=error.code,
            message=error.message,
            request_id=_request_id(request),
            details=error.details,
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request,
        _error: RequestValidationError,
    ) -> JSONResponse:
        return _error_response(
            status_code=422,
            code="validation_error",
            message="Request validation failed.",
            request_id=_request_id(request),
        )

    @app.exception_handler(HTTPException)
    async def http_error_handler(request: Request, error: HTTPException) -> JSONResponse:
        if error.status_code == 404:
            code, message = "not_found", "Resource not found."
        elif error.status_code == 405:
            code, message = "method_not_allowed", "Method not allowed."
        else:
            code, message = "http_error", "HTTP request failed."
        return _error_response(
            status_code=error.status_code,
            code=code,
            message=message,
            request_id=_request_id(request),
            headers=_safe_http_exception_headers(error.headers),
        )

    @app.get("/healthz")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    app.include_router(create_management_router(backend_router, authenticate, metrics))

    async def synthesis_response(
        *,
        request_id: str,
        model: ModelID,
        input_chars: int | None,
        operation: Callable[[], Awaitable[AudioResult]],
    ) -> Response:
        started_at = perf_counter()
        model_name = model.value

        try:
            audio = await operation()
        except ServiceError as error:
            telemetry_error_code = normalize_error_code(error.code) or "backend_error"
            metrics.record_request(
                model=model_name,
                status=error.status_code,
                elapsed_seconds=perf_counter() - started_at,
                audio_seconds=None,
            )
            if error.backend_failure:
                metrics.record_backend_error(
                    backend=_backend_name(model),
                    code=telemetry_error_code,
                )
            log_gateway_request(
                request_id=request_id,
                model=model_name,
                input_chars=input_chars or 0,
                queue_ms=None,
                processing_ms=None,
                status=error.status_code,
                error_code=telemetry_error_code,
            )
            raise
        except Exception:
            metrics.record_request(
                model=model_name,
                status=500,
                elapsed_seconds=perf_counter() - started_at,
                audio_seconds=None,
            )
            log_gateway_request(
                request_id=request_id,
                model=model_name,
                input_chars=input_chars or 0,
                queue_ms=None,
                processing_ms=None,
                status=500,
                error_code="internal_error",
            )
            raise ServiceError(
                status_code=500,
                code="internal_error",
                message="Internal gateway error.",
            ) from None

        metrics.record_request(
            model=model_name,
            status=200,
            elapsed_seconds=perf_counter() - started_at,
            audio_seconds=_parse_nonnegative_float(audio.duration_seconds),
        )
        log_gateway_request(
            request_id=request_id,
            model=model_name,
            input_chars=input_chars if input_chars is not None else audio.input_chars or 0,
            queue_ms=audio.queue_ms,
            processing_ms=audio.processing_ms,
            status=200,
            error_code=None,
        )

        headers = {"X-TTS-Model": model_name, "Content-Type": audio.content_type}
        _copy_header(headers, "X-Audio-Duration", audio.duration_seconds)
        _copy_header(headers, "X-Queue-Ms", audio.queue_ms)
        _copy_header(headers, "X-Processing-Ms", audio.processing_ms)
        return Response(content=audio.content, headers=headers)

    @app.post("/v1/audio/speech")
    async def speech(
        request: Request,
        speech_request: SpeechRequest,
        _authenticated: Annotated[None, Depends(authenticate)],
    ) -> Response:
        request_id = _request_id(request)

        async def synthesize() -> AudioResult:
            if not isinstance(backend_router, SpeechSynthesisRouter):
                raise ServiceError(
                    status_code=503,
                    code="backend_not_configured",
                    message="Backend router is not configured.",
                )
            return await backend_router.synthesize(speech_request, request_id)

        return await synthesis_response(
            request_id=request_id,
            model=speech_request.model,
            input_chars=len(speech_request.input),
            operation=synthesize,
        )

    @app.post("/v1/audio/speech/clone")
    async def ephemeral_clone(
        request: Request,
        _authenticated: Annotated[None, Depends(authenticate)],
    ) -> Response:
        request_id = _request_id(request)

        async def synthesize() -> AudioResult:
            if not isinstance(backend_router, EphemeralSynthesisRouter):
                raise ServiceError(
                    status_code=503,
                    code="backend_not_configured",
                    message="Backend router is not configured.",
                )
            return await backend_router.synthesize_ephemeral(
                content=request.stream(),
                content_type=request.headers.get("content-type"),
                content_length=_content_length(request),
                request_id=request_id,
            )

        return await synthesis_response(
            request_id=request_id,
            model=ModelID.QWEN_CLONE,
            input_chars=None,
            operation=synthesize,
        )

    app.add_middleware(
        _SynthesisBodyLimitMiddleware,
        maximum_bytes=settings.max_synthesis_body_bytes,
    )
    return app


def create_runtime_gateway_app(settings: GatewaySettings | None = None) -> FastAPI:
    runtime_settings = settings or GatewaySettings()
    runtime_authenticate = _RuntimeAuthentication()
    runtime_backend = _RuntimeBackendRouter()

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        runtime_authenticate.attach(
            require_api_key(read_api_key(runtime_settings.api_key_file))
        )
        try:
            async with httpx.AsyncClient() as backend_client:
                backend_router = BackendRouter(runtime_settings, backend_client)
                runtime_backend.attach(backend_router)
                application.state.backend_router = backend_router
                try:
                    yield
                finally:
                    application.state.backend_router = None
                    runtime_backend.detach()
        finally:
            runtime_authenticate.detach()

    return create_gateway_app(
        runtime_settings,
        runtime_backend,
        authenticate=runtime_authenticate,
        lifespan=lifespan,
    )


class _RuntimeAuthentication:
    def __init__(self) -> None:
        self._authenticate: Authenticator | None = None

    def attach(self, authenticate: Authenticator) -> None:
        self._authenticate = authenticate

    def detach(self) -> None:
        self._authenticate = None

    def __call__(
        self,
        authorization: Annotated[str | None, Header(alias="Authorization")] = None,
    ) -> None:
        if self._authenticate is None:
            raise RuntimeError("Gateway authentication is unavailable before startup")
        self._authenticate(authorization)


class _RuntimeBackendRouter:
    def __init__(self) -> None:
        self._backend: BackendRouter | None = None

    def attach(self, backend: BackendRouter) -> None:
        self._backend = backend

    def detach(self) -> None:
        self._backend = None

    async def synthesize(self, request: SpeechRequest, request_id: str) -> AudioResult:
        return await self._require_backend().synthesize(request, request_id)

    async def synthesize_ephemeral(
        self,
        *,
        content: AsyncIterator[bytes],
        content_type: str | None,
        content_length: int | None,
        request_id: str,
    ) -> AudioResult:
        return await self._require_backend().synthesize_ephemeral(
            content=content,
            content_type=content_type,
            content_length=content_length,
            request_id=request_id,
        )

    async def discover(self, request_id: str) -> DiscoveryResult:
        return await self._require_backend().discover(request_id)

    async def register_voice(
        self,
        *,
        content: AsyncIterator[bytes],
        content_type: str | None,
        content_length: int | None,
        request_id: str,
    ) -> ProxyResult:
        return await self._require_backend().register_voice(
            content=content,
            content_type=content_type,
            content_length=content_length,
            request_id=request_id,
        )

    async def list_voices(self, request_id: str) -> ProxyResult:
        return await self._require_backend().list_voices(request_id)

    async def get_voice(self, voice_id: str, request_id: str) -> ProxyResult:
        return await self._require_backend().get_voice(voice_id, request_id)

    async def delete_voice(self, voice_id: str, request_id: str) -> ProxyResult:
        return await self._require_backend().delete_voice(voice_id, request_id)

    async def ready(self, request_id: str) -> None:
        await self._require_backend().ready(request_id)

    async def qwen_metrics(self, request_id: str) -> bytes | None:
        return await self._require_backend().qwen_metrics(request_id)

    def _require_backend(self) -> BackendRouter:
        if self._backend is None:
            raise RuntimeError("Gateway backend is unavailable before startup")
        return self._backend


def _request_id(request: Request) -> str:
    request_id = getattr(request.state, "request_id", None)
    if isinstance(request_id, str):
        return request_id
    return resolve_request_id(None)


def _content_length(request: Request) -> int | None:
    raw_content_length = request.headers.get("content-length")
    if raw_content_length is None:
        return None
    try:
        content_length = int(raw_content_length)
    except ValueError:
        raise ServiceError(
            status_code=422,
            code="validation_error",
            message="Request validation failed.",
        ) from None
    if content_length < 0:
        raise ServiceError(
            status_code=422,
            code="validation_error",
            message="Request validation failed.",
        )
    return content_length


def _error_response(
    *,
    status_code: int,
    code: str,
    message: str,
    request_id: str,
    headers: dict[str, str] | None = None,
    details: tuple[ValidationIssue, ...] | None = None,
) -> JSONResponse:
    error_content: dict[str, object] = {
        "code": code,
        "message": message,
        "request_id": request_id,
    }
    if code == "validation_error" and details is not None:
        error_content["details"] = validation_issues_to_public(details)
    return JSONResponse(
        status_code=status_code,
        headers=headers,
        content={"error": error_content},
    )


def _safe_http_exception_headers(headers: Mapping[str, str] | None) -> dict[str, str] | None:
    if not headers:
        return None
    safe_names = {"allow", "retry-after", "www-authenticate"}
    safe_headers = {name: value for name, value in headers.items() if name.lower() in safe_names}
    return safe_headers or None


class _SynthesisBodyLimitMiddleware:
    def __init__(self, app: ASGIApp, maximum_bytes: int) -> None:
        self._app = app
        self._maximum_bytes = maximum_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if not self._applies(scope):
            await self._app(scope, receive, send)
            return

        request_id = resolve_request_id(_scope_header(scope, b"x-request-id"))
        content_length = _parse_content_length(_scope_header(scope, b"content-length"))
        if content_length is not None and content_length > self._maximum_bytes:
            await _send_request_too_large(send, request_id)
            return

        messages: list[Message] = []
        received_bytes = 0
        while True:
            message = await receive()
            if message["type"] != "http.request":
                messages.append(message)
                break
            body = message.get("body", b"")
            received_bytes += len(body)
            if received_bytes > self._maximum_bytes:
                await _send_request_too_large(send, request_id)
                return
            messages.append(message)
            if not message.get("more_body", False):
                break

        async def replay_receive() -> Message:
            if messages:
                return messages.pop(0)
            return await receive()

        await self._app(scope, replay_receive, send)

    @staticmethod
    def _applies(scope: Scope) -> bool:
        return (
            scope["type"] == "http"
            and scope.get("method") == "POST"
            and scope.get("path") == "/v1/audio/speech"
        )


def _scope_header(scope: Scope, name: bytes) -> str | None:
    headers = cast(list[tuple[bytes, bytes]], scope.get("headers", []))
    for header_name, value in headers:
        if header_name.lower() == name:
            return value.decode("latin-1")
    return None


def _parse_content_length(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        parsed = int(value)
    except ValueError:
        return None
    return parsed if parsed >= 0 else None


async def _send_request_too_large(send: Send, request_id: str) -> None:
    body = json.dumps(
        {
            "error": {
                "code": "request_too_large",
                "message": "Synthesis request exceeds the size limit.",
                "request_id": request_id,
            }
        },
        separators=(",", ":"),
    ).encode("utf-8")
    await send(
        {
            "type": "http.response.start",
            "status": 413,
            "headers": [
                (b"content-length", str(len(body)).encode("ascii")),
                (b"content-type", b"application/json"),
                (b"x-request-id", request_id.encode("ascii")),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})


def _backend_name(model: ModelID) -> str:
    if model == ModelID.SUPERTONIC:
        return "supertonic"
    return "qwen"


def _parse_nonnegative_float(value: str | None) -> float | None:
    normalized = normalize_bounded_nonnegative(
        value,
        maximum=MAX_AUDIO_DURATION_SECONDS,
    )
    if normalized is None:
        return None
    return float(normalized)


def _copy_header(headers: dict[str, str], name: str, value: str | None) -> None:
    if value is not None:
        headers[name] = value


app = create_runtime_gateway_app()
