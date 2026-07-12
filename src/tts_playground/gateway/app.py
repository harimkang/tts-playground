from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from time import perf_counter
from typing import Annotated, Protocol, runtime_checkable

import httpx
from fastapi import Depends, FastAPI, Header, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException
from starlette.responses import Response

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
class SynthesisRouter(Protocol):
    async def synthesize(self, request: SpeechRequest, request_id: str) -> AudioResult: ...


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
        )

    @app.get("/healthz")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    app.include_router(create_management_router(backend_router, authenticate, metrics))

    @app.post("/v1/audio/speech")
    async def speech(
        request: Request,
        speech_request: SpeechRequest,
        _authenticated: Annotated[None, Depends(authenticate)],
    ) -> Response:
        started_at = perf_counter()
        request_id = _request_id(request)
        model = speech_request.model.value

        try:
            if not isinstance(backend_router, SynthesisRouter):
                raise ServiceError(
                    status_code=503,
                    code="backend_not_configured",
                    message="Backend router is not configured.",
                )
            audio = await backend_router.synthesize(speech_request, request_id)
        except ServiceError as error:
            telemetry_error_code = normalize_error_code(error.code) or "backend_error"
            metrics.record_request(
                model=model,
                status=error.status_code,
                elapsed_seconds=perf_counter() - started_at,
                audio_seconds=None,
            )
            metrics.record_backend_error(
                backend=_backend_name(speech_request.model),
                code=telemetry_error_code,
            )
            log_gateway_request(
                request_id=request_id,
                model=model,
                input_chars=len(speech_request.input),
                queue_ms=None,
                processing_ms=None,
                status=error.status_code,
                error_code=telemetry_error_code,
            )
            raise

        metrics.record_request(
            model=model,
            status=200,
            elapsed_seconds=perf_counter() - started_at,
            audio_seconds=_parse_nonnegative_float(audio.duration_seconds),
        )
        log_gateway_request(
            request_id=request_id,
            model=model,
            input_chars=len(speech_request.input),
            queue_ms=audio.queue_ms,
            processing_ms=audio.processing_ms,
            status=200,
            error_code=None,
        )

        headers = {"X-TTS-Model": model, "Content-Type": audio.content_type}
        _copy_header(headers, "X-Audio-Duration", audio.duration_seconds)
        _copy_header(headers, "X-Queue-Ms", audio.queue_ms)
        _copy_header(headers, "X-Processing-Ms", audio.processing_ms)
        return Response(content=audio.content, headers=headers)

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


def _error_response(*, status_code: int, code: str, message: str, request_id: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "code": code,
                "message": message,
                "request_id": request_id,
            }
        },
    )


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
