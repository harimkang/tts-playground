from collections.abc import Awaitable, Callable
from time import perf_counter
from typing import Annotated, Protocol, runtime_checkable

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
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
from tts_playground.gateway.backends import AudioResult
from tts_playground.gateway.metrics import GatewayMetrics
from tts_playground.gateway.schemas import SpeechRequest

RequestHandler = Callable[[Request], Awaitable[Response]]


@runtime_checkable
class SynthesisRouter(Protocol):
    async def synthesize(self, request: SpeechRequest, request_id: str) -> AudioResult: ...


def create_gateway_app(settings: GatewaySettings, backend_router: object | None) -> FastAPI:
    expected_api_key = read_api_key(settings.api_key_file)
    authenticate = require_api_key(expected_api_key)
    configure_logging()
    metrics = GatewayMetrics.create()
    app = FastAPI()
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

    @app.get("/metrics")
    async def prometheus_metrics(
        _authenticated: Annotated[None, Depends(authenticate)],
    ) -> Response:
        return Response(
            content=generate_latest(metrics.registry),
            headers={"Content-Type": CONTENT_TYPE_LATEST},
        )

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
