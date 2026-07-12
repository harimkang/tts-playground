from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from time import perf_counter
from typing import Annotated, cast

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.exceptions import HTTPException
from starlette.responses import Response

from tts_playground.common.errors import ServiceError
from tts_playground.common.request_id import resolve_request_id
from tts_playground.common.settings import QwenSettings
from tts_playground.qwen.adapter import (
    InternalSpeechRequest,
    QwenAdapter,
    SynthesisResult,
)
from tts_playground.qwen.manager import QwenModelManager
from tts_playground.qwen.metrics import QwenMetrics
from tts_playground.qwen.real_adapter import QwenTTSAdapter
from tts_playground.qwen.schemas import (
    QwenSpeechRequest,
    VoiceResponse,
    WorkerStatusResponse,
)
from tts_playground.qwen.voice_registry import MAX_AUDIO_BYTES, VoiceRegistry

RequestHandler = Callable[[Request], Awaitable[Response]]
_UPLOAD_CHUNK_BYTES = 1024 * 1024


class _ObservedAdapter:
    def __init__(self, adapter: QwenAdapter, metrics: QwenMetrics) -> None:
        self._adapter = adapter
        self._metrics = metrics

    def free_vram_bytes(self) -> int:
        free_bytes = self._adapter.free_vram_bytes()
        self._metrics.observe_free_vram(free_bytes)
        return free_bytes

    def load(self, repository: str, attention_implementation: str) -> object:
        return self._adapter.load(repository, attention_implementation)

    def cleanup_device_memory(self) -> None:
        self._adapter.cleanup_device_memory()

    def build_clone_prompt(
        self,
        model: object,
        audio_path: Path,
        transcript: str,
    ) -> object:
        return self._adapter.build_clone_prompt(model, audio_path, transcript)

    def synthesize(
        self,
        model: object,
        request: InternalSpeechRequest,
        clone_prompt: object | None,
    ) -> SynthesisResult:
        return self._adapter.synthesize(model, request, clone_prompt)


def create_qwen_app(settings: QwenSettings, adapter: QwenAdapter) -> FastAPI:
    metrics = QwenMetrics.create()

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        registry = await asyncio.to_thread(VoiceRegistry, settings.voice_data_dir)
        manager = QwenModelManager(
            adapter=_ObservedAdapter(adapter, metrics),
            registry=registry,
            settings=settings,
        )
        application.state.voice_registry = registry
        application.state.qwen_manager = manager
        idle_task = asyncio.create_task(
            _idle_unload_loop(manager, settings.idle_unload_seconds)
        )
        try:
            yield
        finally:
            try:
                idle_task.cancel()
                with suppress(asyncio.CancelledError):
                    await idle_task
            finally:
                try:
                    await manager.shutdown()
                finally:
                    application.state.qwen_manager = None
                    application.state.voice_registry = None

    app = FastAPI(lifespan=lifespan)
    app.state.qwen_manager = None
    app.state.voice_registry = None
    app.state.qwen_metrics = metrics

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

    @app.exception_handler(Exception)
    async def unexpected_error_handler(request: Request, _error: Exception) -> JSONResponse:
        return _error_response(
            status_code=500,
            code="internal_error",
            message="Internal worker error.",
            request_id=_request_id(request),
        )

    @app.get("/healthz")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    async def ready(request: Request) -> dict[str, str]:
        manager = _manager(request)
        status = manager.status()
        metrics.observe_status(status)
        if manager.in_error_cooldown():
            raise ServiceError(
                status_code=503,
                code="model_unavailable",
                message="Qwen model is recovering from an error.",
            )
        return {"status": "ready", "state": status.state.value}

    @app.get("/metrics")
    async def prometheus_metrics(request: Request) -> Response:
        metrics.observe_status(_manager(request).status())
        return Response(
            content=generate_latest(metrics.registry),
            headers={"Content-Type": CONTENT_TYPE_LATEST},
        )

    @app.get("/internal/v1/status")
    async def worker_status(request: Request) -> WorkerStatusResponse:
        status = _manager(request).status()
        metrics.observe_status(status)
        return WorkerStatusResponse(
            state=status.state.value,
            loaded_model=status.loaded_model,
            queue_depth=status.queue_depth,
            last_error_code=status.last_error_code,
        )

    @app.post("/internal/v1/speech")
    async def speech(request: Request, body: QwenSpeechRequest) -> Response:
        manager = _manager(request)
        started_at = perf_counter()
        model = body.model.value
        try:
            result = await manager.synthesize(body.to_internal())
        except ServiceError as error:
            elapsed_seconds = perf_counter() - started_at
            metrics.record_request(
                model=model,
                status=error.status_code,
                elapsed_seconds=elapsed_seconds,
                audio_seconds=None,
            )
            metrics.record_error(error.code)
            metrics.observe_status(manager.status())
            raise
        except Exception:
            elapsed_seconds = perf_counter() - started_at
            metrics.record_request(
                model=model,
                status=500,
                elapsed_seconds=elapsed_seconds,
                audio_seconds=None,
            )
            metrics.record_error("internal_error")
            metrics.observe_status(manager.status())
            raise

        elapsed_seconds = perf_counter() - started_at
        metrics.record_request(
            model=model,
            status=200,
            elapsed_seconds=elapsed_seconds,
            audio_seconds=result.duration_seconds,
        )
        metrics.observe_status(manager.status())
        return Response(
            content=result.audio,
            headers={
                "Content-Type": "audio/wav",
                "X-TTS-Model": model,
                "X-Audio-Duration": str(result.duration_seconds),
                "X-Processing-Ms": f"{elapsed_seconds * 1000:.3f}",
            },
        )

    @app.post("/internal/v1/voices", status_code=201)
    async def register_voice(
        request: Request,
        name: Annotated[str, Form(min_length=1, max_length=100)],
        audio: Annotated[UploadFile, File()],
        transcript: Annotated[str, Form(min_length=1, max_length=4000)],
        language: Annotated[str, Form(min_length=2, max_length=10)],
        consent_confirmed: Annotated[bool, Form()],
    ) -> VoiceResponse:
        audio_bytes = await _read_bounded_upload(audio)
        voice = await asyncio.to_thread(
            _registry(request).register,
            name=name,
            audio=audio_bytes,
            transcript=transcript,
            language=language,
            consent_confirmed=consent_confirmed,
        )
        return VoiceResponse.model_validate(voice)

    @app.get("/internal/v1/voices")
    async def list_voices(request: Request) -> list[VoiceResponse]:
        voices = await asyncio.to_thread(_registry(request).list_public)
        return [VoiceResponse.model_validate(voice) for voice in voices]

    @app.get("/internal/v1/voices/{voice_id}")
    async def get_voice(request: Request, voice_id: str) -> VoiceResponse:
        voice = await asyncio.to_thread(_registry(request).get_public, voice_id)
        return VoiceResponse.model_validate(voice)

    @app.delete("/internal/v1/voices/{voice_id}", status_code=204)
    async def delete_voice(request: Request, voice_id: str) -> Response:
        await asyncio.to_thread(_registry(request).delete, voice_id)
        _manager(request).invalidate_voice(voice_id)
        return Response(status_code=204)

    return app


async def _read_bounded_upload(audio: UploadFile) -> bytes:
    content = bytearray()
    while chunk := await audio.read(_UPLOAD_CHUNK_BYTES):
        content.extend(chunk)
        if len(content) > MAX_AUDIO_BYTES:
            raise ServiceError(
                status_code=413,
                code="audio_too_large",
                message="Reference audio exceeds the 20 MB limit.",
            )
    return bytes(content)


async def _idle_unload_loop(
    manager: QwenModelManager,
    idle_unload_seconds: float,
) -> None:
    interval = min(30.0, idle_unload_seconds / 2)
    while True:
        await asyncio.sleep(interval)
        await manager.unload_if_idle()


def _manager(request: Request) -> QwenModelManager:
    manager = request.app.state.qwen_manager
    if manager is None:
        raise ServiceError(
            status_code=503,
            code="worker_not_ready",
            message="Qwen worker is not ready.",
        )
    return cast(QwenModelManager, manager)


def _registry(request: Request) -> VoiceRegistry:
    registry = request.app.state.voice_registry
    if registry is None:
        raise ServiceError(
            status_code=503,
            code="worker_not_ready",
            message="Qwen worker is not ready.",
        )
    return cast(VoiceRegistry, registry)


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


app = create_qwen_app(QwenSettings(), QwenTTSAdapter())
