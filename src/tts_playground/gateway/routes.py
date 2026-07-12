from collections.abc import AsyncIterator, Callable
from typing import Annotated, Protocol, runtime_checkable

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.responses import Response

from tts_playground.common.errors import ServiceError
from tts_playground.common.models import ModelID
from tts_playground.common.request_id import resolve_request_id
from tts_playground.gateway.backends import DiscoveryResult, ProxyResult
from tts_playground.gateway.metrics import GatewayMetrics

QWEN_CUSTOM_SPEAKERS = [
    "Vivian",
    "Serena",
    "Uncle_Fu",
    "Dylan",
    "Eric",
    "Ryan",
    "Aiden",
    "Ono_Anna",
    "Sohee",
]

_SUPERTONIC_CAPABILITIES = {
    "instructions": False,
    "speed": True,
    "steps": True,
    "voice": True,
    "voice_cloning": False,
    "voice_design": False,
}
_QWEN_CUSTOM_CAPABILITIES = {
    "instructions": True,
    "speed": False,
    "steps": False,
    "voice": True,
    "voice_cloning": False,
    "voice_design": False,
}
_QWEN_DESIGN_CAPABILITIES = {
    "instructions": True,
    "speed": False,
    "steps": False,
    "voice": False,
    "voice_cloning": False,
    "voice_design": True,
}
_QWEN_CLONE_CAPABILITIES = {
    "instructions": False,
    "speed": False,
    "steps": False,
    "voice": True,
    "voice_cloning": True,
    "voice_design": False,
}


@runtime_checkable
class ManagementBackend(Protocol):
    async def discover(self, request_id: str) -> DiscoveryResult: ...

    async def register_voice(
        self,
        *,
        content: AsyncIterator[bytes],
        content_type: str | None,
        content_length: int | None,
        request_id: str,
    ) -> ProxyResult: ...

    async def list_voices(self, request_id: str) -> ProxyResult: ...

    async def get_voice(self, voice_id: str, request_id: str) -> ProxyResult: ...

    async def delete_voice(self, voice_id: str, request_id: str) -> ProxyResult: ...

    async def ready(self, request_id: str) -> None: ...

    async def qwen_metrics(self, request_id: str) -> bytes | None: ...


def create_management_router(
    backend_router: object | None,
    authenticate: Callable[[str | None], None],
    metrics: GatewayMetrics,
) -> APIRouter:
    router = APIRouter()
    authentication = Annotated[None, Depends(authenticate)]

    @router.get("/v1/models")
    async def models(
        request: Request,
        _authenticated: authentication,
    ) -> dict[str, object]:
        discovery = await _backend(backend_router).discover(_request_id(request))
        return _models_response(discovery)

    @router.post("/v1/audio/voices", status_code=201)
    async def register_voice(
        request: Request,
        _authenticated: authentication,
    ) -> JSONResponse:
        result = await _backend(backend_router).register_voice(
            content=request.stream(),
            content_type=request.headers.get("content-type"),
            content_length=_content_length(request),
            request_id=_request_id(request),
        )
        return JSONResponse(status_code=result.status_code, content=result.payload)

    @router.get("/v1/audio/voices")
    async def list_voices(
        request: Request,
        _authenticated: authentication,
    ) -> JSONResponse:
        result = await _backend(backend_router).list_voices(_request_id(request))
        return JSONResponse(status_code=result.status_code, content=result.payload)

    @router.get("/v1/audio/voices/{voice_id}")
    async def get_voice(
        request: Request,
        voice_id: str,
        _authenticated: authentication,
    ) -> JSONResponse:
        result = await _backend(backend_router).get_voice(voice_id, _request_id(request))
        return JSONResponse(status_code=result.status_code, content=result.payload)

    @router.delete("/v1/audio/voices/{voice_id}", status_code=204)
    async def delete_voice(
        request: Request,
        voice_id: str,
        _authenticated: authentication,
    ) -> Response:
        result = await _backend(backend_router).delete_voice(voice_id, _request_id(request))
        return Response(status_code=result.status_code)

    @router.get("/readyz")
    async def ready(
        request: Request,
        _authenticated: authentication,
    ) -> dict[str, str]:
        await _backend(backend_router).ready(_request_id(request))
        return {"status": "ready"}

    @router.get("/metrics")
    async def prometheus_metrics(
        request: Request,
        _authenticated: authentication,
    ) -> Response:
        content = generate_latest(metrics.registry)
        if isinstance(backend_router, ManagementBackend):
            qwen_content = await backend_router.qwen_metrics(_request_id(request))
            if qwen_content:
                content = (
                    content.rstrip(b"\n")
                    + b"\n"
                    + qwen_content.strip(b"\n")
                    + b"\n"
                )
        return Response(
            content=content,
            headers={"Content-Type": CONTENT_TYPE_LATEST},
        )

    return router


def _backend(backend_router: object | None) -> ManagementBackend:
    if isinstance(backend_router, ManagementBackend):
        return backend_router
    raise ServiceError(
        status_code=503,
        code="backend_not_configured",
        message="Backend router is not configured.",
    )


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


def _models_response(discovery: DiscoveryResult) -> dict[str, object]:
    def qwen_model(
        model: ModelID,
        capabilities: dict[str, bool],
        voices: list[str],
    ) -> dict[str, object]:
        return {
            "id": model.value,
            "object": "model",
            "backend": "qwen",
            "voices": voices,
            "speakers": voices,
            "capabilities": capabilities,
            "worker_state": discovery.qwen_state,
            "loaded": discovery.qwen_loaded_model == model.value,
        }

    return {
        "object": "list",
        "data": [
            {
                "id": ModelID.SUPERTONIC.value,
                "object": "model",
                "backend": "supertonic",
                "voices": discovery.supertonic_voices,
                "speakers": discovery.supertonic_voices,
                "capabilities": _SUPERTONIC_CAPABILITIES,
                "worker_state": "ready",
                "loaded": True,
            },
            qwen_model(
                ModelID.QWEN_CUSTOM,
                _QWEN_CUSTOM_CAPABILITIES,
                QWEN_CUSTOM_SPEAKERS,
            ),
            qwen_model(ModelID.QWEN_DESIGN, _QWEN_DESIGN_CAPABILITIES, []),
            qwen_model(ModelID.QWEN_CLONE, _QWEN_CLONE_CAPABILITIES, []),
        ],
    }
