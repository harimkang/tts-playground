from collections.abc import Awaitable, Callable
from typing import Annotated

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.responses import Response

from tts_playground.common.errors import ServiceError
from tts_playground.common.request_id import resolve_request_id
from tts_playground.common.settings import GatewaySettings
from tts_playground.gateway.auth import read_api_key, require_api_key
from tts_playground.gateway.schemas import SpeechRequest

RequestHandler = Callable[[Request], Awaitable[Response]]


def create_gateway_app(settings: GatewaySettings, backend_router: object | None) -> FastAPI:
    expected_api_key = read_api_key(settings.api_key_file)
    authenticate = require_api_key(expected_api_key)
    app = FastAPI()

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

    @app.get("/healthz")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/audio/speech")
    async def speech(
        _speech_request: SpeechRequest,
        _authenticated: Annotated[None, Depends(authenticate)],
    ) -> Response:
        if backend_router is None:
            raise ServiceError(
                status_code=503,
                code="backend_not_configured",
                message="Backend router is not configured.",
            )
        raise NotImplementedError("Backend routing is implemented in Task 3")

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
