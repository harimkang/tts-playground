from dataclasses import dataclass

import httpx

from tts_playground.common.errors import ServiceError
from tts_playground.common.models import ModelID
from tts_playground.common.settings import GatewaySettings
from tts_playground.gateway.schemas import SpeechRequest

QWEN_LANGUAGES = {
    "auto": "Auto",
    "zh": "Chinese",
    "en": "English",
    "ja": "Japanese",
    "ko": "Korean",
    "de": "German",
    "fr": "French",
    "ru": "Russian",
    "pt": "Portuguese",
    "es": "Spanish",
    "it": "Italian",
}

SUPERTONIC_LANGUAGES = {
    "en",
    "ko",
    "ja",
    "ar",
    "bg",
    "cs",
    "da",
    "de",
    "el",
    "es",
    "et",
    "fi",
    "fr",
    "hi",
    "hr",
    "hu",
    "id",
    "it",
    "lt",
    "lv",
    "nl",
    "pl",
    "pt",
    "ro",
    "ru",
    "sk",
    "sl",
    "sv",
    "tr",
    "uk",
    "vi",
    "na",
}


@dataclass(slots=True)
class AudioResult:
    content: bytes
    content_type: str
    duration_seconds: str | None
    queue_ms: str | None
    processing_ms: str | None


class BackendRouter:
    def __init__(self, settings: GatewaySettings, client: httpx.AsyncClient) -> None:
        self._settings = settings
        self._client = client

    async def synthesize(self, request: SpeechRequest, request_id: str) -> AudioResult:
        if request.model == ModelID.SUPERTONIC:
            response = await self._post_supertonic(request, request_id)
        else:
            response = await self._post_qwen(request, request_id)

        if not response.is_success:
            _raise_upstream_error(response)

        return AudioResult(
            content=response.content,
            content_type=response.headers.get("content-type", "audio/wav"),
            duration_seconds=response.headers.get("x-audio-duration"),
            queue_ms=response.headers.get("x-queue-ms"),
            processing_ms=response.headers.get("x-processing-ms"),
        )

    async def _post_supertonic(
        self,
        request: SpeechRequest,
        request_id: str,
    ) -> httpx.Response:
        if request.language not in SUPERTONIC_LANGUAGES:
            raise ServiceError(
                status_code=422,
                code="unsupported_language",
                message=f"Supertonic does not support language '{request.language}'.",
            )

        payload: dict[str, object] = {
            "text": request.input,
            "voice": request.voice,
            "lang": request.language,
            "speed": request.speed,
            "steps": request.steps if request.steps is not None else 8,
            "response_format": request.response_format,
        }
        return await self._post(
            f"{self._settings.supertonic_url.rstrip('/')}/v1/tts",
            payload=payload,
            request_id=request_id,
            timeout_seconds=self._settings.supertonic_timeout_seconds,
        )

    async def _post_qwen(self, request: SpeechRequest, request_id: str) -> httpx.Response:
        language = QWEN_LANGUAGES.get(request.language)
        if language is None:
            raise ServiceError(
                status_code=422,
                code="unsupported_language",
                message=f"Qwen does not support language '{request.language}'.",
            )

        payload: dict[str, object] = {
            "model": request.model.value,
            "text": request.input,
            "language": language,
            "voice": request.voice,
            "instructions": request.instructions,
        }
        return await self._post(
            f"{self._settings.qwen_url.rstrip('/')}/internal/v1/speech",
            payload=payload,
            request_id=request_id,
            timeout_seconds=self._settings.qwen_timeout_seconds,
        )

    async def _post(
        self,
        url: str,
        *,
        payload: dict[str, object],
        request_id: str,
        timeout_seconds: float,
    ) -> httpx.Response:
        try:
            return await self._client.post(
                url,
                json=payload,
                headers={"X-Request-ID": request_id},
                timeout=timeout_seconds,
            )
        except httpx.TimeoutException as error:
            raise ServiceError(
                status_code=504,
                code="backend_timeout",
                message="Backend request timed out.",
            ) from error
        except httpx.RequestError as error:
            raise ServiceError(
                status_code=503,
                code="backend_unavailable",
                message="Backend is unavailable.",
            ) from error


def _raise_upstream_error(response: httpx.Response) -> None:
    try:
        payload = response.json()
    except ValueError:
        payload = None

    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            code = error.get("code")
            message = error.get("message")
            if isinstance(code, str) and isinstance(message, str):
                raise ServiceError(
                    status_code=response.status_code,
                    code=code,
                    message=message,
                )

    raise ServiceError(
        status_code=response.status_code,
        code="backend_error",
        message="Backend request failed.",
    )
