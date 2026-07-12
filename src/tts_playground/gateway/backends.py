from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Final
from urllib.parse import quote

import httpx
from python_multipart import MultipartParser
from python_multipart.multipart import parse_options_header

from tts_playground.common.errors import ServiceError
from tts_playground.common.logging import (
    MAX_AUDIO_DURATION_SECONDS,
    MAX_TIMING_MILLISECONDS,
    normalize_bounded_nonnegative,
    normalize_error_code,
)
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

_UPLOAD_MULTIPART_OVERHEAD_BYTES: Final = 16 * 1024
_PUBLIC_VOICE_FIELDS: Final = (
    "id",
    "name",
    "language",
    "duration_seconds",
    "consent_confirmed",
    "created_at",
)


@dataclass(slots=True)
class AudioResult:
    content: bytes
    content_type: str
    duration_seconds: str | None
    queue_ms: str | None
    processing_ms: str | None


@dataclass(frozen=True, slots=True)
class DiscoveryResult:
    supertonic_voices: list[str]
    qwen_state: str
    qwen_loaded_model: str | None


@dataclass(frozen=True, slots=True)
class ProxyResult:
    status_code: int
    payload: object | None


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
        _validate_wav_response(response)

        return AudioResult(
            content=response.content,
            content_type="audio/wav",
            duration_seconds=normalize_bounded_nonnegative(
                response.headers.get("x-audio-duration"),
                maximum=MAX_AUDIO_DURATION_SECONDS,
            ),
            queue_ms=normalize_bounded_nonnegative(
                response.headers.get("x-queue-ms"),
                maximum=MAX_TIMING_MILLISECONDS,
            ),
            processing_ms=normalize_bounded_nonnegative(
                response.headers.get("x-processing-ms"),
                maximum=MAX_TIMING_MILLISECONDS,
            ),
        )

    async def discover(self, request_id: str) -> DiscoveryResult:
        styles_response = await self._request(
            "GET",
            f"{self._settings.supertonic_url.rstrip('/')}/v1/styles",
            request_id=request_id,
            timeout_seconds=self._settings.supertonic_timeout_seconds,
        )
        if not styles_response.is_success:
            _raise_upstream_error(styles_response)
        voices = _read_supertonic_voices(styles_response)

        status_response = await self._request(
            "GET",
            f"{self._settings.qwen_url.rstrip('/')}/internal/v1/status",
            request_id=request_id,
            timeout_seconds=self._settings.qwen_timeout_seconds,
        )
        if not status_response.is_success:
            _raise_upstream_error(status_response)
        state, loaded_model = _read_qwen_status(status_response)
        return DiscoveryResult(
            supertonic_voices=voices,
            qwen_state=state,
            qwen_loaded_model=loaded_model,
        )

    async def register_voice(
        self,
        *,
        content: AsyncIterator[bytes],
        content_type: str | None,
        content_length: int | None,
        request_id: str,
    ) -> ProxyResult:
        limiter = _MultipartAudioLimiter(content_type, self._settings.max_upload_bytes)
        if (
            content_length is not None
            and content_length
            > self._settings.max_upload_bytes + _UPLOAD_MULTIPART_OVERHEAD_BYTES
        ):
            raise _audio_too_large()

        async def bounded_content() -> AsyncIterator[bytes]:
            async for chunk in content:
                limiter.write(chunk)
                yield chunk
            limiter.finalize()

        headers = {
            "Content-Type": limiter.content_type,
            "X-Request-ID": request_id,
        }
        if content_length is not None:
            headers["Content-Length"] = str(content_length)
        response = await self._request(
            "POST",
            f"{self._settings.qwen_url.rstrip('/')}/internal/v1/voices",
            request_id=request_id,
            timeout_seconds=self._settings.qwen_timeout_seconds,
            headers=headers,
            content=bounded_content(),
        )
        if not response.is_success:
            _raise_upstream_error(response)
        return ProxyResult(
            status_code=response.status_code,
            payload=_public_voice(_read_json(response)),
        )

    async def list_voices(self, request_id: str) -> ProxyResult:
        response = await self._qwen_voice_request("GET", None, request_id)
        payload = _read_json(response)
        if not isinstance(payload, list):
            raise _invalid_backend_response()
        return ProxyResult(
            status_code=response.status_code,
            payload=[_public_voice(voice) for voice in payload],
        )

    async def get_voice(self, voice_id: str, request_id: str) -> ProxyResult:
        response = await self._qwen_voice_request("GET", voice_id, request_id)
        return ProxyResult(
            status_code=response.status_code,
            payload=_public_voice(_read_json(response)),
        )

    async def delete_voice(self, voice_id: str, request_id: str) -> ProxyResult:
        response = await self._qwen_voice_request("DELETE", voice_id, request_id)
        return ProxyResult(status_code=response.status_code, payload=None)

    async def ready(self, request_id: str) -> None:
        await self._check_ready(
            f"{self._settings.supertonic_url.rstrip('/')}/v1/styles",
            request_id,
            self._settings.supertonic_timeout_seconds,
        )
        await self._check_ready(
            f"{self._settings.qwen_url.rstrip('/')}/readyz",
            request_id,
            self._settings.qwen_timeout_seconds,
        )

    async def qwen_metrics(self, request_id: str) -> bytes | None:
        try:
            response = await self._client.get(
                f"{self._settings.qwen_url.rstrip('/')}/metrics",
                headers={"X-Request-ID": request_id},
                timeout=2.0,
            )
        except Exception:
            # Metrics are best-effort and must never make gateway metrics unavailable.
            return None
        if response.status_code != 200 or not _is_prometheus_text(response):
            return None
        return response.content

    async def _qwen_voice_request(
        self,
        method: str,
        voice_id: str | None,
        request_id: str,
    ) -> httpx.Response:
        path = "/internal/v1/voices"
        if voice_id is not None:
            path = f"{path}/{quote(voice_id, safe='')}"
        response = await self._request(
            method,
            f"{self._settings.qwen_url.rstrip('/')}{path}",
            request_id=request_id,
            timeout_seconds=self._settings.qwen_timeout_seconds,
        )
        if not response.is_success:
            _raise_upstream_error(response)
        return response

    async def _check_ready(
        self,
        url: str,
        request_id: str,
        timeout_seconds: float,
    ) -> None:
        try:
            response = await self._client.get(
                url,
                headers={"X-Request-ID": request_id},
                timeout=timeout_seconds,
            )
        except httpx.RequestError as error:
            raise ServiceError(
                status_code=503,
                code="backend_unavailable",
                message="Required backend is unavailable.",
            ) from error
        if response.is_success:
            return
        if url.endswith("/readyz"):
            _raise_upstream_error(response)
        raise ServiceError(
            status_code=503,
            code="backend_unavailable",
            message="Required backend is unavailable.",
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
        return await self._request(
            "POST",
            url,
            request_id=request_id,
            timeout_seconds=timeout_seconds,
            json=payload,
        )

    async def _request(
        self,
        method: str,
        url: str,
        *,
        request_id: str,
        timeout_seconds: float,
        headers: dict[str, str] | None = None,
        content: AsyncIterator[bytes] | None = None,
        json: dict[str, object] | None = None,
    ) -> httpx.Response:
        request_headers = {"X-Request-ID": request_id}
        if headers is not None:
            request_headers.update(headers)
        try:
            return await self._client.request(
                method,
                url,
                content=content,
                json=json,
                headers=request_headers,
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


class _MultipartAudioLimiter:
    def __init__(self, content_type: str | None, maximum: int) -> None:
        media_type, options = parse_options_header(content_type)
        boundary = options.get(b"boundary")
        if media_type != b"multipart/form-data" or boundary is None:
            raise ServiceError(
                status_code=422,
                code="validation_error",
                message="Request validation failed.",
            )
        self.content_type = content_type or ""
        self._maximum = maximum
        self._audio_bytes = 0
        self._header_field = bytearray()
        self._header_value = bytearray()
        self._content_disposition = b""
        self._is_audio = False
        self._parser = MultipartParser(
            boundary,
            {
                "on_part_begin": self._on_part_begin,
                "on_header_field": self._on_header_field,
                "on_header_value": self._on_header_value,
                "on_header_end": self._on_header_end,
                "on_headers_finished": self._on_headers_finished,
                "on_part_data": self._on_part_data,
                "on_part_end": self._on_part_end,
            },
        )

    def write(self, chunk: bytes) -> None:
        self._parser.write(chunk)

    def finalize(self) -> None:
        self._parser.finalize()

    def _on_part_begin(self) -> None:
        self._content_disposition = b""
        self._is_audio = False

    def _on_header_field(self, data: bytes, start: int, end: int) -> None:
        self._header_field.extend(data[start:end])

    def _on_header_value(self, data: bytes, start: int, end: int) -> None:
        self._header_value.extend(data[start:end])

    def _on_header_end(self) -> None:
        if bytes(self._header_field).lower() == b"content-disposition":
            self._content_disposition = bytes(self._header_value)
        self._header_field.clear()
        self._header_value.clear()

    def _on_headers_finished(self) -> None:
        _disposition, options = parse_options_header(self._content_disposition)
        self._is_audio = options.get(b"name") == b"audio"

    def _on_part_data(self, data: bytes, start: int, end: int) -> None:
        if not self._is_audio:
            return
        self._audio_bytes += end - start
        if self._audio_bytes > self._maximum:
            raise _audio_too_large()

    def _on_part_end(self) -> None:
        self._is_audio = False


def _read_json(response: httpx.Response) -> object:
    try:
        return response.json()
    except ValueError:
        raise _invalid_backend_response() from None


def _read_supertonic_voices(response: httpx.Response) -> list[str]:
    payload = _read_json(response)
    if not isinstance(payload, dict) or not isinstance(payload.get("styles"), list):
        raise _invalid_backend_response()
    voices: list[str] = []
    for style in payload["styles"]:
        if not isinstance(style, dict) or not isinstance(style.get("name"), str):
            raise _invalid_backend_response()
        voices.append(style["name"])
    return voices


def _read_qwen_status(response: httpx.Response) -> tuple[str, str | None]:
    payload = _read_json(response)
    if not isinstance(payload, dict) or not isinstance(payload.get("state"), str):
        raise _invalid_backend_response()
    loaded_model = payload.get("loaded_model")
    if loaded_model is not None and not isinstance(loaded_model, str):
        raise _invalid_backend_response()
    return payload["state"], loaded_model


def _public_voice(payload: object) -> dict[str, object]:
    if not isinstance(payload, dict) or any(field not in payload for field in _PUBLIC_VOICE_FIELDS):
        raise _invalid_backend_response()
    return {field: payload[field] for field in _PUBLIC_VOICE_FIELDS}


def _is_prometheus_text(response: httpx.Response) -> bool:
    media_type = response.headers.get("content-type", "").partition(";")[0].strip().lower()
    return media_type in {"text/plain", "application/openmetrics-text"}


def _audio_too_large() -> ServiceError:
    return ServiceError(
        status_code=413,
        code="audio_too_large",
        message="Reference audio exceeds the 20 MB limit.",
    )


def _invalid_backend_response() -> ServiceError:
    return ServiceError(
        status_code=503,
        code="backend_invalid_response",
        message="Backend returned an invalid response.",
    )


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
                    code=normalize_error_code(code) or "backend_error",
                    message=message,
                )

    raise ServiceError(
        status_code=response.status_code,
        code="backend_error",
        message="Backend request failed.",
    )


def _validate_wav_response(response: httpx.Response) -> None:
    content_type = response.headers.get("content-type", "")
    media_type = content_type.partition(";")[0].strip().lower()
    content = response.content
    if media_type == "audio/wav" and content[:4] == b"RIFF" and content[8:12] == b"WAVE":
        return
    raise ServiceError(
        status_code=503,
        code="backend_invalid_response",
        message="Backend returned an invalid audio response.",
    )
