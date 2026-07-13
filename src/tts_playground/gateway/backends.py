from collections.abc import AsyncIterator
from dataclasses import dataclass
from math import isfinite
from typing import Final
from urllib.parse import quote

import httpx
from prometheus_client.parser import text_string_to_metric_families
from python_multipart import MultipartParser
from python_multipart.exceptions import MultipartParseError
from python_multipart.multipart import parse_options_header

from tts_playground.common.errors import ServiceError
from tts_playground.common.logging import (
    MAX_AUDIO_DURATION_SECONDS,
    MAX_TIMING_MILLISECONDS,
    normalize_bounded_nonnegative,
    normalize_error_code,
)
from tts_playground.common.models import QWEN_LANGUAGES, ModelID
from tts_playground.common.settings import GatewaySettings
from tts_playground.gateway.schemas import SpeechRequest

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

_MAX_MULTIPART_NON_AUDIO_BYTES: Final = 1024 * 1024
_MAX_MULTIPART_HEADER_BYTES: Final = 4224
_MAX_MULTIPART_HEADER_COUNT: Final = 8
_MAX_MULTIPART_BOUNDARY_BYTES: Final = 200
_MAX_INPUT_CHARS: Final = 2000
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
    input_chars: int | None = None


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

        return _audio_result(response)

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
        response = await self._post_bounded_multipart(
            path="/internal/v1/voices",
            content=content,
            content_type=content_type,
            content_length=content_length,
            request_id=request_id,
        )
        if not response.is_success:
            _raise_upstream_error(response)
        return ProxyResult(
            status_code=response.status_code,
            payload=_public_voice(_read_json(response)),
        )

    async def synthesize_ephemeral(
        self,
        *,
        content: AsyncIterator[bytes],
        content_type: str | None,
        content_length: int | None,
        request_id: str,
    ) -> AudioResult:
        response = await self._post_bounded_multipart(
            path="/internal/v1/speech/clone",
            content=content,
            content_type=content_type,
            content_length=content_length,
            request_id=request_id,
        )
        if not response.is_success:
            _raise_upstream_error(response)
        _validate_wav_response(response)
        return _audio_result(response, include_input_chars=True)

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
        if (
            response.status_code != 200
            or not _is_prometheus_text(response)
            or not _is_valid_qwen_metrics(response.content)
        ):
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

    async def _post_bounded_multipart(
        self,
        *,
        path: str,
        content: AsyncIterator[bytes],
        content_type: str | None,
        content_length: int | None,
        request_id: str,
    ) -> httpx.Response:
        maximum_request_bytes = (
            self._settings.max_upload_bytes + _MAX_MULTIPART_NON_AUDIO_BYTES
        )
        limiter = _MultipartAudioLimiter(
            content_type,
            maximum_audio_bytes=self._settings.max_upload_bytes,
            maximum_request_bytes=maximum_request_bytes,
        )
        if content_length is not None and content_length > maximum_request_bytes:
            raise _multipart_request_too_large()

        async def bounded_content() -> AsyncIterator[bytes]:
            async for chunk in content:
                limiter.write(chunk)
                yield chunk
            limiter.finalize()

        headers = {"Content-Type": limiter.content_type}
        if content_length is not None:
            headers["Content-Length"] = str(content_length)
        return await self._request(
            "POST",
            f"{self._settings.qwen_url.rstrip('/')}{path}",
            request_id=request_id,
            timeout_seconds=self._settings.qwen_timeout_seconds,
            headers=headers,
            content=bounded_content(),
        )

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
                backend_failure=True,
            ) from error
        if response.is_success:
            return
        raise ServiceError(
            status_code=503,
            code="backend_unavailable",
            message="Required backend is unavailable.",
            backend_failure=True,
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
                backend_failure=True,
            ) from error
        except httpx.RequestError as error:
            raise ServiceError(
                status_code=503,
                code="backend_unavailable",
                message="Backend is unavailable.",
                backend_failure=True,
            ) from error


class _MultipartAudioLimiter:
    def __init__(
        self,
        content_type: str | None,
        *,
        maximum_audio_bytes: int,
        maximum_request_bytes: int,
    ) -> None:
        try:
            media_type, options = parse_options_header(content_type)
        except (TypeError, ValueError):
            raise _invalid_multipart() from None
        boundary = options.get(b"boundary")
        if (
            media_type != b"multipart/form-data"
            or boundary is None
            or not 0 < len(boundary) <= _MAX_MULTIPART_BOUNDARY_BYTES
        ):
            raise _invalid_multipart()
        self.content_type = content_type or ""
        self._maximum_audio_bytes = maximum_audio_bytes
        self._maximum_request_bytes = maximum_request_bytes
        self._audio_bytes = 0
        self._request_bytes = 0
        self._header_field = bytearray()
        self._header_value = bytearray()
        self._content_disposition = b""
        self._is_audio = False
        self._finished = False
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
                "on_end": self._on_end,
            },
            max_header_count=_MAX_MULTIPART_HEADER_COUNT,
            max_header_size=_MAX_MULTIPART_HEADER_BYTES,
        )

    def write(self, chunk: bytes) -> None:
        self._request_bytes += len(chunk)
        if self._request_bytes > self._maximum_request_bytes:
            raise _multipart_request_too_large()
        try:
            self._parser.write(chunk)
        except MultipartParseError:
            raise _invalid_multipart() from None

    def finalize(self) -> None:
        try:
            self._parser.finalize()
        except MultipartParseError:
            raise _invalid_multipart() from None
        if not self._finished:
            raise _invalid_multipart()

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
        if self._audio_bytes > self._maximum_audio_bytes:
            raise _audio_too_large()

    def _on_part_end(self) -> None:
        self._is_audio = False

    def _on_end(self) -> None:
        self._finished = True


def _read_json(response: httpx.Response) -> object:
    try:
        return response.json()
    except ValueError:
        raise _invalid_backend_response() from None


def _audio_result(
    response: httpx.Response,
    *,
    include_input_chars: bool = False,
) -> AudioResult:
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
        input_chars=(
            _parse_bounded_input_chars(response.headers.get("x-input-chars"))
            if include_input_chars
            else None
        ),
    )


def _parse_bounded_input_chars(value: str | None) -> int | None:
    if (
        value is None
        or not value
        or len(value) > len(str(_MAX_INPUT_CHARS))
        or not value.isascii()
        or not value.isdecimal()
    ):
        return None
    parsed = int(value)
    return parsed if parsed <= _MAX_INPUT_CHARS else None


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
    if not all(
        isinstance(payload[field], str)
        for field in ("id", "name", "language", "created_at")
    ):
        raise _invalid_backend_response()
    if type(payload["consent_confirmed"]) is not bool:
        raise _invalid_backend_response()
    duration = payload["duration_seconds"]
    if type(duration) not in (int, float):
        raise _invalid_backend_response()
    try:
        normalized_duration = float(duration)
    except OverflowError:
        raise _invalid_backend_response() from None
    if not isfinite(normalized_duration) or normalized_duration < 0:
        raise _invalid_backend_response()
    return {field: payload[field] for field in _PUBLIC_VOICE_FIELDS}


def _is_prometheus_text(response: httpx.Response) -> bool:
    media_type = response.headers.get("content-type", "").partition(";")[0].strip().lower()
    return media_type in {"text/plain", "application/openmetrics-text"}


def _is_valid_qwen_metrics(content: bytes) -> bool:
    try:
        text = content.decode("utf-8")
        families = list(text_string_to_metric_families(text))
    except (UnicodeDecodeError, ValueError):
        return False
    if not families:
        return False
    has_sample = False
    for family in families:
        if not family.name.startswith("tts_qwen_"):
            return False
        for sample in family.samples:
            has_sample = True
            if not sample.name.startswith("tts_qwen_"):
                return False
    return has_sample


def _audio_too_large() -> ServiceError:
    return ServiceError(
        status_code=413,
        code="audio_too_large",
        message="Reference audio exceeds the 20 MB limit.",
    )


def _multipart_request_too_large() -> ServiceError:
    return ServiceError(
        status_code=413,
        code="request_too_large",
        message="Multipart request exceeds the size limit.",
    )


def _invalid_multipart() -> ServiceError:
    return ServiceError(
        status_code=422,
        code="invalid_multipart",
        message="Multipart request is invalid.",
    )


def _invalid_backend_response() -> ServiceError:
    return ServiceError(
        status_code=503,
        code="backend_invalid_response",
        message="Backend returned an invalid response.",
        backend_failure=True,
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
            if isinstance(code, str):
                raise ServiceError(
                    status_code=response.status_code,
                    code=normalize_error_code(code) or "backend_error",
                    message="Backend request failed.",
                    backend_failure=True,
                )

    raise ServiceError(
        status_code=response.status_code,
        code="backend_error",
        message="Backend request failed.",
        backend_failure=True,
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
        backend_failure=True,
    )
