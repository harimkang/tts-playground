from __future__ import annotations

import io
from pathlib import Path
from typing import Final, Protocol, cast

import soundfile as sf  # type: ignore[import-untyped]

from tts_playground.common.errors import ServiceError

MAX_AUDIO_BYTES: Final = 20 * 1024 * 1024
MIN_AUDIO_SECONDS: Final = 3.0
MAX_AUDIO_SECONDS: Final = 30.0
MAX_AUDIO_CHANNELS: Final = 2
MAX_AUDIO_SAMPLE_RATE: Final = 192_000
MAX_DECODED_SAMPLES: Final = 6_000_000
_SUPPORTED_AUDIO_FORMATS: Final = frozenset({"FLAC", "WAV"})


class _AudioMetadata(Protocol):
    format: str
    frames: int
    samplerate: int
    channels: int


def normalize_optional_transcript(value: str | None) -> str | None:
    if value is None or not value.strip():
        return None
    return value


def normalize_reference_audio(audio: bytes, destination: Path) -> float:
    if len(audio) > MAX_AUDIO_BYTES:
        raise _audio_too_large()
    audio_stream = io.BytesIO(audio)
    try:
        audio_info = sf.info(audio_stream)
    except (sf.LibsndfileError, RuntimeError, TypeError, ValueError):
        raise invalid_audio() from None
    duration_seconds = validate_audio_metadata(audio_info)
    try:
        audio_stream.seek(0)
        samples, sample_rate = sf.read(
            audio_stream,
            dtype="float32",
            always_2d=True,
        )
    except (sf.LibsndfileError, RuntimeError, TypeError, ValueError):
        raise invalid_audio() from None
    if sample_rate != audio_info.samplerate:
        raise invalid_audio()
    sf.write(destination, samples, sample_rate, format="WAV", subtype="PCM_16")
    return duration_seconds


def validate_audio_metadata(audio_info: object) -> float:
    metadata = cast(_AudioMetadata, audio_info)
    try:
        audio_format = metadata.format
        frames = metadata.frames
        sample_rate = metadata.samplerate
        channels = metadata.channels
    except (AttributeError, TypeError, ValueError):
        raise invalid_audio() from None
    if audio_format not in _SUPPORTED_AUDIO_FORMATS:
        raise invalid_audio()
    numeric_metadata = (frames, sample_rate, channels)
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in numeric_metadata
    ):
        raise invalid_audio()
    if frames <= 0 or sample_rate <= 0 or channels <= 0:
        raise invalid_audio()

    duration_seconds = frames / sample_rate
    if not MIN_AUDIO_SECONDS <= duration_seconds <= MAX_AUDIO_SECONDS:
        raise ServiceError(
            status_code=422,
            code="invalid_audio_duration",
            message="Reference audio must be between 3 and 30 seconds.",
        )
    if (
        channels > MAX_AUDIO_CHANNELS
        or sample_rate > MAX_AUDIO_SAMPLE_RATE
        or frames * channels > MAX_DECODED_SAMPLES
    ):
        raise invalid_audio()
    return duration_seconds


def invalid_audio() -> ServiceError:
    return ServiceError(
        status_code=400,
        code="invalid_audio",
        message="Reference audio is invalid.",
    )


def _audio_too_large() -> ServiceError:
    return ServiceError(
        status_code=413,
        code="audio_too_large",
        message="Reference audio exceeds the 20 MB limit.",
    )
