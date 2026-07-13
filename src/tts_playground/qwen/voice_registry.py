from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import threading
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from math import isfinite
from pathlib import Path
from typing import Final

from tts_playground.common.errors import ServiceError
from tts_playground.qwen.reference_audio import (
    MAX_AUDIO_BYTES as MAX_AUDIO_BYTES,
)
from tts_playground.qwen.reference_audio import (
    MAX_AUDIO_CHANNELS as MAX_AUDIO_CHANNELS,
)
from tts_playground.qwen.reference_audio import (
    MAX_AUDIO_SAMPLE_RATE as MAX_AUDIO_SAMPLE_RATE,
)
from tts_playground.qwen.reference_audio import (
    MAX_AUDIO_SECONDS as MAX_AUDIO_SECONDS,
)
from tts_playground.qwen.reference_audio import (
    MAX_DECODED_SAMPLES as MAX_DECODED_SAMPLES,
)
from tts_playground.qwen.reference_audio import (
    MIN_AUDIO_SECONDS as MIN_AUDIO_SECONDS,
)
from tts_playground.qwen.reference_audio import (
    normalize_optional_transcript,
    normalize_reference_audio,
)

_VOICE_ID_PATTERN: Final = re.compile(r"voice_[0-9a-f]{32}")


@dataclass(frozen=True, slots=True)
class VoicePublic:
    id: str
    name: str
    language: str
    duration_seconds: float
    consent_confirmed: bool
    created_at: str


@dataclass(frozen=True, slots=True)
class VoiceInternal:
    id: str
    name: str
    language: str
    duration_seconds: float
    consent_confirmed: bool
    created_at: str
    transcript: str | None
    audio_path: Path


class VoiceRegistry:
    def __init__(self, root: Path | str) -> None:
        configured_root = Path(root)
        configured_root.mkdir(parents=True, exist_ok=True)
        self._root = configured_root.resolve(strict=True)
        self._lock = threading.RLock()
        self._voices = self._load_voices()

    def register(
        self,
        *,
        name: str,
        audio: bytes,
        transcript: str | None,
        language: str,
        consent_confirmed: bool,
    ) -> VoicePublic:
        if not consent_confirmed:
            raise ServiceError(
                status_code=422,
                code="consent_required",
                message="Consent confirmation is required.",
            )
        transcript = normalize_optional_transcript(transcript)

        voice_id = f"voice_{uuid.uuid4().hex}"
        created_at = datetime.now(UTC).isoformat()
        target_directory = self._root / voice_id
        staging_directory = Path(tempfile.mkdtemp(prefix=".voice-", dir=self._root))
        published = False
        try:
            audio_path = staging_directory / "reference.wav"
            duration_seconds = normalize_reference_audio(audio, audio_path)
            _fsync_file(audio_path)

            metadata = {
                "id": voice_id,
                "name": name,
                "language": language,
                "duration_seconds": duration_seconds,
                "transcript": transcript,
                "consent_confirmed": consent_confirmed,
                "created_at": created_at,
            }
            metadata_path = staging_directory / "metadata.json"
            with metadata_path.open("w", encoding="utf-8") as metadata_file:
                json.dump(metadata, metadata_file, ensure_ascii=False)
                metadata_file.flush()
                os.fsync(metadata_file.fileno())

            _fsync_directory(staging_directory)
            staging_directory.rename(target_directory)
            published = True
            _fsync_directory(self._root)
        except BaseException:
            cleanup_directory = target_directory if published else staging_directory
            if cleanup_directory.exists() and not cleanup_directory.is_symlink():
                shutil.rmtree(cleanup_directory)
            raise

        internal = VoiceInternal(
            id=voice_id,
            name=name,
            language=language,
            duration_seconds=duration_seconds,
            consent_confirmed=consent_confirmed,
            created_at=created_at,
            transcript=transcript,
            audio_path=target_directory / "reference.wav",
        )
        with self._lock:
            self._voices[voice_id] = internal
        return _to_public(internal)

    def list_public(self) -> list[VoicePublic]:
        with self._lock:
            return [_to_public(self._voices[voice_id]) for voice_id in sorted(self._voices)]

    def get_public(self, voice_id: str) -> VoicePublic:
        return _to_public(self.get_internal(voice_id))

    def get_internal(self, voice_id: str) -> VoiceInternal:
        with self._lock:
            try:
                return self._voices[voice_id]
            except KeyError:
                raise _voice_not_found() from None

    def delete(self, voice_id: str) -> None:
        with self._lock:
            if voice_id not in self._voices:
                raise _voice_not_found()
            voice_directory = self._safe_voice_directory(self._root / voice_id)
            if voice_directory is None:
                del self._voices[voice_id]
                raise _voice_not_found()
            shutil.rmtree(voice_directory)
            _fsync_directory(self._root)
            del self._voices[voice_id]

    def _load_voices(self) -> dict[str, VoiceInternal]:
        voices: dict[str, VoiceInternal] = {}
        for candidate in self._root.iterdir():
            if _VOICE_ID_PATTERN.fullmatch(candidate.name) is None:
                continue
            voice_directory = self._safe_voice_directory(candidate)
            if voice_directory is None:
                continue
            voice = _load_voice(voice_directory)
            if voice is not None:
                voices[voice.id] = voice
        return voices

    def _safe_voice_directory(self, candidate: Path) -> Path | None:
        if candidate.is_symlink() or not candidate.is_dir():
            return None
        try:
            resolved = candidate.resolve(strict=True)
        except OSError:
            return None
        if resolved.parent != self._root:
            return None
        return resolved


def _load_voice(voice_directory: Path) -> VoiceInternal | None:
    metadata_path = voice_directory / "metadata.json"
    audio_path = voice_directory / "reference.wav"
    if not _is_safe_file(metadata_path, voice_directory):
        return None
    if not _is_safe_file(audio_path, voice_directory):
        return None

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        voice_id = metadata["id"]
        name = metadata["name"]
        language = metadata["language"]
        duration_seconds = metadata["duration_seconds"]
        transcript = metadata["transcript"]
        consent_confirmed = metadata["consent_confirmed"]
        created_at = metadata["created_at"]
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return None

    if voice_id != voice_directory.name or not isinstance(voice_id, str):
        return None
    if not all(isinstance(value, str) for value in (name, language, created_at)):
        return None
    if transcript is not None and not isinstance(transcript, str):
        return None
    transcript = normalize_optional_transcript(transcript)
    if isinstance(duration_seconds, bool) or not isinstance(duration_seconds, int | float):
        return None
    try:
        duration_seconds = float(duration_seconds)
    except OverflowError:
        return None
    if not isfinite(duration_seconds) or not (
        MIN_AUDIO_SECONDS <= duration_seconds <= MAX_AUDIO_SECONDS
    ):
        return None
    if consent_confirmed is not True:
        return None
    try:
        parsed_created_at = datetime.fromisoformat(created_at)
    except ValueError:
        return None
    if parsed_created_at.utcoffset() is None:
        return None

    return VoiceInternal(
        id=voice_id,
        name=name,
        language=language,
        duration_seconds=duration_seconds,
        consent_confirmed=consent_confirmed,
        created_at=created_at,
        transcript=transcript,
        audio_path=audio_path,
    )


def _is_safe_file(path: Path, expected_parent: Path) -> bool:
    if path.is_symlink() or not path.is_file():
        return False
    try:
        return path.resolve(strict=True).parent == expected_parent
    except OSError:
        return False


def _to_public(voice: VoiceInternal) -> VoicePublic:
    return VoicePublic(
        id=voice.id,
        name=voice.name,
        language=voice.language,
        duration_seconds=voice.duration_seconds,
        consent_confirmed=voice.consent_confirmed,
        created_at=voice.created_at,
    )


def _voice_not_found() -> ServiceError:
    return ServiceError(
        status_code=404,
        code="voice_not_found",
        message="Voice not found.",
    )


def _fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
