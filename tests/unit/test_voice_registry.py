import io
import json
import os
import re
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from tts_playground.common.errors import ServiceError
from tts_playground.qwen.voice_registry import VoiceRegistry

MAX_AUDIO_BYTES = 20 * 1024 * 1024


def _wav_bytes(duration_seconds: float, sample_rate: int = 8_000) -> bytes:
    audio = np.zeros(round(duration_seconds * sample_rate), dtype=np.float32)
    output = io.BytesIO()
    sf.write(output, audio, sample_rate, format="WAV", subtype="FLOAT")
    return output.getvalue()


@pytest.fixture
def wav_bytes() -> bytes:
    return _wav_bytes(4)


def test_registry_persists_normalized_audio_without_exposing_secrets(
    tmp_path: Path,
    wav_bytes: bytes,
) -> None:
    registry = VoiceRegistry(tmp_path)

    created = registry.register(
        name="internal speaker",
        audio=wav_bytes,
        transcript="정확한 문장",
        language="ko",
        consent_confirmed=True,
    )

    assert re.fullmatch(r"voice_[0-9a-f]{32}", created.id)
    assert set(asdict(created)) == {
        "id",
        "name",
        "language",
        "duration_seconds",
        "consent_confirmed",
        "created_at",
    }
    assert not hasattr(created, "transcript")
    assert not hasattr(created, "audio_path")
    assert created.duration_seconds == pytest.approx(4.0)
    assert created.consent_confirmed is True
    assert datetime.fromisoformat(created.created_at).utcoffset() is not None

    voice_directory = tmp_path / created.id
    metadata_path = voice_directory / "metadata.json"
    audio_path = voice_directory / "reference.wav"
    assert sorted(path.name for path in voice_directory.iterdir()) == [
        "metadata.json",
        "reference.wav",
    ]
    assert sf.info(audio_path).format == "WAV"
    assert sf.info(audio_path).subtype == "PCM_16"

    metadata = json.loads(metadata_path.read_text())
    assert metadata == {
        "id": created.id,
        "name": "internal speaker",
        "language": "ko",
        "duration_seconds": 4.0,
        "transcript": "정확한 문장",
        "consent_confirmed": True,
        "created_at": created.created_at,
    }

    reloaded = VoiceRegistry(tmp_path)
    loaded_internal = reloaded.get_internal(created.id)
    assert loaded_internal.transcript == "정확한 문장"
    assert loaded_internal.audio_path == audio_path
    assert reloaded.get_public(created.id) == created
    assert reloaded.list_public() == [created]


def test_delete_removes_only_the_opaque_voice_directory(
    tmp_path: Path,
    wav_bytes: bytes,
) -> None:
    registry = VoiceRegistry(tmp_path)
    created = registry.register(
        name="../../client-name-must-not-be-a-path",
        audio=wav_bytes,
        transcript="exact transcript",
        language="en",
        consent_confirmed=True,
    )
    voice_directory = tmp_path / created.id

    registry.delete(created.id)

    assert not voice_directory.exists()
    assert list(tmp_path.iterdir()) == []
    with pytest.raises(ServiceError) as error:
        registry.get_public(created.id)
    assert (error.value.status_code, error.value.code) == (404, "voice_not_found")


def test_corrupt_audio_has_a_stable_error_without_decoder_details(tmp_path: Path) -> None:
    with pytest.raises(ServiceError) as error:
        VoiceRegistry(tmp_path).register(
            name="corrupt",
            audio=b"definitely not audio data",
            transcript="exact transcript",
            language="en",
            consent_confirmed=True,
        )

    assert error.value.status_code == 400
    assert error.value.code == "invalid_audio"
    assert error.value.message == "Reference audio is invalid."


@pytest.mark.parametrize("duration_seconds", [2, 31])
def test_out_of_range_audio_duration_is_rejected(
    tmp_path: Path,
    duration_seconds: int,
) -> None:
    with pytest.raises(ServiceError) as error:
        VoiceRegistry(tmp_path).register(
            name="bad duration",
            audio=_wav_bytes(duration_seconds),
            transcript="exact transcript",
            language="en",
            consent_confirmed=True,
        )

    assert error.value.status_code == 422
    assert error.value.code == "invalid_audio_duration"


def test_oversized_audio_is_rejected_before_decode(tmp_path: Path) -> None:
    with pytest.raises(ServiceError) as error:
        VoiceRegistry(tmp_path).register(
            name="too large",
            audio=b"not decoded".ljust(MAX_AUDIO_BYTES + 1, b"x"),
            transcript="exact transcript",
            language="en",
            consent_confirmed=True,
        )

    assert error.value.status_code == 413
    assert error.value.code == "audio_too_large"


def test_registration_requires_consent(tmp_path: Path, wav_bytes: bytes) -> None:
    with pytest.raises(ServiceError) as error:
        VoiceRegistry(tmp_path).register(
            name="no consent",
            audio=wav_bytes,
            transcript="exact transcript",
            language="en",
            consent_confirmed=False,
        )

    assert error.value.status_code == 422
    assert error.value.code == "consent_required"
    assert list(tmp_path.iterdir()) == []


def test_registry_rejects_symlinked_voice_directories_and_path_escapes(
    tmp_path: Path,
) -> None:
    registry_root = tmp_path / "registry"
    registry_root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    voice_id = "voice_0123456789abcdef0123456789abcdef"
    (outside / "reference.wav").write_bytes(_wav_bytes(4))
    (outside / "metadata.json").write_text(
        json.dumps(
            {
                "id": voice_id,
                "name": "outside",
                "language": "en",
                "duration_seconds": 4.0,
                "transcript": "must stay private",
                "consent_confirmed": True,
                "created_at": "2026-07-10T00:00:00+00:00",
            }
        )
    )
    os.symlink(outside, registry_root / voice_id, target_is_directory=True)

    registry = VoiceRegistry(registry_root)

    assert registry.list_public() == []
    with pytest.raises(ServiceError) as error:
        registry.delete(voice_id)
    assert (error.value.status_code, error.value.code) == (404, "voice_not_found")
    assert (outside / "metadata.json").exists()
