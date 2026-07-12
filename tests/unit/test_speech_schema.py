import pytest
from pydantic import ValidationError

from tts_playground.gateway.schemas import SpeechRequest
from tts_playground.qwen.schemas import QwenSpeechRequest


def test_supertonic_accepts_speed_and_steps() -> None:
    request = SpeechRequest(
        model="supertonic-3",
        input="안녕하세요",
        voice="F1",
        language="ko",
        speed=1.1,
        steps=10,
    )

    assert request.steps == 10


@pytest.mark.parametrize(
    "model",
    [
        "qwen3-tts-1.7b-custom",
        "qwen3-tts-1.7b-design",
        "qwen3-tts-1.7b-clone",
    ],
)
def test_qwen_rejects_steps(model: str) -> None:
    with pytest.raises(ValidationError, match="Qwen does not support speed or steps"):
        SpeechRequest(
            model=model,
            input="테스트",
            voice="Sohee",
            language="ko",
            instructions="차분하게",
            steps=8,
        )


def test_design_requires_instructions_and_rejects_voice() -> None:
    with pytest.raises(
        ValidationError,
        match="VoiceDesign requires instructions and rejects voice",
    ):
        SpeechRequest(
            model="qwen3-tts-1.7b-design",
            input="테스트",
            voice="Sohee",
            language="ko",
        )


@pytest.mark.parametrize("voice", [None, "M0", "M6", "F0", "F6", "Sohee"])
def test_supertonic_rejects_unsupported_voice(voice: str | None) -> None:
    with pytest.raises(ValidationError, match="Supertonic voice must be M1-M5 or F1-F5"):
        SpeechRequest(
            model="supertonic-3",
            input="test",
            voice=voice,
            language="en",
        )


def test_supertonic_rejects_instructions() -> None:
    with pytest.raises(ValidationError, match="Supertonic does not support instructions"):
        SpeechRequest(
            model="supertonic-3",
            input="test",
            voice="M1",
            language="en",
            instructions="calm",
        )


def test_custom_voice_requires_voice() -> None:
    with pytest.raises(ValidationError, match="CustomVoice requires voice"):
        SpeechRequest(
            model="qwen3-tts-1.7b-custom",
            input="test",
            language="en",
        )


@pytest.mark.parametrize("schema", [SpeechRequest, QwenSpeechRequest])
def test_custom_voice_rejects_unknown_speaker(
    schema: type[SpeechRequest] | type[QwenSpeechRequest],
) -> None:
    values = {
        "model": "qwen3-tts-1.7b-custom",
        "voice": "UnknownSpeaker",
        "language": "en",
    }
    values["input" if schema is SpeechRequest else "text"] = "test"

    with pytest.raises(ValidationError, match="unsupported CustomVoice speaker"):
        schema.model_validate(values)


def test_voice_design_accepts_instructions_without_voice() -> None:
    request = SpeechRequest(
        model="qwen3-tts-1.7b-design",
        input="test",
        language="en",
        instructions="calm",
    )

    assert request.instructions == "calm"
    assert request.voice is None


@pytest.mark.parametrize("voice", [None, "Sohee", "voice"])
def test_voice_clone_requires_registered_voice_id(voice: str | None) -> None:
    with pytest.raises(ValidationError, match="Voice clone requires a registered voice ID"):
        SpeechRequest(
            model="qwen3-tts-1.7b-clone",
            input="test",
            voice=voice,
            language="en",
        )


def test_voice_clone_accepts_registered_voice_id() -> None:
    request = SpeechRequest(
        model="qwen3-tts-1.7b-clone",
        input="test",
        voice="voice_example",
        language="en",
    )

    assert request.voice == "voice_example"


@pytest.mark.parametrize("schema", [SpeechRequest, QwenSpeechRequest])
def test_voice_clone_rejects_instructions(
    schema: type[SpeechRequest] | type[QwenSpeechRequest],
) -> None:
    values = {
        "model": "qwen3-tts-1.7b-clone",
        "voice": "voice_example",
        "language": "en",
        "instructions": "must not be silently ignored",
    }
    values["input" if schema is SpeechRequest else "text"] = "test"

    with pytest.raises(ValidationError, match="Voice clone does not support instructions"):
        schema.model_validate(values)


def test_qwen_rejects_non_default_speed() -> None:
    with pytest.raises(ValidationError, match="Qwen does not support speed or steps"):
        SpeechRequest(
            model="qwen3-tts-1.7b-custom",
            input="test",
            voice="Sohee",
            language="en",
            speed=1.1,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("input", ""),
        ("input", "x" * 2001),
        ("voice", "x" * 101),
        ("language", "e"),
        ("language", "x" * 11),
        ("instructions", ""),
        ("instructions", "x" * 1001),
        ("response_format", "mp3"),
        ("speed", 0.69),
        ("speed", 2.01),
        ("steps", 4),
        ("steps", 13),
    ],
)
def test_field_limits_are_enforced(field: str, value: object) -> None:
    values: dict[str, object] = {
        "model": "supertonic-3",
        "input": "test",
        "voice": "F1",
        "language": "en",
    }
    values[field] = value

    with pytest.raises(ValidationError):
        SpeechRequest.model_validate(values)


def test_unknown_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        SpeechRequest(
            model="supertonic-3",
            input="test",
            voice="F1",
            language="en",
            unknown="value",  # type: ignore[call-arg]
        )
