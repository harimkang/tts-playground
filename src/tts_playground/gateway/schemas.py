from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic.json_schema import SkipJsonSchema

from tts_playground.common.models import QWEN_CUSTOM_SPEAKERS, ModelID


class ValidationDetail(BaseModel):
    field: str = Field(
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z_][A-Za-z0-9_]{0,63}$",
    )
    reason: Literal["required", "invalid", "duplicate", "unknown"]


class ErrorBody(BaseModel):
    code: str
    message: str
    request_id: str
    details: (
        Annotated[list[ValidationDetail], Field(min_length=1, max_length=16)]
        | SkipJsonSchema[None]
    ) = Field(default_factory=lambda: None)


class ErrorEnvelope(BaseModel):
    error: ErrorBody


class SpeechRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: ModelID
    input: str = Field(min_length=1, max_length=2000)
    voice: str | None = Field(default=None, max_length=100)
    language: str = Field(min_length=2, max_length=10)
    instructions: str | None = Field(default=None, min_length=1, max_length=1000)
    response_format: Literal["wav"] = "wav"
    speed: float = Field(default=1.0, ge=0.7, le=2.0)
    steps: int | None = Field(default=None, ge=5, le=12)

    @model_validator(mode="after")
    def validate_for_model(self) -> "SpeechRequest":
        if self.model == ModelID.SUPERTONIC:
            valid_voices = {f"M{i}" for i in range(1, 6)} | {f"F{i}" for i in range(1, 6)}
            if self.voice not in valid_voices:
                raise ValueError("Supertonic voice must be M1-M5 or F1-F5")
            if self.instructions is not None:
                raise ValueError("Supertonic does not support instructions")
            return self

        if self.speed != 1.0 or self.steps is not None:
            raise ValueError("Qwen does not support speed or steps")
        if self.model == ModelID.QWEN_CUSTOM:
            if not self.voice:
                raise ValueError("CustomVoice requires voice")
            if self.voice not in QWEN_CUSTOM_SPEAKERS:
                raise ValueError("CustomVoice uses an unsupported CustomVoice speaker")
        if self.model == ModelID.QWEN_DESIGN:
            if not self.instructions or self.voice is not None:
                raise ValueError("VoiceDesign requires instructions and rejects voice")
        if self.model == ModelID.QWEN_CLONE and (
            not self.voice or not self.voice.startswith("voice_")
        ):
            raise ValueError("Voice clone requires a registered voice ID")
        if self.model == ModelID.QWEN_CLONE and self.instructions is not None:
            raise ValueError("Voice clone does not support instructions")
        return self
