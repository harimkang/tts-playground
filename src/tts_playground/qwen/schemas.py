from pydantic import BaseModel, ConfigDict, Field, model_validator

from tts_playground.common.models import QWEN_CUSTOM_SPEAKERS, ModelID
from tts_playground.qwen.adapter import InternalSpeechRequest


class QwenSpeechRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: ModelID
    text: str = Field(min_length=1, max_length=2000)
    language: str = Field(min_length=2, max_length=32)
    voice: str | None = Field(default=None, max_length=100)
    instructions: str | None = Field(default=None, min_length=1, max_length=1000)

    @model_validator(mode="after")
    def validate_for_model(self) -> "QwenSpeechRequest":
        if self.model == ModelID.SUPERTONIC:
            raise ValueError("The Qwen worker only accepts Qwen models")
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

    def to_internal(self) -> InternalSpeechRequest:
        return InternalSpeechRequest(
            model=self.model,
            text=self.text,
            language=self.language,
            voice=self.voice,
            instructions=self.instructions,
        )


class WorkerStatusResponse(BaseModel):
    state: str
    loaded_model: ModelID | None
    queue_depth: int
    last_error_code: str | None


class VoiceResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    language: str
    duration_seconds: float
    consent_confirmed: bool
    created_at: str
