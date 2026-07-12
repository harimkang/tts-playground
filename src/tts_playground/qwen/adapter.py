from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from tts_playground.common.models import ModelID


class CudaOutOfMemory(RuntimeError):
    """Raised when a Qwen adapter operation exhausts CUDA memory."""


@dataclass(slots=True)
class InternalSpeechRequest:
    model: ModelID
    text: str
    language: str
    voice: str | None
    instructions: str | None


@dataclass(slots=True)
class SynthesisResult:
    audio: bytes
    sample_rate: int
    duration_seconds: float


class QwenAdapter(Protocol):
    def free_vram_bytes(self) -> int:
        raise NotImplementedError

    def load(self, repository: str, attention_implementation: str) -> object:
        raise NotImplementedError

    def cleanup_device_memory(self) -> None:
        raise NotImplementedError

    def build_clone_prompt(
        self,
        model: object,
        audio_path: Path,
        transcript: str,
    ) -> object:
        raise NotImplementedError

    def synthesize(
        self,
        model: object,
        request: InternalSpeechRequest,
        clone_prompt: object | None,
    ) -> SynthesisResult:
        raise NotImplementedError
