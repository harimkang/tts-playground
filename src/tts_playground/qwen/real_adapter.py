from __future__ import annotations

import gc
import importlib
import io
from collections.abc import Callable, Sized
from pathlib import Path
from typing import Protocol, cast

import soundfile as sf  # type: ignore[import-untyped]

from tts_playground.common.models import ModelID
from tts_playground.qwen.adapter import (
    CudaOutOfMemory,
    InternalSpeechRequest,
    SynthesisResult,
)


class _CudaModule(Protocol):
    OutOfMemoryError: type[Exception]

    def mem_get_info(self) -> tuple[int, int]: ...

    def empty_cache(self) -> None: ...

    def ipc_collect(self) -> None: ...


class _TorchModule(Protocol):
    cuda: _CudaModule
    bfloat16: object


class _ModelClass(Protocol):
    def from_pretrained(
        self,
        repository: str,
        *,
        device_map: str,
        dtype: object,
        attn_implementation: str,
    ) -> object: ...


class _QwenModel(Protocol):
    def generate_custom_voice(
        self,
        *,
        text: str,
        language: str,
        speaker: str,
        instruct: str | None,
    ) -> tuple[list[object], int]: ...

    def generate_voice_design(
        self,
        *,
        text: str,
        language: str,
        instruct: str,
    ) -> tuple[list[object], int]: ...

    def create_voice_clone_prompt(
        self,
        *,
        ref_audio: str,
        ref_text: str,
        x_vector_only_mode: bool,
    ) -> object: ...

    def generate_voice_clone(
        self,
        *,
        text: str,
        language: str,
        voice_clone_prompt: object,
    ) -> tuple[list[object], int]: ...


class QwenTTSAdapter:
    """Lazy adapter around the optional qwen-tts and torch packages."""

    def __init__(
        self,
        *,
        torch_module: _TorchModule | None = None,
        model_class: _ModelClass | None = None,
    ) -> None:
        self._torch_module = torch_module
        self._model_class = model_class

    def free_vram_bytes(self) -> int:
        torch_module = self._torch()
        free_bytes, _total_bytes = self._with_oom_conversion(
            torch_module.cuda.mem_get_info
        )
        return free_bytes

    def load(self, repository: str, attention_implementation: str) -> object:
        torch_module = self._torch()
        model_class = self._qwen_model_class()
        return self._with_oom_conversion(
            model_class.from_pretrained,
            repository,
            device_map="cuda:0",
            dtype=torch_module.bfloat16,
            attn_implementation=attention_implementation,
        )

    def unload(self, model: object) -> None:
        torch_module = self._torch()
        del model
        gc.collect()
        self._with_oom_conversion(torch_module.cuda.empty_cache)
        self._with_oom_conversion(torch_module.cuda.ipc_collect)

    def build_clone_prompt(
        self,
        model: object,
        audio_path: Path,
        transcript: str,
    ) -> object:
        qwen_model = cast(_QwenModel, model)
        return self._with_oom_conversion(
            qwen_model.create_voice_clone_prompt,
            ref_audio=str(audio_path),
            ref_text=transcript,
            x_vector_only_mode=False,
        )

    def synthesize(
        self,
        model: object,
        request: InternalSpeechRequest,
        clone_prompt: object | None,
    ) -> SynthesisResult:
        qwen_model = cast(_QwenModel, model)
        if request.model == ModelID.QWEN_CUSTOM:
            if request.voice is None:
                raise ValueError("CustomVoice synthesis requires a speaker")
            generated = self._with_oom_conversion(
                qwen_model.generate_custom_voice,
                text=request.text,
                language=request.language,
                speaker=request.voice,
                instruct=request.instructions,
            )
        elif request.model == ModelID.QWEN_DESIGN:
            if request.instructions is None:
                raise ValueError("VoiceDesign synthesis requires instructions")
            generated = self._with_oom_conversion(
                qwen_model.generate_voice_design,
                text=request.text,
                language=request.language,
                instruct=request.instructions,
            )
        elif request.model == ModelID.QWEN_CLONE:
            if clone_prompt is None:
                raise ValueError("Voice clone synthesis requires a reusable prompt")
            generated = self._with_oom_conversion(
                qwen_model.generate_voice_clone,
                text=request.text,
                language=request.language,
                voice_clone_prompt=clone_prompt,
            )
        else:
            raise ValueError(f"Unsupported Qwen model: {request.model.value}")
        return _encode_generation(generated)

    def _torch(self) -> _TorchModule:
        if self._torch_module is None:
            self._torch_module = cast(
                _TorchModule,
                importlib.import_module("torch"),
            )
        return self._torch_module

    def _qwen_model_class(self) -> _ModelClass:
        if self._model_class is None:
            qwen_tts = importlib.import_module("qwen_tts")
            self._model_class = cast(_ModelClass, qwen_tts.Qwen3TTSModel)
        return self._model_class

    def _with_oom_conversion[ResultT, **Parameters](
        self,
        function: Callable[Parameters, ResultT],
        /,
        *args: Parameters.args,
        **kwargs: Parameters.kwargs,
    ) -> ResultT:
        try:
            return function(*args, **kwargs)
        except Exception as error:
            torch_module = self._torch()
            if isinstance(error, torch_module.cuda.OutOfMemoryError):
                raise CudaOutOfMemory from error
            raise


def _encode_generation(generated: tuple[list[object], int]) -> SynthesisResult:
    waveforms, sample_rate = generated
    if not waveforms:
        raise ValueError("Qwen returned no generated waveform")
    waveform = waveforms[0]
    sized_waveform = cast(Sized, waveform)
    output = io.BytesIO()
    sf.write(
        output,
        waveform,
        sample_rate,
        format="WAV",
        subtype="PCM_16",
    )
    return SynthesisResult(
        audio=output.getvalue(),
        sample_rate=sample_rate,
        duration_seconds=len(sized_waveform) / sample_rate,
    )
