from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

from tts_playground.common.errors import ServiceError
from tts_playground.common.models import QWEN_REPOSITORIES, ModelID
from tts_playground.common.settings import QwenSettings
from tts_playground.qwen.adapter import (
    CudaOutOfMemory,
    InternalSpeechRequest,
    QwenAdapter,
    SynthesisResult,
)
from tts_playground.qwen.gate import BoundedInferenceGate
from tts_playground.qwen.voice_registry import VoiceRegistry


class WorkerState(StrEnum):
    UNLOADED = "unloaded"
    LOADING = "loading"
    READY = "ready"
    SWITCHING = "switching"
    ERROR = "error"


@dataclass(slots=True)
class WorkerStatus:
    state: WorkerState
    loaded_model: ModelID | None
    queue_depth: int
    last_error_code: str | None


class QwenModelManager:
    def __init__(
        self,
        *,
        adapter: QwenAdapter,
        registry: VoiceRegistry,
        settings: QwenSettings,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._adapter = adapter
        self._registry = registry
        self._settings = settings
        self._monotonic = monotonic
        self._gate = BoundedInferenceGate(
            max_waiting=settings.max_waiting,
            queue_timeout_seconds=settings.queue_timeout_seconds,
        )
        self._state = WorkerState.UNLOADED
        self._model: object | None = None
        self._loaded_model: ModelID | None = None
        self._clone_prompts: dict[str, object] = {}
        self._last_used_at: float | None = None
        self._error_started_at: float | None = None
        self._last_error_code: str | None = None

    async def synthesize(self, request: InternalSpeechRequest) -> SynthesisResult:
        async with self._gate.slot():
            checkpoint = _CancellationCheckpoint()
            return await self._synthesize_locked(request, checkpoint)

    async def unload_if_idle(self) -> None:
        if not self._is_idle_unload_due():
            return
        async with self._gate.slot():
            checkpoint = _CancellationCheckpoint()
            await self._unload_if_idle_locked(checkpoint)

    def invalidate_voice(self, voice_id: str) -> None:
        self._clone_prompts.pop(voice_id, None)

    def status(self) -> WorkerStatus:
        return WorkerStatus(
            state=self._state,
            loaded_model=self._loaded_model,
            queue_depth=self._gate.waiting_count,
            last_error_code=self._last_error_code,
        )

    def in_error_cooldown(self) -> bool:
        if self._state != WorkerState.ERROR:
            return False
        if self._error_started_at is None:
            raise AssertionError("ERROR state requires a cooldown start time")
        return (
            self._monotonic() - self._error_started_at
            < self._settings.error_cooldown_seconds
        )

    async def shutdown(self) -> None:
        async with self._gate.slot():
            checkpoint = _CancellationCheckpoint()
            await self._unload_model(checkpoint)

    async def _synthesize_locked(
        self,
        request: InternalSpeechRequest,
        checkpoint: _CancellationCheckpoint,
    ) -> SynthesisResult:
        self._recover_from_error_or_raise()
        for attempt in range(2):
            cuda_oom = False
            model: object | None = None
            clone_prompt: object | None = None
            try:
                model = await self._ensure_model(request.model, checkpoint)
                self._touch_resident_model()
                clone_prompt = await self._clone_prompt(model, request, checkpoint)
                result = await checkpoint.run(
                    self._adapter.synthesize,
                    model,
                    request,
                    clone_prompt,
                )
            except CudaOutOfMemory:
                cuda_oom = True
            except asyncio.CancelledError:
                self._touch_resident_model()
                raise
            except Exception:
                self._touch_resident_model()
                checkpoint.raise_if_requested()
                raise

            if cuda_oom:
                model = None
                clone_prompt = None
                try:
                    await self._unload_model(checkpoint, cleanup_if_unloaded=True)
                except asyncio.CancelledError:
                    if attempt == 1:
                        self._enter_error("cuda_out_of_memory")
                    raise
                if attempt == 0:
                    continue
                self._enter_error("cuda_out_of_memory")
                raise ServiceError(
                    status_code=503,
                    code="cuda_out_of_memory",
                    message="Qwen ran out of GPU memory.",
                ) from None

            self._last_used_at = self._monotonic()
            self._last_error_code = None
            checkpoint.raise_if_requested()
            return result

        raise AssertionError("Qwen synthesis retry loop did not return or raise")

    async def _unload_if_idle_locked(self, checkpoint: _CancellationCheckpoint) -> None:
        if self._is_idle_unload_due():
            await self._unload_model(checkpoint)

    def _recover_from_error_or_raise(self) -> None:
        if self._state != WorkerState.ERROR:
            return
        if self._error_started_at is None:
            raise AssertionError("ERROR state requires a cooldown start time")
        elapsed = self._monotonic() - self._error_started_at
        if elapsed < self._settings.error_cooldown_seconds:
            raise ServiceError(
                status_code=503,
                code="model_unavailable",
                message="Qwen model is recovering from an error.",
            )
        self._state = WorkerState.UNLOADED
        self._error_started_at = None
        self._last_error_code = None

    async def _ensure_model(
        self,
        requested_model: ModelID,
        checkpoint: _CancellationCheckpoint,
    ) -> object:
        if self._model is not None and self._loaded_model == requested_model:
            return self._model

        if self._model is not None:
            self._state = WorkerState.SWITCHING
            await self._unload_model(checkpoint)

        self._state = WorkerState.LOADING
        try:
            free_vram = await checkpoint.run(self._adapter.free_vram_bytes)
        except CudaOutOfMemory:
            self._state = WorkerState.UNLOADED
            raise
        except Exception:
            self._enter_error("model_load_failed")
            checkpoint.raise_if_requested()
            raise ServiceError(
                status_code=503,
                code="model_load_failed",
                message="Qwen model could not be loaded.",
            ) from None

        if checkpoint.requested:
            self._state = WorkerState.UNLOADED
            checkpoint.raise_if_requested()

        if free_vram < self._settings.min_free_vram_bytes:
            self._state = WorkerState.UNLOADED
            self._last_error_code = "insufficient_vram"
            raise ServiceError(
                status_code=503,
                code="insufficient_vram",
                message="Insufficient free VRAM to load the Qwen model.",
            )

        repository = QWEN_REPOSITORIES[requested_model]
        try:
            model = await checkpoint.run(
                self._adapter.load,
                repository,
                self._settings.attention_implementation,
            )
        except CudaOutOfMemory:
            self._state = WorkerState.UNLOADED
            raise
        except Exception:
            self._enter_error("model_load_failed")
            checkpoint.raise_if_requested()
            raise ServiceError(
                status_code=503,
                code="model_load_failed",
                message="Qwen model could not be loaded.",
            ) from None

        self._model = model
        self._loaded_model = requested_model
        self._last_used_at = self._monotonic()
        self._state = WorkerState.READY
        checkpoint.raise_if_requested()
        return model

    async def _clone_prompt(
        self,
        model: object,
        request: InternalSpeechRequest,
        checkpoint: _CancellationCheckpoint,
    ) -> object | None:
        if request.model != ModelID.QWEN_CLONE:
            return None
        if request.voice is None:
            raise ServiceError(
                status_code=422,
                code="voice_required",
                message="A registered voice is required for voice cloning.",
            )
        cached = self._clone_prompts.get(request.voice)
        if cached is not None:
            return cached
        voice = self._registry.get_internal(request.voice)
        prompt = await checkpoint.run(
            self._adapter.build_clone_prompt,
            model,
            voice.audio_path,
            voice.transcript,
        )
        self._clone_prompts[request.voice] = prompt
        checkpoint.raise_if_requested()
        return prompt

    async def _unload_model(
        self,
        checkpoint: _CancellationCheckpoint,
        *,
        cleanup_if_unloaded: bool = False,
    ) -> None:
        had_model = self._model is not None
        loaded_model = self._loaded_model
        self._model = None
        self._loaded_model = None
        self._last_used_at = None
        if loaded_model == ModelID.QWEN_CLONE:
            self._clone_prompts.clear()
        if self._state != WorkerState.ERROR:
            self._state = WorkerState.UNLOADED
        if had_model or cleanup_if_unloaded:
            try:
                await checkpoint.run(self._adapter.cleanup_device_memory)
            except Exception:
                checkpoint.raise_if_requested()
                raise
        checkpoint.raise_if_requested()

    def _is_idle_unload_due(self) -> bool:
        return (
            self._model is not None
            and self._last_used_at is not None
            and self._monotonic() - self._last_used_at >= self._settings.idle_unload_seconds
        )

    def _enter_error(self, code: str) -> None:
        self._state = WorkerState.ERROR
        self._error_started_at = self._monotonic()
        self._last_error_code = code

    def _touch_resident_model(self) -> None:
        if self._model is not None:
            self._last_used_at = self._monotonic()


class _CancellationCheckpoint:
    def __init__(self) -> None:
        self.requested = False

    async def run[ResultT, **Parameters](
        self,
        function: Callable[Parameters, ResultT],
        /,
        *args: Parameters.args,
        **kwargs: Parameters.kwargs,
    ) -> ResultT:
        task = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
        while True:
            try:
                return await asyncio.shield(task)
            except asyncio.CancelledError:
                self.requested = True
                if task.done():
                    return task.result()

    def raise_if_requested(self) -> None:
        if self.requested:
            raise asyncio.CancelledError
