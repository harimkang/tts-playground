from __future__ import annotations

import asyncio
import gc
import io
import threading
import weakref
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from tts_playground.common.errors import ServiceError
from tts_playground.common.models import QWEN_REPOSITORIES, ModelID
from tts_playground.common.settings import QwenSettings
from tts_playground.qwen.adapter import (
    CudaOutOfMemory,
    InternalSpeechRequest,
    SynthesisResult,
)
from tts_playground.qwen.manager import QwenModelManager, WorkerState
from tts_playground.qwen.voice_registry import VoiceRegistry


@dataclass(slots=True, weakref_slot=True)
class FakeModel:
    name: str
    generation: int


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FakeAdapter:
    def __init__(self) -> None:
        self.events: list[str] = []
        self.free_vram = 11 * 1024**3
        self.free_vram_calls = 0
        self.load_calls: list[str] = []
        self.build_calls: list[str] = []
        self.synthesize_oom_failures = 0
        self.synthesize_failures: list[BaseException] = []
        self.load_failures: list[BaseException] = []
        self.thread_calls: list[tuple[str, int]] = []
        self.block_call: str | None = None
        self.blocked_call_started = threading.Event()
        self.release_blocked_call = threading.Event()
        self.block_load_call: int | None = None
        self.blocked_load_started = threading.Event()
        self.release_blocked_load = threading.Event()
        self.block_synthesis = False
        self.block_synthesis_call: int | None = None
        self.first_synthesis_started = threading.Event()
        self.second_synthesis_started = threading.Event()
        self.release_synthesis = threading.Event()
        self._synthesis_calls = 0
        self._synthesis_lock = threading.Lock()
        self._load_attempts = 0
        self._generation = 0
        self.model_references: list[weakref.ReferenceType[FakeModel]] = []
        self.models_alive_during_cleanup: list[list[bool]] = []

    def _record_thread(self, call: str) -> None:
        self.thread_calls.append((call, threading.get_ident()))

    def _block_if_configured(self, call: str) -> None:
        if self.block_call != call:
            return
        self.blocked_call_started.set()
        if not self.release_blocked_call.wait(timeout=2.0):
            raise AssertionError(f"test did not release blocked {call}")

    def free_vram_bytes(self) -> int:
        self._record_thread("free_vram_bytes")
        self.free_vram_calls += 1
        self._block_if_configured("free_vram_bytes")
        return self.free_vram

    def load(self, repository: str, attention_implementation: str) -> object:
        self._record_thread("load")
        assert attention_implementation == "sdpa"
        name = _repository_name(repository)
        self.events.append(f"load:{name}")
        self.load_calls.append(repository)
        self._block_if_configured("load")
        self._load_attempts += 1
        if self.block_load_call == self._load_attempts:
            self.blocked_load_started.set()
            if not self.release_blocked_load.wait(timeout=2.0):
                raise AssertionError("test did not release blocked load")
        if self.load_failures:
            raise self.load_failures.pop(0)
        self._generation += 1
        model = FakeModel(name=name, generation=self._generation)
        self.model_references.append(weakref.ref(model))
        return model

    def cleanup_device_memory(self) -> None:
        self._record_thread("cleanup_device_memory")
        gc.collect()
        self.models_alive_during_cleanup.append(
            [model_reference() is not None for model_reference in self.model_references]
        )
        self.events.append("cleanup")
        self._block_if_configured("cleanup_device_memory")

    def build_clone_prompt(
        self,
        model: object,
        audio_path: Path,
        transcript: str,
    ) -> object:
        self._record_thread("build_clone_prompt")
        assert isinstance(model, FakeModel)
        assert model.name == "base"
        voice_id = audio_path.parent.name
        assert transcript == "test transcript"
        self.events.append(f"build:{voice_id}")
        self.build_calls.append(voice_id)
        self._block_if_configured("build_clone_prompt")
        return (voice_id, model.generation)

    def synthesize(
        self,
        model: object,
        request: InternalSpeechRequest,
        clone_prompt: object | None,
    ) -> SynthesisResult:
        self._record_thread("synthesize")
        assert isinstance(model, FakeModel)
        self.events.append(f"synthesize:{model.name}")
        self._block_if_configured("synthesize")
        with self._synthesis_lock:
            self._synthesis_calls += 1
            synthesis_call = self._synthesis_calls
        if self.block_synthesis or self.block_synthesis_call == synthesis_call:
            if synthesis_call == 1:
                self.first_synthesis_started.set()
            else:
                self.second_synthesis_started.set()
            if not self.release_synthesis.wait(timeout=2.0):
                raise AssertionError("test did not release blocked synthesis")
        if self.synthesize_failures:
            raise self.synthesize_failures.pop(0)
        if self.synthesize_oom_failures:
            self.synthesize_oom_failures -= 1
            raise CudaOutOfMemory
        if request.model == ModelID.QWEN_CLONE:
            assert clone_prompt is not None
        else:
            assert clone_prompt is None
        return SynthesisResult(
            audio=f"audio:{model.name}".encode(),
            sample_rate=24_000,
            duration_seconds=1.25,
        )


def _repository_name(repository: str) -> str:
    if repository.endswith("CustomVoice"):
        return "custom"
    if repository.endswith("VoiceDesign"):
        return "design"
    if repository.endswith("Base"):
        return "base"
    raise AssertionError(f"unexpected repository: {repository}")


def custom_request() -> InternalSpeechRequest:
    return InternalSpeechRequest(
        model=ModelID.QWEN_CUSTOM,
        text="hello",
        language="English",
        voice="Ryan",
        instructions=None,
    )


def design_request() -> InternalSpeechRequest:
    return InternalSpeechRequest(
        model=ModelID.QWEN_DESIGN,
        text="hello",
        language="English",
        voice=None,
        instructions="calm and clear",
    )


def clone_request(voice_id: str) -> InternalSpeechRequest:
    return InternalSpeechRequest(
        model=ModelID.QWEN_CLONE,
        text="hello",
        language="English",
        voice=voice_id,
        instructions=None,
    )


def _wav_bytes(duration_seconds: float = 4.0, sample_rate: int = 8_000) -> bytes:
    audio = np.zeros(round(duration_seconds * sample_rate), dtype=np.float32)
    output = io.BytesIO()
    sf.write(output, audio, sample_rate, format="WAV", subtype="PCM_16")
    return output.getvalue()


@pytest.fixture
def fake_adapter() -> FakeAdapter:
    return FakeAdapter()


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def registry(tmp_path: Path) -> VoiceRegistry:
    return VoiceRegistry(tmp_path)


@pytest.fixture
def manager(
    fake_adapter: FakeAdapter,
    registry: VoiceRegistry,
    clock: FakeClock,
) -> Iterator[QwenModelManager]:
    instance = QwenModelManager(
        adapter=fake_adapter,
        registry=registry,
        settings=QwenSettings(),
        monotonic=clock,
    )
    yield instance


def _register_voice(registry: VoiceRegistry) -> str:
    registered = registry.register(
        name="test voice",
        audio=_wav_bytes(),
        transcript="test transcript",
        language="en",
        consent_confirmed=True,
    )
    return registered.id


async def test_first_custom_request_loads_only_custom_repository(
    manager: QwenModelManager,
    fake_adapter: FakeAdapter,
) -> None:
    result = await manager.synthesize(custom_request())

    assert result.audio == b"audio:custom"
    assert fake_adapter.load_calls == [QWEN_REPOSITORIES[ModelID.QWEN_CUSTOM]]
    assert fake_adapter.events == ["load:custom", "synthesize:custom"]
    status = manager.status()
    assert (status.state, status.loaded_model) == (WorkerState.READY, ModelID.QWEN_CUSTOM)


async def test_second_request_reuses_resident_custom_model(
    manager: QwenModelManager,
    fake_adapter: FakeAdapter,
) -> None:
    await manager.synthesize(custom_request())
    await manager.synthesize(custom_request())

    assert fake_adapter.events == [
        "load:custom",
        "synthesize:custom",
        "synthesize:custom",
    ]
    assert fake_adapter.free_vram_calls == 1


async def test_switch_releases_model_before_cleanup_and_next_load(
    manager: QwenModelManager,
    fake_adapter: FakeAdapter,
) -> None:
    await manager.synthesize(custom_request())
    await manager.synthesize(design_request())

    assert fake_adapter.events == [
        "load:custom",
        "synthesize:custom",
        "cleanup",
        "load:design",
        "synthesize:design",
    ]
    assert fake_adapter.models_alive_during_cleanup == [[False]]


async def test_free_vram_guard_rejects_without_loading(
    manager: QwenModelManager,
    fake_adapter: FakeAdapter,
) -> None:
    fake_adapter.free_vram = 10 * 1024**3 - 1

    with pytest.raises(ServiceError) as error:
        await manager.synthesize(custom_request())

    assert (error.value.status_code, error.value.code) == (503, "insufficient_vram")
    assert fake_adapter.free_vram_calls == 1
    assert fake_adapter.load_calls == []
    assert manager.status().state == WorkerState.UNLOADED


async def test_one_cuda_oom_releases_model_before_cleanup_and_retries_once(
    manager: QwenModelManager,
    fake_adapter: FakeAdapter,
) -> None:
    fake_adapter.synthesize_oom_failures = 1

    result = await manager.synthesize(custom_request())

    assert result.audio == b"audio:custom"
    assert fake_adapter.events == [
        "load:custom",
        "synthesize:custom",
        "cleanup",
        "load:custom",
        "synthesize:custom",
    ]
    assert fake_adapter.models_alive_during_cleanup == [[False]]
    assert manager.status().state == WorkerState.READY


async def test_two_cuda_oom_failures_enter_error_and_enforce_cooldown(
    manager: QwenModelManager,
    fake_adapter: FakeAdapter,
) -> None:
    fake_adapter.synthesize_oom_failures = 2

    with pytest.raises(ServiceError) as error:
        await manager.synthesize(custom_request())

    assert (error.value.status_code, error.value.code) == (503, "cuda_out_of_memory")
    status = manager.status()
    assert status.state == WorkerState.ERROR
    assert status.loaded_model is None
    assert status.last_error_code == "cuda_out_of_memory"
    events_after_failure = list(fake_adapter.events)

    with pytest.raises(ServiceError) as cooldown_error:
        await manager.synthesize(custom_request())

    assert (cooldown_error.value.status_code, cooldown_error.value.code) == (
        503,
        "model_unavailable",
    )
    assert fake_adapter.events == events_after_failure


async def test_first_request_after_cooldown_makes_one_clean_load_attempt(
    manager: QwenModelManager,
    fake_adapter: FakeAdapter,
    clock: FakeClock,
) -> None:
    fake_adapter.synthesize_oom_failures = 2
    with pytest.raises(ServiceError):
        await manager.synthesize(custom_request())
    load_count_after_failure = len(fake_adapter.load_calls)
    clock.advance(30.0)

    result = await manager.synthesize(custom_request())

    assert result.audio == b"audio:custom"
    assert len(fake_adapter.load_calls) == load_count_after_failure + 1
    status = manager.status()
    assert status.state == WorkerState.READY
    assert status.last_error_code is None


async def test_non_oom_load_failure_is_sanitized(
    manager: QwenModelManager,
    fake_adapter: FakeAdapter,
) -> None:
    fake_adapter.load_failures = [RuntimeError("secret checkpoint path")]

    with pytest.raises(ServiceError) as error:
        await manager.synthesize(custom_request())

    assert (error.value.status_code, error.value.code) == (503, "model_load_failed")
    assert "secret checkpoint path" not in error.value.message
    assert manager.status().last_error_code == "model_load_failed"


async def test_unload_if_idle_unloads_only_after_configured_interval(
    manager: QwenModelManager,
    fake_adapter: FakeAdapter,
    clock: FakeClock,
) -> None:
    await manager.synthesize(custom_request())
    clock.advance(599.0)

    await manager.unload_if_idle()
    assert fake_adapter.events[-1] == "synthesize:custom"

    clock.advance(1.0)
    await manager.unload_if_idle()
    assert fake_adapter.events[-1] == "cleanup"
    assert fake_adapter.models_alive_during_cleanup == [[False]]
    status = manager.status()
    assert (status.state, status.loaded_model) == (WorkerState.UNLOADED, None)


async def test_invalid_clone_lookup_still_starts_resident_idle_timer(
    manager: QwenModelManager,
    fake_adapter: FakeAdapter,
    clock: FakeClock,
) -> None:
    missing_voice = f"voice_{'0' * 32}"

    with pytest.raises(ServiceError) as error:
        await manager.synthesize(clone_request(missing_voice))

    assert error.value.code == "voice_not_found"
    assert manager.status().loaded_model == ModelID.QWEN_CLONE
    clock.advance(600.0)
    await manager.unload_if_idle()
    assert fake_adapter.events == ["load:base", "cleanup"]


async def test_non_oom_synthesis_failure_still_starts_resident_idle_timer(
    manager: QwenModelManager,
    fake_adapter: FakeAdapter,
    clock: FakeClock,
) -> None:
    fake_adapter.synthesize_failures = [RuntimeError("generation failed")]

    with pytest.raises(RuntimeError, match="generation failed"):
        await manager.synthesize(custom_request())

    assert manager.status().loaded_model == ModelID.QWEN_CUSTOM
    clock.advance(600.0)
    await manager.unload_if_idle()
    assert fake_adapter.events == [
        "load:custom",
        "synthesize:custom",
        "cleanup",
    ]


async def test_failed_use_of_resident_model_resets_idle_timer(
    manager: QwenModelManager,
    fake_adapter: FakeAdapter,
    clock: FakeClock,
) -> None:
    await manager.synthesize(custom_request())
    clock.advance(500.0)
    fake_adapter.synthesize_failures = [RuntimeError("generation failed")]
    with pytest.raises(RuntimeError, match="generation failed"):
        await manager.synthesize(custom_request())

    clock.advance(599.0)
    await manager.unload_if_idle()
    assert fake_adapter.events[-1] == "synthesize:custom"

    clock.advance(1.0)
    await manager.unload_if_idle()
    assert fake_adapter.events[-1] == "cleanup"


async def test_clone_prompts_are_per_voice_and_cleared_when_base_unloads(
    manager: QwenModelManager,
    fake_adapter: FakeAdapter,
    registry: VoiceRegistry,
) -> None:
    first_voice = _register_voice(registry)
    second_voice = _register_voice(registry)

    await manager.synthesize(clone_request(first_voice))
    await manager.synthesize(clone_request(first_voice))
    await manager.synthesize(clone_request(second_voice))
    assert fake_adapter.build_calls == [first_voice, second_voice]

    await manager.synthesize(custom_request())
    await manager.synthesize(clone_request(first_voice))
    assert fake_adapter.build_calls == [first_voice, second_voice, first_voice]


async def test_invalidate_voice_discards_its_cached_clone_prompt(
    manager: QwenModelManager,
    fake_adapter: FakeAdapter,
    registry: VoiceRegistry,
) -> None:
    voice_id = _register_voice(registry)
    await manager.synthesize(clone_request(voice_id))

    manager.invalidate_voice(voice_id)
    await manager.synthesize(clone_request(voice_id))

    assert fake_adapter.build_calls == [voice_id, voice_id]


async def test_all_blocking_adapter_calls_run_off_the_event_loop(
    manager: QwenModelManager,
    fake_adapter: FakeAdapter,
    registry: VoiceRegistry,
) -> None:
    voice_id = _register_voice(registry)
    await manager.synthesize(clone_request(voice_id))
    await manager.shutdown()

    calls = {call for call, _ in fake_adapter.thread_calls}
    assert calls == {
        "free_vram_bytes",
        "load",
        "build_clone_prompt",
        "synthesize",
        "cleanup_device_memory",
    }
    main_thread = threading.get_ident()
    assert all(thread_id != main_thread for _, thread_id in fake_adapter.thread_calls)


async def test_cancelling_active_request_does_not_overlap_adapter_calls(
    manager: QwenModelManager,
    fake_adapter: FakeAdapter,
) -> None:
    fake_adapter.block_synthesis = True
    first = asyncio.create_task(manager.synthesize(custom_request()))
    assert await asyncio.to_thread(fake_adapter.first_synthesis_started.wait, 1.0)

    first.cancel()
    second = asyncio.create_task(manager.synthesize(custom_request()))
    await asyncio.sleep(0.02)
    overlapped = fake_adapter.second_synthesis_started.is_set()

    fake_adapter.release_synthesis.set()
    results = await asyncio.gather(first, second, return_exceptions=True)

    assert not overlapped
    assert isinstance(results[0], asyncio.CancelledError)
    assert isinstance(results[1], SynthesisResult)


async def test_cancellation_during_free_vram_check_stops_before_load(
    manager: QwenModelManager,
    fake_adapter: FakeAdapter,
) -> None:
    fake_adapter.block_call = "free_vram_bytes"
    synthesis = asyncio.create_task(manager.synthesize(custom_request()))
    assert await asyncio.to_thread(fake_adapter.blocked_call_started.wait, 1.0)

    synthesis.cancel()
    fake_adapter.release_blocked_call.set()
    with pytest.raises(asyncio.CancelledError):
        await synthesis

    assert fake_adapter.load_calls == []
    assert fake_adapter.events == []
    assert manager.status().state == WorkerState.UNLOADED


async def test_cancellation_during_load_preserves_model_and_stops_before_synthesis(
    manager: QwenModelManager,
    fake_adapter: FakeAdapter,
    clock: FakeClock,
) -> None:
    fake_adapter.block_call = "load"
    synthesis = asyncio.create_task(manager.synthesize(custom_request()))
    assert await asyncio.to_thread(fake_adapter.blocked_call_started.wait, 1.0)

    synthesis.cancel()
    fake_adapter.release_blocked_call.set()
    with pytest.raises(asyncio.CancelledError):
        await synthesis

    assert fake_adapter.events == ["load:custom"]
    status = manager.status()
    assert (status.state, status.loaded_model) == (WorkerState.READY, ModelID.QWEN_CUSTOM)

    clock.advance(600.0)
    await manager.unload_if_idle()
    assert fake_adapter.events[-1] == "cleanup"


async def test_cancellation_during_clone_prompt_stops_before_synthesis(
    manager: QwenModelManager,
    fake_adapter: FakeAdapter,
    registry: VoiceRegistry,
) -> None:
    voice_id = _register_voice(registry)
    fake_adapter.block_call = "build_clone_prompt"
    synthesis = asyncio.create_task(manager.synthesize(clone_request(voice_id)))
    assert await asyncio.to_thread(fake_adapter.blocked_call_started.wait, 1.0)

    synthesis.cancel()
    fake_adapter.release_blocked_call.set()
    with pytest.raises(asyncio.CancelledError):
        await synthesis

    assert fake_adapter.events == ["load:base", f"build:{voice_id}"]
    assert fake_adapter.build_calls == [voice_id]
    assert manager.status().loaded_model == ModelID.QWEN_CLONE


async def test_cancellation_during_oom_waits_for_cleanup_before_propagating(
    manager: QwenModelManager,
    fake_adapter: FakeAdapter,
) -> None:
    fake_adapter.synthesize_oom_failures = 1
    fake_adapter.block_synthesis = True
    fake_adapter.block_call = "cleanup_device_memory"
    synthesis = asyncio.create_task(manager.synthesize(custom_request()))
    assert await asyncio.to_thread(fake_adapter.first_synthesis_started.wait, 1.0)

    synthesis.cancel()
    fake_adapter.release_synthesis.set()
    assert await asyncio.to_thread(fake_adapter.blocked_call_started.wait, 1.0)
    assert not synthesis.done()
    fake_adapter.release_blocked_call.set()
    with pytest.raises(asyncio.CancelledError):
        await synthesis

    assert fake_adapter.events == [
        "load:custom",
        "synthesize:custom",
        "cleanup",
    ]
    assert fake_adapter.models_alive_during_cleanup == [[False]]
    assert manager.status().state == WorkerState.UNLOADED


async def test_cancellation_during_second_oom_commits_error_cooldown(
    manager: QwenModelManager,
    fake_adapter: FakeAdapter,
) -> None:
    fake_adapter.synthesize_oom_failures = 2
    fake_adapter.block_synthesis_call = 2
    synthesis = asyncio.create_task(manager.synthesize(custom_request()))
    assert await asyncio.to_thread(fake_adapter.second_synthesis_started.wait, 1.0)

    synthesis.cancel()
    fake_adapter.release_synthesis.set()
    with pytest.raises(asyncio.CancelledError):
        await synthesis

    assert fake_adapter.events == [
        "load:custom",
        "synthesize:custom",
        "cleanup",
        "load:custom",
        "synthesize:custom",
        "cleanup",
    ]
    status = manager.status()
    assert status.state == WorkerState.ERROR
    assert status.loaded_model is None
    assert status.last_error_code == "cuda_out_of_memory"

    events_after_cancellation = list(fake_adapter.events)
    with pytest.raises(ServiceError) as cooldown_error:
        await manager.synthesize(custom_request())
    assert cooldown_error.value.code == "model_unavailable"
    assert fake_adapter.events == events_after_cancellation


async def test_cancellation_during_first_load_oom_stops_before_retry(
    manager: QwenModelManager,
    fake_adapter: FakeAdapter,
) -> None:
    fake_adapter.load_failures = [CudaOutOfMemory()]
    fake_adapter.block_load_call = 1
    synthesis = asyncio.create_task(manager.synthesize(custom_request()))
    assert await asyncio.to_thread(fake_adapter.blocked_load_started.wait, 1.0)

    synthesis.cancel()
    fake_adapter.release_blocked_load.set()
    with pytest.raises(asyncio.CancelledError):
        await synthesis

    assert fake_adapter.events == ["load:custom", "cleanup"]
    assert len(fake_adapter.load_calls) == 1
    assert manager.status().state == WorkerState.UNLOADED


async def test_cancellation_during_second_load_oom_commits_error_cooldown(
    manager: QwenModelManager,
    fake_adapter: FakeAdapter,
) -> None:
    fake_adapter.load_failures = [CudaOutOfMemory(), CudaOutOfMemory()]
    fake_adapter.block_load_call = 2
    synthesis = asyncio.create_task(manager.synthesize(custom_request()))
    assert await asyncio.to_thread(fake_adapter.blocked_load_started.wait, 1.0)

    synthesis.cancel()
    fake_adapter.release_blocked_load.set()
    with pytest.raises(asyncio.CancelledError):
        await synthesis

    assert fake_adapter.events == [
        "load:custom",
        "cleanup",
        "load:custom",
        "cleanup",
    ]
    assert len(fake_adapter.load_calls) == 2
    status = manager.status()
    assert status.state == WorkerState.ERROR
    assert status.loaded_model is None
    assert status.last_error_code == "cuda_out_of_memory"

    events_after_cancellation = list(fake_adapter.events)
    with pytest.raises(ServiceError) as cooldown_error:
        await manager.synthesize(custom_request())
    assert cooldown_error.value.code == "model_unavailable"
    assert fake_adapter.events == events_after_cancellation


async def test_shutdown_releases_model_before_device_cleanup(
    manager: QwenModelManager,
    fake_adapter: FakeAdapter,
) -> None:
    await manager.synthesize(custom_request())

    await manager.shutdown()

    assert fake_adapter.events == ["load:custom", "synthesize:custom", "cleanup"]
    assert fake_adapter.models_alive_during_cleanup == [[False]]
    assert manager.status().loaded_model is None
