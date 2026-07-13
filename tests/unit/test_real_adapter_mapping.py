from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import soundfile as sf

from tts_playground.common.models import ModelID
from tts_playground.qwen.adapter import CudaOutOfMemory, InternalSpeechRequest
from tts_playground.qwen.real_adapter import QwenTTSAdapter


class FakeCudaOutOfMemoryError(RuntimeError):
    pass


class FakeCuda:
    OutOfMemoryError = FakeCudaOutOfMemoryError

    def __init__(self) -> None:
        self.empty_cache_calls = 0
        self.ipc_collect_calls = 0
        self.mem_get_info_result = (12_345, 67_890)

    def mem_get_info(self) -> tuple[int, int]:
        return self.mem_get_info_result

    def empty_cache(self) -> None:
        self.empty_cache_calls += 1

    def ipc_collect(self) -> None:
        self.ipc_collect_calls += 1


class FakeTorch:
    def __init__(self) -> None:
        self.cuda = FakeCuda()
        self.bfloat16 = object()


def _request(
    *,
    model: ModelID,
    voice: str | None,
    instructions: str | None,
) -> InternalSpeechRequest:
    return InternalSpeechRequest(
        model=model,
        text="안녕",
        language="Korean",
        voice=voice,
        instructions=instructions,
    )


def _model_with_audio(sample_rate: int = 32_000) -> Mock:
    model = Mock()
    generated = [np.array([-1.0, -0.25, 0.25, 1.0], dtype=np.float32)]
    model.generate_custom_voice.return_value = (generated, sample_rate)
    model.generate_voice_design.return_value = (generated, sample_rate)
    model.generate_voice_clone.return_value = (generated, sample_rate)
    return model


def _assert_pcm16_wav(audio: bytes, sample_rate: int) -> None:
    info = sf.info(io.BytesIO(audio))
    assert info.format == "WAV"
    assert info.subtype == "PCM_16"
    assert info.samplerate == sample_rate
    assert info.frames == 4


@pytest.fixture
def torch_module() -> FakeTorch:
    return FakeTorch()


@pytest.fixture
def model_class() -> Mock:
    return Mock()


@pytest.fixture
def adapter(torch_module: FakeTorch, model_class: Mock) -> QwenTTSAdapter:
    return QwenTTSAdapter(torch_module=torch_module, model_class=model_class)


def test_custom_maps_to_qwen_package(adapter: QwenTTSAdapter) -> None:
    model = _model_with_audio(sample_rate=22_050)

    result = adapter.synthesize(
        model,
        request=_request(
            model=ModelID.QWEN_CUSTOM,
            voice="Sohee",
            instructions="차분하게",
        ),
        clone_prompt=None,
    )

    model.generate_custom_voice.assert_called_once_with(
        text="안녕",
        language="Korean",
        speaker="Sohee",
        instruct="차분하게",
    )
    _assert_pcm16_wav(result.audio, 22_050)
    assert result.sample_rate == 22_050
    assert result.duration_seconds == pytest.approx(4 / 22_050)


def test_voice_design_maps_to_qwen_package(adapter: QwenTTSAdapter) -> None:
    model = _model_with_audio()

    adapter.synthesize(
        model,
        request=_request(
            model=ModelID.QWEN_DESIGN,
            voice=None,
            instructions="따뜻하고 낮은 목소리",
        ),
        clone_prompt=None,
    )

    model.generate_voice_design.assert_called_once_with(
        text="안녕",
        language="Korean",
        instruct="따뜻하고 낮은 목소리",
    )


@pytest.mark.parametrize(
    ("transcript", "x_vector_only_mode"),
    [("정확한 녹취", False), (None, True)],
)
def test_build_clone_prompt_selects_mode_from_transcript(
    adapter: QwenTTSAdapter,
    transcript: str | None,
    x_vector_only_mode: bool,
) -> None:
    model = Mock()
    audio_path = Path("/tmp/reference.wav")

    adapter.build_clone_prompt(model, audio_path, transcript)

    model.create_voice_clone_prompt.assert_called_once_with(
        ref_audio=str(audio_path),
        ref_text=transcript,
        x_vector_only_mode=x_vector_only_mode,
    )


def test_voice_clone_reuses_prebuilt_prompt(adapter: QwenTTSAdapter) -> None:
    model = _model_with_audio()
    prompt = object()

    adapter.synthesize(
        model,
        request=_request(
            model=ModelID.QWEN_CLONE,
            voice="voice_0123456789abcdef0123456789abcdef",
            instructions=None,
        ),
        clone_prompt=prompt,
    )

    model.generate_voice_clone.assert_called_once_with(
        text="안녕",
        language="Korean",
        voice_clone_prompt=prompt,
    )


def test_load_maps_exact_model_arguments(
    adapter: QwenTTSAdapter,
    torch_module: FakeTorch,
    model_class: Mock,
) -> None:
    loaded_model = object()
    model_class.from_pretrained.return_value = loaded_model

    result = adapter.load("Qwen/repository", "sdpa")

    assert result is loaded_model
    model_class.from_pretrained.assert_called_once_with(
        "Qwen/repository",
        device_map="cuda:0",
        dtype=torch_module.bfloat16,
        attn_implementation="sdpa",
    )


def test_free_vram_returns_cuda_free_bytes(
    adapter: QwenTTSAdapter,
    torch_module: FakeTorch,
) -> None:
    assert adapter.free_vram_bytes() == 12_345


def test_free_vram_does_not_require_qwen_package(torch_module: FakeTorch) -> None:
    adapter = QwenTTSAdapter(torch_module=torch_module)

    assert adapter.free_vram_bytes() == 12_345


def test_cleanup_device_memory_collects_python_and_cuda_resources(
    adapter: QwenTTSAdapter,
    torch_module: FakeTorch,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collect = Mock()
    monkeypatch.setattr("tts_playground.qwen.real_adapter.gc.collect", collect)

    adapter.cleanup_device_memory()

    collect.assert_called_once_with()
    assert torch_module.cuda.empty_cache_calls == 1
    assert torch_module.cuda.ipc_collect_calls == 1


def test_generation_cuda_oom_is_converted(adapter: QwenTTSAdapter) -> None:
    model = Mock()
    model.generate_custom_voice.side_effect = FakeCudaOutOfMemoryError

    with pytest.raises(CudaOutOfMemory):
        adapter.synthesize(
            model,
            request=_request(
                model=ModelID.QWEN_CUSTOM,
                voice="Sohee",
                instructions=None,
            ),
            clone_prompt=None,
        )


def test_load_cuda_oom_is_converted(adapter: QwenTTSAdapter, model_class: Mock) -> None:
    model_class.from_pretrained.side_effect = FakeCudaOutOfMemoryError

    with pytest.raises(CudaOutOfMemory):
        adapter.load("Qwen/repository", "sdpa")
