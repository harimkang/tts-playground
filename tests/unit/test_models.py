from tts_playground.common.errors import ServiceError
from tts_playground.common.models import QWEN_CUSTOM_SPEAKERS, QWEN_REPOSITORIES, ModelID


def test_public_model_ids_are_stable() -> None:
    assert [item.value for item in ModelID] == [
        "supertonic-3",
        "qwen3-tts-1.7b-custom",
        "qwen3-tts-1.7b-design",
        "qwen3-tts-1.7b-clone",
    ]
    assert QWEN_REPOSITORIES[ModelID.QWEN_CUSTOM].endswith("1.7B-CustomVoice")
    assert QWEN_REPOSITORIES[ModelID.QWEN_DESIGN].endswith("1.7B-VoiceDesign")
    assert QWEN_REPOSITORIES[ModelID.QWEN_CLONE].endswith("1.7B-Base")


def test_service_error_string_is_its_public_message() -> None:
    error = ServiceError(status_code=503, code="worker_unavailable", message="Worker unavailable")

    assert str(error) == "Worker unavailable"


def test_qwen_custom_voice_speaker_contract_is_exact() -> None:
    assert QWEN_CUSTOM_SPEAKERS == (
        "Vivian",
        "Serena",
        "Uncle_Fu",
        "Dylan",
        "Eric",
        "Ryan",
        "Aiden",
        "Ono_Anna",
        "Sohee",
    )
