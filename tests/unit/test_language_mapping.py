import httpx
import pytest

from tts_playground.common.errors import ServiceError
from tts_playground.common.settings import GatewaySettings
from tts_playground.gateway.backends import (
    QWEN_LANGUAGES,
    SUPERTONIC_LANGUAGES,
    BackendRouter,
)
from tts_playground.gateway.schemas import SpeechRequest


def test_language_maps_are_explicit_and_complete() -> None:
    assert QWEN_LANGUAGES == {
        "auto": "Auto",
        "zh": "Chinese",
        "en": "English",
        "ja": "Japanese",
        "ko": "Korean",
        "de": "German",
        "fr": "French",
        "ru": "Russian",
        "pt": "Portuguese",
        "es": "Spanish",
        "it": "Italian",
    }
    assert SUPERTONIC_LANGUAGES == {
        "en",
        "ko",
        "ja",
        "ar",
        "bg",
        "cs",
        "da",
        "de",
        "el",
        "es",
        "et",
        "fi",
        "fr",
        "hi",
        "hr",
        "hu",
        "id",
        "it",
        "lt",
        "lv",
        "nl",
        "pl",
        "pt",
        "ro",
        "ru",
        "sk",
        "sl",
        "sv",
        "tr",
        "uk",
        "vi",
        "na",
    }


@pytest.mark.parametrize(
    ("speech_request", "expected_message"),
    [
        (
            SpeechRequest(
                model="supertonic-3",
                input="test",
                voice="F1",
                language="auto",
            ),
            "Supertonic does not support language 'auto'.",
        ),
        (
            SpeechRequest(
                model="qwen3-tts-1.7b-custom",
                input="test",
                voice="Sohee",
                language="ar",
            ),
            "Qwen does not support language 'ar'.",
        ),
    ],
)
async def test_router_rejects_unsupported_backend_language(
    speech_request: SpeechRequest,
    expected_message: str,
) -> None:
    async with httpx.AsyncClient() as client:
        router = BackendRouter(GatewaySettings(), client)

        with pytest.raises(ServiceError) as exc_info:
            await router.synthesize(speech_request, "req_language-test")

    assert exc_info.value.status_code == 422
    assert exc_info.value.code == "unsupported_language"
    assert exc_info.value.message == expected_message
