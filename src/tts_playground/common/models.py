from enum import StrEnum
from typing import Final


class ModelID(StrEnum):
    SUPERTONIC = "supertonic-3"
    QWEN_CUSTOM = "qwen3-tts-1.7b-custom"
    QWEN_DESIGN = "qwen3-tts-1.7b-design"
    QWEN_CLONE = "qwen3-tts-1.7b-clone"


QWEN_CUSTOM_SPEAKERS = (
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


QWEN_LANGUAGES: Final = {
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


QWEN_REPOSITORIES: dict[ModelID, str] = {
    ModelID.QWEN_CUSTOM: "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    ModelID.QWEN_DESIGN: "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    ModelID.QWEN_CLONE: "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
}
