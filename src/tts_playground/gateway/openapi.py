from typing import Any, Final

from tts_playground.common.models import ModelID
from tts_playground.gateway.schemas import ErrorEnvelope

_NUMERIC_HEADER_SCHEMA: Final = {
    "type": "string",
    "minLength": 1,
    "maxLength": 24,
}

EPHEMERAL_CLONE_OPENAPI_EXTRA: Final[dict[str, Any]] = {
    "requestBody": {
        "required": True,
        "content": {
            "multipart/form-data": {
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["input", "audio", "consent_confirmed"],
                    "properties": {
                        "input": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": 2000,
                        },
                        "audio": {"type": "string", "format": "binary"},
                        "consent_confirmed": {"type": "boolean"},
                        "transcript": {"type": "string", "maxLength": 4000},
                        "language": {
                            "type": "string",
                            "minLength": 2,
                            "maxLength": 10,
                            "default": "auto",
                        },
                        "response_format": {
                            "type": "string",
                            "enum": ["wav"],
                            "default": "wav",
                        },
                    },
                }
            }
        },
    }
}

EPHEMERAL_CLONE_RESPONSES: Final[dict[int | str, dict[str, Any]]] = {
    200: {
        "description": "WAV audio synthesized with the uploaded reference voice.",
        "content": {
            "audio/wav": {"schema": {"type": "string", "format": "binary"}}
        },
        "headers": {
            "X-Request-ID": {
                "description": "Safe request correlation identifier.",
                "schema": {
                    "type": "string",
                    "minLength": 12,
                    "maxLength": 84,
                    "pattern": r"^req_[A-Za-z0-9_-]{8,80}$",
                },
            },
            "X-TTS-Model": {
                "description": "Public model used for synthesis.",
                "schema": {
                    "type": "string",
                    "minLength": len(ModelID.QWEN_CLONE.value),
                    "maxLength": len(ModelID.QWEN_CLONE.value),
                    "enum": [ModelID.QWEN_CLONE.value],
                },
            },
            "X-Audio-Duration": {
                "description": "Generated audio duration in seconds, when available.",
                "schema": _NUMERIC_HEADER_SCHEMA,
            },
            "X-Queue-Ms": {
                "description": "Backend queue time in milliseconds, when available.",
                "schema": _NUMERIC_HEADER_SCHEMA,
            },
            "X-Processing-Ms": {
                "description": "Backend processing time in milliseconds, when available.",
                "schema": _NUMERIC_HEADER_SCHEMA,
            },
        },
    },
    401: {
        "description": "Authentication failed.",
        "model": ErrorEnvelope,
    },
    413: {
        "description": "The request or reference audio is too large.",
        "model": ErrorEnvelope,
    },
    422: {
        "description": "The multipart request failed public validation.",
        "model": ErrorEnvelope,
    },
    500: {
        "description": "The gateway encountered an internal error.",
        "model": ErrorEnvelope,
    },
}
