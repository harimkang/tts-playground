import math
import re

import structlog
from structlog.typing import EventDict, WrappedLogger

MAX_TIMING_MILLISECONDS = 86_400_000.0
MAX_AUDIO_DURATION_SECONDS = 86_400.0

_ERROR_CODE_PATTERN = re.compile(r"[a-z][a-z0-9_]{0,63}")
_SAFE_LOG_FIELDS = (
    "event",
    "request_id",
    "model",
    "input_chars",
    "queue_ms",
    "processing_ms",
    "status",
    "error_code",
)


def configure_logging() -> None:
    structlog.configure(
        processors=[_retain_privacy_safe_fields, structlog.processors.JSONRenderer()],
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=False,
    )


def log_gateway_request(
    *,
    request_id: str,
    model: str,
    input_chars: int,
    queue_ms: str | None,
    processing_ms: str | None,
    status: int,
    error_code: str | None,
) -> None:
    structlog.get_logger().info(
        "gateway_synthesis",
        request_id=request_id,
        model=model,
        input_chars=input_chars,
        queue_ms=queue_ms,
        processing_ms=processing_ms,
        status=status,
        error_code=error_code,
    )


def normalize_bounded_nonnegative(
    value: str | None,
    *,
    maximum: float,
) -> str | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    if not math.isfinite(parsed) or parsed < 0 or parsed > maximum:
        return None
    if parsed == 0:
        return "0"
    return format(parsed, ".15g")


def normalize_error_code(value: str | None) -> str | None:
    if value is None:
        return None
    if _ERROR_CODE_PATTERN.fullmatch(value) is None:
        return "backend_error"
    return value


def _retain_privacy_safe_fields(
    _logger: WrappedLogger,
    _method_name: str,
    event_dict: EventDict,
) -> EventDict:
    safe_event = {
        field: event_dict[field]
        for field in _SAFE_LOG_FIELDS
        if field in event_dict and field not in {"queue_ms", "processing_ms", "error_code"}
    }

    for field in ("queue_ms", "processing_ms"):
        value = event_dict.get(field)
        if isinstance(value, str):
            normalized = normalize_bounded_nonnegative(
                value,
                maximum=MAX_TIMING_MILLISECONDS,
            )
            if normalized is not None:
                safe_event[field] = normalized

    if "error_code" in event_dict:
        value = event_dict["error_code"]
        if value is None or isinstance(value, str):
            safe_event["error_code"] = normalize_error_code(value)

    return safe_event
