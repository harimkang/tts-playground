import structlog
from structlog.typing import EventDict, WrappedLogger

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


def _retain_privacy_safe_fields(
    _logger: WrappedLogger,
    _method_name: str,
    event_dict: EventDict,
) -> EventDict:
    return {field: event_dict[field] for field in _SAFE_LOG_FIELDS if field in event_dict}
