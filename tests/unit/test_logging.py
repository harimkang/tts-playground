import json

import pytest
import structlog

from tts_playground.common.logging import configure_logging, log_gateway_request


def test_gateway_request_log_contains_only_privacy_safe_fields(
    capsys: pytest.CaptureFixture[str],
) -> None:
    configure_logging()
    log_gateway_request(
        request_id="req_logging-test",
        model="qwen3-tts-1.7b-design",
        input_chars=12,
        queue_ms="3",
        processing_ms="45",
        status=200,
        error_code=None,
    )

    output = capsys.readouterr().out
    payload = json.loads(output)
    assert payload == {
        "event": "gateway_synthesis",
        "request_id": "req_logging-test",
        "model": "qwen3-tts-1.7b-design",
        "input_chars": 12,
        "queue_ms": "3",
        "processing_ms": "45",
        "status": 200,
        "error_code": None,
    }


def test_logging_processor_drops_sensitive_fields(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging()
    structlog.get_logger().info(
        "gateway_synthesis",
        request_id="req_logging-test",
        input="secret input marker",
        content="secret request content marker",
        instructions="secret instruction marker",
        authorization="Bearer secret credential marker",
        transcript="secret transcript marker",
        audio=b"secret audio marker",
    )

    output = capsys.readouterr().out
    assert "secret input marker" not in output
    assert "secret request content marker" not in output
    assert "secret instruction marker" not in output
    assert "secret credential marker" not in output
    assert "secret transcript marker" not in output
    assert "secret audio marker" not in output
    assert json.loads(output) == {
        "event": "gateway_synthesis",
        "request_id": "req_logging-test",
    }


def test_gateway_request_log_sanitizes_sensitive_allowed_field_values(
    capsys: pytest.CaptureFixture[str],
) -> None:
    configure_logging()
    log_gateway_request(
        request_id="req_logging-test",
        model="supertonic-3",
        input_chars=4,
        queue_ms="sensitive queue marker",
        processing_ms="sensitive processing marker",
        status=503,
        error_code="SENSITIVE Error Code Marker!",
    )

    output = capsys.readouterr().out
    assert "sensitive queue marker" not in output
    assert "sensitive processing marker" not in output
    assert "SENSITIVE Error Code Marker!" not in output
    assert json.loads(output) == {
        "event": "gateway_synthesis",
        "request_id": "req_logging-test",
        "model": "supertonic-3",
        "input_chars": 4,
        "status": 503,
        "error_code": "backend_error",
    }


@pytest.mark.parametrize(
    ("queue_ms", "processing_ms"),
    [("NaN", "Inf"), ("-1", "86400001")],
)
def test_gateway_request_log_drops_nonfinite_or_out_of_range_timings(
    capsys: pytest.CaptureFixture[str],
    queue_ms: str,
    processing_ms: str,
) -> None:
    configure_logging()
    log_gateway_request(
        request_id="req_logging-test",
        model="supertonic-3",
        input_chars=4,
        queue_ms=queue_ms,
        processing_ms=processing_ms,
        status=200,
        error_code=None,
    )

    payload = json.loads(capsys.readouterr().out)
    assert "queue_ms" not in payload
    assert "processing_ms" not in payload
