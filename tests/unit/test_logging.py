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
