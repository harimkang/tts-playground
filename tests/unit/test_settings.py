import json
from pathlib import Path

from tts_playground.common.settings import GatewaySettings, QwenSettings


def test_settings_defaults_match_design() -> None:
    gateway = GatewaySettings(api_key_file=Path("/tmp/key"))
    qwen = QwenSettings()
    assert gateway.public_port == 8020
    assert gateway.qwen_timeout_seconds == 360.0
    largest_legal_body = json.dumps(
        {
            "model": "qwen3-tts-1.7b-custom",
            "input": "😀" * 2000,
            "voice": "Sohee",
            "language": "en",
            "instructions": "😀" * 1000,
        }
    ).encode()
    assert gateway.max_synthesis_body_bytes >= len(largest_legal_body)
    assert qwen.max_waiting == 3
    assert qwen.queue_timeout_seconds == 180.0
    assert qwen.idle_unload_seconds == 600.0
    assert qwen.min_free_vram_bytes == 10 * 1024**3
    assert qwen.error_cooldown_seconds == 30.0
    assert qwen.device == "cuda:0"
