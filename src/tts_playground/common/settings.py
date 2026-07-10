from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class GatewaySettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TTS_", extra="ignore")
    api_key_file: Path = Path("/run/secrets/tts_api_key")
    public_port: int = 8020
    qwen_url: str = "http://qwen-worker:8001"
    supertonic_url: str = "http://supertonic-worker:7788"
    qwen_timeout_seconds: float = 360.0
    supertonic_timeout_seconds: float = 60.0
    max_upload_bytes: int = 20 * 1024 * 1024


class QwenSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="QWEN_", extra="ignore")
    voice_data_dir: Path = Path("/data/voices")
    max_waiting: int = 3
    queue_timeout_seconds: float = 180.0
    idle_unload_seconds: float = 600.0
    min_free_vram_bytes: int = Field(default=10 * 1024**3)
    error_cooldown_seconds: float = 30.0
    attention_implementation: str = "sdpa"
