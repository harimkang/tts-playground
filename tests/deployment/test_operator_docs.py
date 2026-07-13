import re
from pathlib import Path

ROOT = Path(__file__).parents[2]


def _fenced_code_after(document: str, marker: str) -> str:
    marker_position = document.index(marker)
    match = re.search(
        r"```(?:sh|bash)\s*\n(?P<code>.*?)\n```",
        document[marker_position:],
        re.DOTALL,
    )
    assert match is not None, f"Missing fenced shell example after {marker}"
    return match.group("code")


def test_backend_timeout_configuration_is_documented() -> None:
    env_example = (ROOT / ".env.example").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    for name in ("TTS_QWEN_TIMEOUT_SECONDS", "TTS_SUPERTONIC_TIMEOUT_SECONDS"):
        assert name in env_example
        assert name in readme


def test_readme_documents_complete_uv_locked_operator_workflow() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    required_fragments = (
        "NVIDIA Container Toolkit",
        "cp .env.example .env",
        "./scripts/prepare-host.sh",
        "./scripts/generate-api-key.sh",
        "docker compose --env-file .env build",
        "/app/scripts/prefetch-models.py",
        "docker compose --env-file .env up -d",
        "docker compose --env-file .env ps",
        "docker compose --env-file .env logs",
        "./scripts/smoke.sh",
        "uv sync --frozen --extra dev",
        "uv run --frozen python scripts/benchmark.py",
        "docker compose --env-file .env down",
        "supertonic-3",
        "qwen3-tts-1.7b-custom",
        "qwen3-tts-1.7b-design",
        "qwen3-tts-1.7b-clone",
        "POST /v1/audio/voices",
        "GET /v1/audio/voices",
        "DELETE /v1/audio/voices/{voice_id}",
        "office",
        "AWS",
        "TLS",
        "consent_confirmed",
    )
    assert all(fragment in readme for fragment in required_fragments)
    assert not re.search(r"\b(?:pip(?:3)? install|poetry|conda)\b", readme, re.IGNORECASE)


def test_docs_cover_registered_and_ephemeral_clone_lifecycles() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    required_fragments = (
        "POST /v1/audio/speech/clone",
        "transcript",
        "speaker-embedding-only",
        "text-assisted",
        "x_vector_only_mode",
        "not persisted",
        "data volume",
        "/tmp",
        "after every exit path",
        "DELETE /v1/audio/voices/{voice_id}",
        "WAV or FLAC",
        "3 and 30 seconds",
        "20 MB",
        "2 channels",
        "192 kHz",
        "6,000,000 decoded samples",
    )
    assert all(fragment in readme for fragment in required_fragments)

    for example in (
        "Persistent registration, text-assisted",
        "Persistent registration, speaker-embedding-only",
        "Ephemeral clone, text-assisted",
        "Ephemeral clone, speaker-embedding-only",
    ):
        assert example in readme

    examples = {
        "Persistent registration, text-assisted": (
            "/v1/audio/voices",
            True,
        ),
        "Persistent registration, speaker-embedding-only": (
            "/v1/audio/voices",
            False,
        ),
        "Ephemeral clone, text-assisted": (
            "/v1/audio/speech/clone",
            True,
        ),
        "Ephemeral clone, speaker-embedding-only": (
            "/v1/audio/speech/clone",
            False,
        ),
    }
    for marker, (endpoint, includes_transcript) in examples.items():
        code = _fenced_code_after(readme, marker)
        assert f'"$TTS_BASE_URL{endpoint}"' in code
        assert ("-F 'transcript=" in code) is includes_transcript


def test_readme_documents_discoverable_ephemeral_clone_contract() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    required_fragments = (
        "/docs",
        "/openapi.json",
        "`input`, `audio`, and `consent_confirmed` are required",
        "`transcript`, `language`, and `response_format` are optional",
        '"details": [',
        '"field": "audio"',
        '"reason": "required"',
        "Field values and backend text are intentionally omitted",
    )
    assert all(fragment in readme for fragment in required_fragments)


def test_readme_enumerates_public_error_codes() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    public_error_codes = {
        "invalid_audio": "400",
        "unauthorized": "401",
        "not_found": "404",
        "voice_not_found": "404",
        "method_not_allowed": "405",
        "audio_too_large": "413",
        "request_too_large": "413",
        "validation_error": "422",
        "unsupported_language": "422",
        "consent_required": "422",
        "invalid_audio_duration": "422",
        "invalid_multipart": "422",
        "queue_full": "429",
        "internal_error": "500",
        "backend_not_configured": "503",
        "backend_unavailable": "503",
        "backend_invalid_response": "503",
        "worker_not_ready": "503",
        "insufficient_vram": "503",
        "model_load_failed": "503",
        "model_unavailable": "503",
        "cuda_out_of_memory": "503",
        "queue_timeout": "504",
        "backend_timeout": "504",
        "backend_error": "Upstream status",
        "http_error": "Other framework status",
    }
    for code, status in public_error_codes.items():
        row = rf"(?m)^\| {re.escape(status)} \| `{code}` \| \S.* \|$"
        assert re.search(row, readme), f"Missing documented mapping for {code}"

    consent_row = re.search(r"(?m)^\| 422 \| `consent_required` \| (?P<text>.*) \|$", readme)
    multipart_row = re.search(r"(?m)^\| 422 \| `invalid_multipart` \| (?P<text>.*) \|$", readme)
    assert consent_row is not None
    assert multipart_row is not None
    for row in (consent_row, multipart_row):
        description = row.group("text").lower()
        assert "registration" in description
        assert "ephemeral" in description


def test_readme_has_explicit_sudo_preparation_command() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    sudo_command_fragments = (
        "sudo env",
        'TTS_UID="$TTS_UID"',
        'TTS_GID="$TTS_GID"',
        'TTS_HF_CACHE_DIR="$TTS_HF_CACHE_DIR"',
        'TTS_SUPERTONIC_CACHE_DIR="$TTS_SUPERTONIC_CACHE_DIR"',
        'TTS_DATA_DIR="$TTS_DATA_DIR"',
        "./scripts/prepare-host.sh",
    )
    assert all(fragment in readme for fragment in sudo_command_fragments)


def test_license_notes_cover_qwen_and_both_supertonic_licenses() -> None:
    licenses = (ROOT / "LICENSES.md").read_text(encoding="utf-8")

    assert "Qwen" in licenses and "Apache-2.0" in licenses
    assert "Supertonic" in licenses and "MIT" in licenses
    assert "OpenRAIL-M" in licenses
    assert "https://" in licenses
