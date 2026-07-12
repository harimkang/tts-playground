import re
from pathlib import Path

ROOT = Path(__file__).parents[2]


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


def test_license_notes_cover_qwen_and_both_supertonic_licenses() -> None:
    licenses = (ROOT / "LICENSES.md").read_text(encoding="utf-8")

    assert "Qwen" in licenses and "Apache-2.0" in licenses
    assert "Supertonic" in licenses and "MIT" in licenses
    assert "OpenRAIL-M" in licenses
    assert "https://" in licenses
