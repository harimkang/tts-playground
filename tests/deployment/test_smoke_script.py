import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
SMOKE_SCRIPT = ROOT / "scripts" / "smoke.sh"


@pytest.mark.parametrize("missing", ["TTS_BASE_URL", "TTS_API_KEY"])
def test_smoke_requires_base_url_and_api_key(missing: str) -> None:
    environment = os.environ.copy()
    environment.update(
        {"TTS_BASE_URL": "http://tts.example", "TTS_API_KEY": "smoke-secret"}
    )
    environment.pop(missing)

    result = subprocess.run(
        ["/bin/sh", str(SMOKE_SCRIPT)],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert missing in result.stderr


def test_smoke_calls_public_endpoints_and_optional_qwen_without_printing_key(
    tmp_path: Path,
) -> None:
    curl_log = tmp_path / "curl.log"
    fake_curl = tmp_path / "curl"
    fake_curl.write_text(
        """#!/bin/sh
set -eu
printf '%s\\n' "$*" >> "$FAKE_CURL_LOG"
output=/dev/null
url=
while [ "$#" -gt 0 ]; do
    case "$1" in
        --output|-o)
            shift
            output=$1
            ;;
        http://*|https://*)
            url=$1
            ;;
    esac
    shift
done
case "$url" in
    */v1/audio/speech)
        printf 'RIFF0000WAVEfmt ' > "$output"
        dd if=/dev/zero bs=1 count=48 2>/dev/null >> "$output"
        ;;
    *)
        printf '{}' > "$output"
        ;;
esac
""",
        encoding="utf-8",
    )
    fake_curl.chmod(0o755)
    environment = os.environ.copy()
    environment.update(
        {
            "PATH": f"{tmp_path}:{environment['PATH']}",
            "FAKE_CURL_LOG": str(curl_log),
            "TTS_BASE_URL": "http://tts.example/",
            "TTS_API_KEY": "smoke-secret",
            "RUN_QWEN_SMOKE": "1",
        }
    )

    result = subprocess.run(
        ["/bin/sh", str(SMOKE_SCRIPT)],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    calls = curl_log.read_text(encoding="utf-8")
    assert "http://tts.example/healthz" in calls
    assert "http://tts.example/readyz" in calls
    assert "http://tts.example/v1/models" in calls
    assert calls.count("http://tts.example/v1/audio/speech") == 2
    assert "smoke-secret" not in result.stdout + result.stderr
