import os
import re
import shutil
import stat
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_prepare_host_preserves_quoted_directory_paths(tmp_path: Path) -> None:
    directories = (
        tmp_path / "hf cache",
        tmp_path / "super[cache]",
        tmp_path / "data * cache",
    )
    environment = os.environ | {
        "TTS_UID": str(os.getuid()),
        "TTS_GID": str(os.getgid()),
        "TTS_HF_CACHE_DIR": str(directories[0]),
        "TTS_SUPERTONIC_CACHE_DIR": str(directories[1]),
        "TTS_DATA_DIR": str(directories[2]),
    }

    subprocess.run(
        [str(ROOT / "scripts" / "prepare-host.sh")],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    for directory in directories:
        assert directory.is_dir()
        assert stat.S_IMODE(directory.stat().st_mode) == 0o750
        assert (directory.stat().st_uid, directory.stat().st_gid) == (
            os.getuid(),
            os.getgid(),
        )


def test_api_key_creation_is_atomic_and_exclusive(tmp_path: Path) -> None:
    scripts_directory = tmp_path / "scripts"
    secrets_directory = tmp_path / "secrets"
    scripts_directory.mkdir()
    secrets_directory.mkdir()
    script = scripts_directory / "generate-api-key.sh"
    shutil.copy2(ROOT / "scripts" / "generate-api-key.sh", script)
    source = script.read_text(encoding="utf-8")

    assert "mktemp" in source
    assert 'ln "$temporary_key_file" "$key_file"' in source
    assert 'openssl rand -hex 32 > "$key_file"' not in source

    processes = [
        subprocess.Popen(
            [str(script)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _ in range(8)
    ]
    results = [process.communicate(timeout=10) + (process.returncode,) for process in processes]

    assert sum(return_code == 0 for _stdout, _stderr, return_code in results) == 1
    key_file = secrets_directory / "tts_api_key.txt"
    assert re.fullmatch(r"[0-9a-f]{64}\n", key_file.read_text(encoding="ascii"))
    assert stat.S_IMODE(key_file.stat().st_mode) == 0o600
