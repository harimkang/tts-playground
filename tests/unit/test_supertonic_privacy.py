import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_sitecustomize_redacts_error_message_arguments_and_traceback() -> None:
    marker = "PRIVATE_SUPERTONIC_EXCEPTION_MARKER"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "docker" / "supertonic_privacy")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import logging; "
            "logging.basicConfig(); "
            "logger = logging.getLogger('supertonic.test'); "
            f"logger.error('private args %s', '{marker}'); "
            "\ntry:\n"
            f"    raise RuntimeError('{marker}')\n"
            "except RuntimeError:\n"
            f"    logger.exception('private traceback {marker}')\n",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 0
    assert marker not in result.stderr
    assert result.stderr.count("Supertonic request failed.") == 2
