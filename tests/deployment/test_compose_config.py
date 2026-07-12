import json
import subprocess
import tomllib
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def compose_config() -> dict[str, Any]:
    compose_file = ROOT / "compose.yaml"
    assert compose_file.is_file(), "compose.yaml does not exist"

    result = subprocess.run(
        [
            "docker",
            "compose",
            "--env-file",
            ".env.example",
            "config",
            "--format",
            "json",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def test_only_gateway_publishes_a_port(compose_config: dict[str, Any]) -> None:
    services = compose_config["services"]

    assert services["tts-gateway"]["ports"] == [
        {
            "mode": "ingress",
            "target": 8020,
            "published": "8020",
            "protocol": "tcp",
            "host_ip": "0.0.0.0",
        }
    ]
    assert "ports" not in services["qwen-worker"]
    assert "ports" not in services["supertonic-worker"]


def test_qwen_reserves_one_nvidia_gpu(compose_config: dict[str, Any]) -> None:
    devices = compose_config["services"]["qwen-worker"]["deploy"]["resources"][
        "reservations"
    ]["devices"]

    assert devices == [{"capabilities": ["gpu"], "count": 1, "driver": "nvidia"}]
    assert compose_config["services"]["qwen-worker"]["shm_size"] == "2147483648"


def test_networks_keep_workers_private_with_model_egress(
    compose_config: dict[str, Any],
) -> None:
    services = compose_config["services"]

    assert compose_config["networks"]["tts-private"]["internal"] is True
    assert set(services["tts-gateway"]["networks"]) == {"tts-private"}
    assert set(services["qwen-worker"]["networks"]) == {"tts-private", "model-egress"}
    assert set(services["supertonic-worker"]["networks"]) == {
        "tts-private",
        "model-egress",
    }


def test_gateway_consumes_api_key_as_a_file_secret(
    compose_config: dict[str, Any],
) -> None:
    gateway = compose_config["services"]["tts-gateway"]

    assert compose_config["secrets"]["tts_api_key"]["file"].endswith(
        "/secrets/tts_api_key.txt"
    )
    assert gateway["secrets"] == [
        {
            "source": "tts_api_key",
            "target": "/run/secrets/tts_api_key",
        }
    ]
    assert gateway["environment"]["TTS_API_KEY_FILE"] == "/run/secrets/tts_api_key"


def test_qwen_mounts_model_cache_and_voice_data(compose_config: dict[str, Any]) -> None:
    volumes = compose_config["services"]["qwen-worker"]["volumes"]
    writable_bind_mounts = {
        (volume["source"], volume["target"])
        for volume in volumes
        if volume["type"] == "bind" and volume.get("read_only", False) is False
    }

    assert ("/srv/tts-cache/huggingface", "/cache/huggingface") in writable_bind_mounts
    assert ("/srv/tts-data", "/data") in writable_bind_mounts


def test_services_apply_runtime_hardening(compose_config: dict[str, Any]) -> None:
    for service in compose_config["services"].values():
        assert service["restart"] == "unless-stopped"
        assert service["init"] is True
        assert service["read_only"] is True
        assert service["user"] == "1007:1007"
        assert "/tmp" in service["tmpfs"]
        assert service.get("privileged", False) is False
        assert service.get("network_mode") != "host"
        assert all(
            volume.get("source") != "/var/run/docker.sock"
            and volume.get("target") != "/var/run/docker.sock"
            for volume in service.get("volumes", [])
        )


def test_healthchecks_use_python_and_gateway_waits_for_workers(
    compose_config: dict[str, Any],
) -> None:
    services = compose_config["services"]

    for service_name in ("tts-gateway", "qwen-worker", "supertonic-worker"):
        command = " ".join(services[service_name]["healthcheck"]["test"])
        assert "python" in command
        assert "urllib.request.urlopen" in command

    assert "/healthz" in " ".join(services["tts-gateway"]["healthcheck"]["test"])
    assert "/healthz" in " ".join(services["qwen-worker"]["healthcheck"]["test"])
    assert "/v1/styles" in " ".join(
        services["supertonic-worker"]["healthcheck"]["test"]
    )
    assert services["tts-gateway"]["depends_on"] == {
        "qwen-worker": {"condition": "service_healthy", "required": True},
        "supertonic-worker": {"condition": "service_healthy", "required": True},
    }


def test_uv_images_pin_and_cache_the_project_build_backend_before_offline_sync() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert project["build-system"]["requires"] == ["hatchling==1.31.0"]

    expected_steps = (
        "COPY pyproject.toml uv.lock ./",
        "uv sync --frozen --no-dev",
        "uv pip install --target /tmp/hatchling-cache-warm",
        '"hatchling==1.31.0"',
        '"editables==0.5"',
        "COPY src ./src",
        "uv sync --frozen --no-dev",
        "--offline",
    )
    for dockerfile_name in ("gateway.Dockerfile", "qwen.Dockerfile"):
        dockerfile = (ROOT / "docker" / dockerfile_name).read_text(encoding="utf-8")
        cursor = 0
        for step in expected_steps:
            cursor = dockerfile.index(step, cursor) + len(step)


def test_python_images_use_uv_managed_frozen_environments() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert project["project"]["optional-dependencies"]["supertonic"] == [
        "supertonic[serve]==1.3.1"
    ]

    dockerfiles = {
        path.name: path.read_text(encoding="utf-8")
        for path in sorted((ROOT / "docker").glob("*.Dockerfile"))
    }
    assert set(dockerfiles) == {
        "gateway.Dockerfile",
        "qwen.Dockerfile",
        "supertonic.Dockerfile",
    }
    for dockerfile in dockerfiles.values():
        assert "python -m pip" not in dockerfile
        assert "COPY --from=ghcr.io/astral-sh/uv:0.11.14" in dockerfile
        assert "COPY pyproject.toml uv.lock ./" in dockerfile
        assert "uv sync --frozen --no-dev" in dockerfile
        assert ".venv" in dockerfile

    assert "uv sync --frozen --no-dev --extra qwen" in dockerfiles["qwen.Dockerfile"]
    assert (
        "uv sync --frozen --no-dev --extra supertonic --no-install-project"
        in dockerfiles["supertonic.Dockerfile"]
    )
