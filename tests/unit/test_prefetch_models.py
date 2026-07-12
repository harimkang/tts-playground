import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tts_playground.common.models import QWEN_REPOSITORIES


def _load_prefetch_module() -> object:
    path = Path(__file__).parents[2] / "scripts" / "prefetch-models.py"
    spec = importlib.util.spec_from_file_location("tts_prefetch_models", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load prefetch module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


prefetch = _load_prefetch_module()


def test_prefetch_rejects_cache_with_less_than_twenty_gib_free(tmp_path: Path) -> None:
    downloads: list[str] = []

    with pytest.raises(RuntimeError, match="20 GiB"):
        prefetch.prefetch_models(
            cache_directory=tmp_path,
            snapshot_download=lambda **kwargs: downloads.append(kwargs["repo_id"]),
            disk_usage=lambda _path: SimpleNamespace(free=20 * 1024**3 - 1),
        )

    assert downloads == []


def test_prefetch_downloads_exact_qwen_repositories_without_printing_token(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    def download(**kwargs: Any) -> str:
        calls.append(kwargs)
        return str(tmp_path / "models" / kwargs["repo_id"].replace("/", "--"))

    token = "hf_never_print_this_secret"
    monkeypatch.setenv("HF_TOKEN", token)
    paths = prefetch.prefetch_models(
        cache_directory=tmp_path,
        snapshot_download=download,
        disk_usage=lambda _path: SimpleNamespace(free=20 * 1024**3),
    )

    assert [call["repo_id"] for call in calls] == list(QWEN_REPOSITORIES.values())
    assert all(call == {"repo_id": call["repo_id"], "cache_dir": str(tmp_path)} for call in calls)
    assert len(paths) == 3
    output = capsys.readouterr().out
    assert all(repository in output for repository in QWEN_REPOSITORIES.values())
    assert all(path in output for path in paths)
    assert token not in output
