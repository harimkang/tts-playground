from pathlib import Path
from types import ModuleType

import pytest
from fastapi import FastAPI

from tts_playground.common.settings import GatewaySettings
from tts_playground.gateway import app as gateway_module


def test_gateway_module_exposes_an_asgi_app() -> None:
    assert isinstance(gateway_module, ModuleType)
    assert isinstance(gateway_module.app, FastAPI)


@pytest.mark.asyncio
async def test_runtime_gateway_owns_backend_client_for_its_lifespan(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    clients: list[FakeAsyncClient] = []
    routers: list[FakeBackendRouter] = []

    def create_client() -> FakeAsyncClient:
        client = FakeAsyncClient()
        clients.append(client)
        return client

    def create_router(
        settings: GatewaySettings,
        client: FakeAsyncClient,
    ) -> FakeBackendRouter:
        router = FakeBackendRouter(settings, client)
        routers.append(router)
        return router

    monkeypatch.setattr(gateway_module.httpx, "AsyncClient", create_client)
    monkeypatch.setattr(gateway_module, "BackendRouter", create_router)

    key_file = tmp_path / "tts-api-key"
    key_file.write_text("runtime-secret\n", encoding="utf-8")
    runtime_app = gateway_module.create_runtime_gateway_app(
        GatewaySettings(api_key_file=key_file)
    )

    assert clients == []
    assert routers == []

    async with runtime_app.router.lifespan_context(runtime_app):
        assert len(clients) == 1
        assert len(routers) == 1
        assert clients[0].entered is True
        assert clients[0].closed is False
        assert routers[0].client is clients[0]
        assert runtime_app.state.backend_router is routers[0]

    assert clients[0].closed is True
    assert runtime_app.state.backend_router is None


class FakeAsyncClient:
    def __init__(self) -> None:
        self.entered = False
        self.closed = False

    async def __aenter__(self) -> "FakeAsyncClient":
        self.entered = True
        return self

    async def __aexit__(
        self,
        _exc_type: object,
        _exc_value: object,
        _traceback: object,
    ) -> None:
        self.closed = True


class FakeBackendRouter:
    def __init__(self, settings: GatewaySettings, client: FakeAsyncClient) -> None:
        self.settings = settings
        self.client = client
