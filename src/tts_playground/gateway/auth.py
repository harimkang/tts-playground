import secrets
from collections.abc import Callable
from pathlib import Path
from typing import Annotated

from fastapi import Header

from tts_playground.common.errors import ServiceError


def read_api_key(api_key_file: Path) -> str:
    api_key = api_key_file.read_text(encoding="utf-8").strip()
    if not api_key:
        raise ValueError("API key file must not be empty")
    return api_key


def require_api_key(expected_api_key: str) -> Callable[[str | None], None]:
    def authenticate(
        authorization: Annotated[str | None, Header(alias="Authorization")] = None,
    ) -> None:
        if authorization is None:
            raise _unauthorized()

        parts = authorization.split()
        if len(parts) != 2 or parts[0].lower() != "bearer":
            raise _unauthorized()

        if not secrets.compare_digest(parts[1], expected_api_key):
            raise _unauthorized()

    return authenticate


def _unauthorized() -> ServiceError:
    return ServiceError(
        status_code=401,
        code="unauthorized",
        message="Invalid or missing API key.",
    )
