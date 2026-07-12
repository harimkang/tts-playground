FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/tmp \
    UV_CACHE_DIR=/tmp/uv-cache \
    UV_LINK_MODE=copy \
    UV_NO_SYNC=1 \
    VIRTUAL_ENV=/app/.venv \
    PATH=/app/.venv/bin:$PATH

RUN apt-get update \
    && apt-get install --yes --no-install-recommends ca-certificates libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.11.14 /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --extra build --no-install-project

COPY src ./src
RUN uv sync --frozen --no-dev --extra build --no-build-isolation --offline

CMD ["uv", "run", "--no-sync", "uvicorn", "tts_playground.gateway.app:app", "--host", "0.0.0.0", "--port", "8020"]
