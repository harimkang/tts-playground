FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
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
RUN uv sync --frozen --no-dev --no-install-project \
    && uv pip install --target /tmp/hatchling-cache-warm \
        "hatchling==1.31.0" \
        "editables==0.5" \
    && rm -rf /tmp/hatchling-cache-warm

COPY src ./src
RUN uv sync --frozen --no-dev --offline

CMD ["uv", "run", "uvicorn", "tts_playground.gateway.app:app", "--host", "0.0.0.0", "--port", "8020"]
