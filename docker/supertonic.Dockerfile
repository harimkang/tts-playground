FROM python:3.12-slim

ENV HOME=/home/app \
    PATH=/app/.venv/bin:$PATH \
    PYTHONPATH=/app/privacy \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy

RUN apt-get update \
    && apt-get install --yes --no-install-recommends ca-certificates libsndfile1 \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /home/app/.cache/supertonic3

COPY --from=ghcr.io/astral-sh/uv:0.11.14 /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --extra supertonic --no-install-project

COPY docker/supertonic_privacy/sitecustomize.py /app/privacy/sitecustomize.py

CMD ["supertonic", "serve", "--host", "0.0.0.0", "--port", "7788"]
