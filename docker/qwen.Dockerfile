FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/cache/huggingface \
    HOME=/tmp \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_NO_SYNC=1 \
    VIRTUAL_ENV=/app/.venv \
    PATH=/app/.venv/bin:$PATH

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        ca-certificates \
        libsndfile1 \
        python3.12 \
        python3.12-venv \
        sox \
    && ln -s /usr/bin/python3.12 /usr/local/bin/python \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.11.14 /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --extra build --extra qwen --no-install-project

COPY src ./src
RUN uv sync --frozen --no-dev --extra build --extra qwen --no-build-isolation --offline

COPY scripts/prefetch-models.py ./scripts/prefetch-models.py

CMD ["uv", "run", "uvicorn", "tts_playground.qwen.app:app", "--host", "0.0.0.0", "--port", "8001"]
