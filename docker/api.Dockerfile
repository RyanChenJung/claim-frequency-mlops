# API image (Role D; docs/contracts.md, Images). It serves claim-frequency@champion with
# FastAPI. Unlike the trainer image, it is self-contained: the source is copied in at build
# time, and there is no bind mount. Rebuild it after any code or dependency change.
FROM python:3.12-slim

# libgomp1: the OpenMP runtime that LightGBM and XGBoost need to load a champion.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# uv pinned to the team's version (required-version in pyproject.toml).
COPY --from=ghcr.io/astral-sh/uv:0.13.9 /uv /uvx /usr/local/bin/

# Use the image's Python 3.12, never a downloaded one (pyproject.toml prefers uv-managed
# interpreters on laptops). The venv lives in /opt/venv, and its bin folder is on PATH.
# UV_NO_CACHE keeps uv's download cache (about 800 MB) out of the image.
ENV UV_PYTHON_PREFERENCE=only-system \
    UV_PYTHON_DOWNLOADS=never \
    UV_NO_CACHE=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Dependencies first, so this layer is cached until pyproject.toml or uv.lock changes.
# Only the base dependencies and the api group: no training or test tools.
COPY pyproject.toml uv.lock README.md .python-version ./
RUN uv sync --locked --no-default-groups --group api --no-install-project

# Then the project itself.
COPY src ./src
RUN uv sync --locked --no-default-groups --group api

EXPOSE 8000

# Inside the image, commands are bare (never `uv run`), as in docs/contracts.md.
CMD ["uvicorn", "claimfreq.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
