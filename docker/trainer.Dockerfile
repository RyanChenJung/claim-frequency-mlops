# Trainer image (Role C; docs/contracts.md, Images and Runtime). Every Airflow DockerOperator
# task and `docker compose run --rm trainer claimfreq <command>` run in it. At run time the
# host repo is bind-mounted at /repo, so the code, params.yaml, dvc.lock, and .git all come
# from the checked-out revision; rebuild the image only when dependencies change.
FROM python:3.12-slim

# libgomp1: the OpenMP runtime LightGBM and XGBoost need. git: `git rev-parse HEAD` for the
# git_commit lineage tag. The bind-mounted repo belongs to another user, so git must trust it.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 git \
    && rm -rf /var/lib/apt/lists/* \
    && git config --system --add safe.directory '*'

# uv pinned to the team's version (required-version in pyproject.toml).
COPY --from=ghcr.io/astral-sh/uv:0.13.9 /uv /uvx /usr/local/bin/

# Use the image's Python 3.12, never a downloaded one (pyproject.toml prefers uv-managed
# interpreters on laptops). The venv lives outside /repo, so the bind mount cannot hide it.
ENV UV_PYTHON_PREFERENCE=only-system \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /repo

# Dependencies first, so this layer is cached until pyproject.toml or uv.lock changes.
COPY pyproject.toml uv.lock README.md .python-version ./
RUN uv sync --locked --no-default-groups --group trainer --no-install-project

# Then the project itself, installed in editable mode: /opt/venv points at /repo/src, which
# the bind mount replaces with the host checkout at run time.
COPY src ./src
RUN uv sync --locked --no-default-groups --group trainer

# Inside the image, commands are bare `claimfreq ...`, never `uv run` (docs/contracts.md).
CMD ["claimfreq", "--help"]
