"""Environment contract: the variables every service and CLI command reads, with defaults.

Defaults target local development on the host, outside docker compose. Containers get every
variable set explicitly to its compose value (for example ``http://mlflow:5000``), except
``DATA_DIR`` and ``GIT_COMMIT``, which stay unset (``GIT_COMMIT`` is only a manual override;
the trainer image runs ``git rev-parse HEAD`` instead). DockerOperator tasks do not inherit
the compose ``environment:``, so each task sets the variables of its command itself;
docs/contracts.md lists them per command, with their compose values. Secrets and
machine-specific values go in ``.env``, which git ignores.
"""

import os
import subprocess

# Root of the data tree; paths.py resolves every data file under it. It stays unset (meaning
# "data" under the repo root) in every Airflow task and compose service; only development and
# tests set it.
DATA_DIR = "DATA_DIR"
# MLflow tracking server and model registry.
MLFLOW_TRACKING_URI = "MLFLOW_TRACKING_URI"
# SQLAlchemy URL of the monitoring database (logs.py). In compose this is the `monitoring`
# database on the shared Postgres server:
# postgresql+psycopg://<user>:<password>@postgres:5432/monitoring
MONITORING_DB_URL = "MONITORING_DB_URL"
# Base URL of the prediction API, used by `send-traffic`.
API_URL = "API_URL"
# Evidently UI, to which `monitor` pushes snapshots.
EVIDENTLY_UI_URL = "EVIDENTLY_UI_URL"
# Optional webhook (Discord on the demo machine). When unset, alerts only go to the log.
ALERT_WEBHOOK_URL = "ALERT_WEBHOOK_URL"
# Optional commit hash for the `git_commit` lineage tag; see git_commit().
GIT_COMMIT = "GIT_COMMIT"

DEFAULTS: dict[str, str | None] = {
    DATA_DIR: "data",
    # 5001 rather than MLflow's own 5000, which macOS AirPlay Receiver occupies. Compose
    # publishes 5001 -> 5000; containers use http://mlflow:5000.
    MLFLOW_TRACKING_URI: "http://localhost:5001",
    # A local SQLite file lets the API and the monitor run without Postgres.
    MONITORING_DB_URL: "sqlite:///monitoring.db",
    API_URL: "http://localhost:8000",
    # 8001 rather than Evidently's own default 8000, which the API already uses on localhost.
    EVIDENTLY_UI_URL: "http://localhost:8001",
    ALERT_WEBHOOK_URL: None,
    GIT_COMMIT: None,
}


def get(name: str) -> str | None:
    """Return the value of environment variable ``name``, or its documented default.

    An empty string counts as set, so ``ALERT_WEBHOOK_URL=`` in ``.env`` is not None; check
    truthiness when deciding whether to post an alert.
    """
    return os.environ.get(name, DEFAULTS[name])


def git_commit() -> str:
    """Return the commit for the ``git_commit`` lineage tag.

    ``GIT_COMMIT`` wins when it is set and not empty. Otherwise ``git rev-parse HEAD`` runs in
    the working directory, which is the repo root (bind-mounted in the trainer container). The
    trainer image has git and trusts the mounted repo, so DAG tasks do not set ``GIT_COMMIT``;
    it is only an override.

    Raises:
        RuntimeError: if neither gives a commit. A run without lineage must not be logged, so
            the command fails with exit code 1.
    """
    commit = get(GIT_COMMIT)
    if commit:
        return commit
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(
            "cannot determine git_commit: GIT_COMMIT is unset and `git rev-parse HEAD` failed "
            "in the working directory"
        ) from error
    return result.stdout.strip()
