"""Factory for the DockerOperator tasks that run `claimfreq` in the trainer image (ADR-0003).

Every task:
- runs the trainer image by its fixed tag (``claimfreq-trainer:latest``);
- bind-mounts the host repo at /repo as the working directory, so code, params.yaml,
  dvc.lock, and .git come from the checked-out revision (docs/contracts.md, Images). Docker
  resolves mount sources on the host, so the path comes from HOST_REPO_DIR, not from inside
  the Airflow container;
- joins the compose network, so it reaches mlflow, postgres, and api by service name;
- gets only the variables its command uses, set explicitly (DockerOperator does not inherit
  the compose environment). Secrets go through ``private_environment``, which Airflow keeps
  out of the rendered-template view and the logs.

No credential is written in this file: the database password and the webhook URL are read
from the Airflow container's environment, which compose fills from .env.
"""

from __future__ import annotations

import os

from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount
from sqlalchemy.engine import URL

TRAINER_IMAGE = os.environ.get("TRAINER_IMAGE", "claimfreq-trainer:latest")
COMPOSE_NETWORK = os.environ.get("COMPOSE_NETWORK", "claim-frequency-mlops_default")
DOCKER_URL = "unix://var/run/docker.sock"


def _host_repo_dir() -> str:
    """Return the host path of the repo, failing loudly at DAG import if it is not set."""
    path = os.environ.get("HOST_REPO_DIR", "")
    if not path:
        raise RuntimeError("HOST_REPO_DIR is not set: add the repo's absolute path to .env")
    return path


# Compose values of the contract's variables (docs/contracts.md, Environment variables).
MLFLOW = {"MLFLOW_TRACKING_URI": "http://mlflow:5000"}
EVIDENTLY = {"EVIDENTLY_UI_URL": "http://evidently-ui:8000"}


def monitoring_secrets() -> dict[str, str]:
    """Return MONITORING_DB_URL and ALERT_WEBHOOK_URL, built from the Airflow environment.

    ``URL.create`` percent-encodes the credentials, so any password works in the URL.
    """
    url = URL.create(
        "postgresql+psycopg",
        username=os.environ.get("POSTGRES_USER", ""),
        password=os.environ.get("POSTGRES_PASSWORD", ""),
        host="postgres",
        port=5432,
        database="monitoring",
    )
    return {
        "MONITORING_DB_URL": url.render_as_string(hide_password=False),
        "ALERT_WEBHOOK_URL": os.environ.get("ALERT_WEBHOOK_URL", ""),
    }


def trainer_task(
    task_id: str,
    command: str,
    environment: dict[str, str] | None = None,
    private_environment: dict[str, str] | None = None,
) -> DockerOperator:
    """Return a task that runs ``command`` in the trainer image with the contract's wiring.

    A nonzero exit code fails the task: 3 means the command is still a stub, 10 that a check
    failed, and 1 an unexpected error (docs/contracts.md, CLI). Failed tasks are not retried.
    """
    return DockerOperator(
        task_id=task_id,
        image=TRAINER_IMAGE,
        command=command,
        docker_url=DOCKER_URL,
        network_mode=COMPOSE_NETWORK,
        mounts=[Mount(source=_host_repo_dir(), target="/repo", type="bind")],
        working_dir="/repo",
        environment=environment or {},
        private_environment=private_environment or {},
        # The default temp-dir mount refers to a path inside the Airflow container, which the
        # host's Docker daemon cannot see.
        mount_tmp_dir=False,
        auto_remove="success",
        retries=0,
    )
