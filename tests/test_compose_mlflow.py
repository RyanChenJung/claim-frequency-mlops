"""Guard test for the compose ``mlflow`` service's job execution setting (#26).

MLflow 3.18's server starts a job runner and Huey consumer processes for server-side GenAI jobs
unless ``MLFLOW_SERVER_ENABLE_JOB_EXECUTION`` is false. The project uses none of those jobs, and
in the dry run on #26 they held 1.3-1.6 GiB of Docker Desktop's 8 GB, so the compose service
turns them off. CI never starts the compose stack, so this test checks the file itself.
"""

from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]


def test_mlflow_server_runs_without_job_execution():
    """The mlflow service sets MLFLOW_SERVER_ENABLE_JOB_EXECUTION to the string "false"."""
    services = yaml.safe_load((REPO / "docker-compose.yml").read_text())["services"]
    assert services["mlflow"]["environment"]["MLFLOW_SERVER_ENABLE_JOB_EXECUTION"] == "false"
