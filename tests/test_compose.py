"""Guard tests for the ``api`` service and ``.dockerignore`` (#22; docs/contracts.md, Images).

CI never starts the compose stack, so these tests check the files themselves: the ``api`` service
is built from Role D's Dockerfile and published 8000→8000 on 127.0.0.1 only, its healthcheck fails
on the 503 that ``/health`` answers before a champion is loaded, it and the ``trainer`` service get
the contract's compose values, it waits for a healthy Postgres and MLflow, and it stays out of a
plain ``docker compose up``. ``.dockerignore`` must keep secrets out of every build context at any
depth, because the trainer and api Dockerfiles copy ``src/``.
"""

from pathlib import Path
from urllib.parse import urlsplit

import pytest
import yaml

from claimfreq.contracts import env

REPO = Path(__file__).resolve().parents[1]
SERVICES = yaml.safe_load((REPO / "docker-compose.yml").read_text())["services"]

# Compose values from docs/contracts.md (Environment variables).
COMPOSE_VALUES = {
    env.MLFLOW_TRACKING_URI: "http://mlflow:5000",
    env.MONITORING_DB_URL: (
        "postgresql+psycopg://${POSTGRES_USER}:${POSTGRES_PASSWORD}@postgres:5432/monitoring"
    ),
}


@pytest.fixture
def api() -> dict:
    """The ``api`` service as written in docker-compose.yml."""
    return SERVICES["api"]


def test_api_is_built_from_the_api_dockerfile_and_published_on_8000(api):
    """The contract's api service: docker/api.Dockerfile, container port 8000, local only."""
    assert api["build"] == {"context": ".", "dockerfile": "docker/api.Dockerfile"}
    assert api["ports"] == ["127.0.0.1:8000:8000"]


def test_api_stays_out_of_a_plain_compose_up(api):
    """The api service is in its own "api" profile, so a plain `docker compose up -d` skips it."""
    assert api["profiles"] == ["api"]


def test_api_and_trainer_get_the_compose_values(api):
    """The app reads only these two variables; the api and the trainer get their compose values."""
    assert set(api["environment"]) == set(COMPOSE_VALUES)
    for name, value in COMPOSE_VALUES.items():
        assert api["environment"][name] == value, name
        assert SERVICES["trainer"]["environment"][name] == value, name
    # send-traffic in the trainer service reaches the API by service name and container port.
    api_url = urlsplit(SERVICES["trainer"]["environment"][env.API_URL])
    assert (api_url.hostname, api_url.port) == ("api", 8000)


def test_api_waits_for_healthy_postgres_and_mlflow(api):
    """At startup the app exits if Postgres is down, and stalls in MLflow's retries if MLflow is."""
    assert api["depends_on"] == {
        "postgres": {"condition": "service_healthy"},
        "mlflow": {"condition": "service_healthy"},
    }


def test_api_healthcheck_fails_until_a_champion_is_loaded(api):
    """/health answers 503 before a champion is loaded, and urlopen raises on it (no curl -f)."""
    probe = (
        "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=3)"
    )
    assert api["healthcheck"]["test"] == ["CMD", "python", "-c", probe]


def test_dockerignore_keeps_secrets_out_at_any_depth():
    """Docker anchors .dockerignore patterns at the context root, so secrets need `**/`."""
    lines = (REPO / ".dockerignore").read_text().splitlines()
    assert {"**/.env", "**/*.env.local", "**/.dvc/config.local"} <= set(lines)
