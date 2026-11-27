"""Guard test for the trainer image's environment (docker/trainer.Dockerfile).

stdout is block-buffered when it is not a terminal, so without ``PYTHONUNBUFFERED`` the Airflow
task log shows stderr lines before stdout lines that were printed ahead of them. CI builds the
image but never runs it, so this test checks the file itself.
"""

from pathlib import Path

DOCKERFILE = Path(__file__).resolve().parents[1] / "docker" / "trainer.Dockerfile"


def test_trainer_image_runs_python_unbuffered():
    """The trainer image sets PYTHONUNBUFFERED=1."""
    assert "ENV PYTHONUNBUFFERED=1" in DOCKERFILE.read_text().splitlines()
