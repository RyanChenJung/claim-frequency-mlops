"""Shared fixtures for the Role B tests (``train``, ``register``, ``promote``)."""

import mlflow
import pytest
from mlflow.tracking import MlflowClient

from claimfreq.contracts import paths
from claimfreq.contracts.model import MLFLOW_EXPERIMENT_NAME
from claimfreq.contracts.params import Params
from claimfreq.fakes import make_policies

# Value of GIT_COMMIT in the `workspace` fixture, so tests can check the git_commit tag.
FAKE_COMMIT = "0123456789abcdef0123456789abcdef01234567"

# The fast, deterministic candidate set of most tests: the baseline and the GLM. The tree
# candidates (LightGBM, a timed FLAML search) have their own tests in test_tree_candidates.py.
FAST_CANDIDATES = ["constant", "poisson_glm"]
FAST_TRAINING = {"flaml_time_budget_s": 2, "flaml_max_iter": 8, "candidates": FAST_CANDIDATES}


def fast_params(**training: object) -> Params:
    """Return Params with a fast training section; keyword arguments override its keys."""
    return Params.model_validate({"training": {**FAST_TRAINING, **training}})


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """Fake splits under a temporary DATA_DIR, and a temporary MLflow store and experiment."""
    data_dir = tmp_path / "data"
    (data_dir / "splits").mkdir(parents=True)
    make_policies(8_000, seed=1, id_start=1).to_parquet(data_dir / paths.TRAIN, index=False)
    make_policies(3_000, seed=2, id_start=500_000).to_parquet(
        data_dir / paths.VALIDATION, index=False
    )
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setenv("DATA_DIR", str(data_dir))
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    monkeypatch.setenv("GIT_COMMIT", FAKE_COMMIT)
    # `train` and `register` called without params (as the CLI does) load these, never the
    # shipped params.yaml with its 300 s FLAML budget. The imported names are patched, since
    # load_params binds its default path at import time.
    monkeypatch.setattr("claimfreq.training.load_params", fast_params)
    monkeypatch.setattr("claimfreq.registry.load_params", fast_params)
    # Create the experiment up front with its artifacts under tmp_path; with SQLite, MLflow
    # would otherwise put them in ./mlruns, inside the repository.
    mlflow.set_tracking_uri(tracking_uri)
    MlflowClient().create_experiment(
        MLFLOW_EXPERIMENT_NAME, artifact_location=(tmp_path / "artifacts").as_uri()
    )
    yield data_dir
    mlflow.set_tracking_uri(None)
