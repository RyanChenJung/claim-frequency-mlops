"""Tests for ``claimfreq train``: the run tree, lineage tags, metrics, and logged models.

Every test logs to a temporary SQLite MLflow store whose artifacts also live under
``tmp_path``, so nothing is written into the repository.
"""

import mlflow
import numpy as np
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient

from claimfreq.cli import EXIT_CHECK_FAILED, EXIT_ERROR, EXIT_OK, main
from claimfreq.contracts import paths
from claimfreq.contracts.model import (
    MLFLOW_EXPERIMENT_NAME,
    RUN_TAGS,
    TAG_ALGORITHM,
    TAG_DATA_HASH,
    TAG_GIT_COMMIT,
    TAG_SCHEMA_HASH,
    MetricName,
    MetricPrefix,
    file_md5,
)
from claimfreq.contracts.params import load_params
from claimfreq.contracts.schema import coerce_features, schema_hash
from claimfreq.fakes import make_policies
from claimfreq.training import (
    ALGORITHM_CONSTANT,
    ALGORITHM_POISSON_GLM,
    MODEL_ARTIFACT,
    TAG_CANDIDATE_SET,
    build_candidates,
    fit_candidate,
    train,
)

FAKE_COMMIT = "0123456789abcdef0123456789abcdef01234567"
VAL_KEYS = {f"{MetricPrefix.VAL}{name}" for name in MetricName}


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
    # Create the experiment up front with its artifacts under tmp_path; with SQLite, MLflow
    # would otherwise put them in ./mlruns, inside the repository.
    mlflow.set_tracking_uri(tracking_uri)
    MlflowClient().create_experiment(
        MLFLOW_EXPERIMENT_NAME, artifact_location=(tmp_path / "artifacts").as_uri()
    )
    yield data_dir
    mlflow.set_tracking_uri(None)


def _children(client: MlflowClient, parent_id: str) -> list:
    experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    return client.search_runs(
        [experiment.experiment_id], filter_string=f"tags.mlflow.parentRunId = '{parent_id}'"
    )


def test_train_logs_one_parent_with_a_child_per_candidate(workspace):
    parent_id = train()
    client = MlflowClient()
    parent = client.get_run(parent_id)
    assert parent.data.tags[TAG_CANDIDATE_SET] == parent_id

    children = _children(client, parent_id)
    algorithms = {run.data.tags[TAG_ALGORITHM] for run in children}
    assert algorithms == {ALGORITHM_CONSTANT, ALGORITHM_POISSON_GLM}
    assert len(children) == len(build_candidates(seed=0))


def test_every_child_has_lineage_params_and_val_metrics(workspace):
    parent_id = train()
    expected_lineage = {
        TAG_GIT_COMMIT: FAKE_COMMIT,
        TAG_DATA_HASH: file_md5(workspace / paths.TRAIN),
        TAG_SCHEMA_HASH: schema_hash(),
    }
    seed = load_params().training.seed
    for run in _children(MlflowClient(), parent_id):
        tags = run.data.tags
        assert set(RUN_TAGS) <= set(tags)
        assert {key: tags[key] for key in RUN_TAGS} == expected_lineage
        assert tags[TAG_CANDIDATE_SET] == parent_id
        assert run.data.params["training.seed"] == str(seed)
        assert set(run.data.metrics) == VAL_KEYS
        assert all(np.isfinite(value) for value in run.data.metrics.values())


def test_glm_beats_the_constant_baseline(workspace):
    children = _children(MlflowClient(), train())
    by_algorithm = {run.data.tags[TAG_ALGORITHM]: run.data.metrics for run in children}
    deviance = f"{MetricPrefix.VAL}{MetricName.POISSON_DEVIANCE}"
    gini = f"{MetricPrefix.VAL}{MetricName.GINI}"
    constant = by_algorithm[ALGORITHM_CONSTANT]
    glm = by_algorithm[ALGORITHM_POISSON_GLM]
    assert glm[deviance] < constant[deviance]
    assert glm[gini] > constant[gini]
    assert constant[gini] == pytest.approx(0.0)


def test_logged_models_load_as_pyfunc_and_predict_positive_frequencies(workspace):
    children = _children(MlflowClient(), train())
    X = coerce_features(make_policies(50, seed=9))
    for run in children:
        model = mlflow.pyfunc.load_model(f"runs:/{run.info.run_id}/{MODEL_ARTIFACT}")
        predictions = np.asarray(model.predict(X), dtype=float)
        assert predictions.shape == (len(X),)
        assert (predictions > 0).all()


def test_fit_routes_exposure_weights():
    policies = make_policies(500, seed=4)
    X = coerce_features(policies)
    y = policies["ClaimNb"].to_numpy(float) / policies["Exposure"].to_numpy(float)
    w = policies["Exposure"].to_numpy(float)
    constant = build_candidates(seed=0)[0]
    fitted = fit_candidate(constant, X, y, w)
    # The weighted mean of the frequency is total claims / total exposure.
    expected = policies["ClaimNb"].sum() / policies["Exposure"].sum()
    assert fitted.predict(X)[0] == pytest.approx(expected)


def test_cli_train_exits_0(workspace, capsys):
    assert main(["train"]) == EXIT_OK
    assert "candidate set" in capsys.readouterr().out


def test_cli_train_exits_10_on_a_schema_violation(workspace, capsys):
    broken = pd.read_parquet(workspace / paths.TRAIN).drop(columns=["DrivAge"])
    broken.to_parquet(workspace / paths.TRAIN, index=False)
    assert main(["train"]) == EXIT_CHECK_FAILED


def test_cli_train_exits_1_without_a_train_split(workspace):
    (workspace / paths.TRAIN).unlink()
    assert main(["train"]) == EXIT_ERROR
