"""Tests for ``claimfreq train``: the run tree, lineage tags, metrics, and logged models.

Every test logs to a temporary SQLite MLflow store whose artifacts also live under
``tmp_path``, so nothing is written into the repository.
"""

import os

import mlflow
import numpy as np
import pandas as pd
import pytest
from conftest import fast_params
from mlflow.tracking import MlflowClient

from claimfreq.cli import EXIT_CHECK_FAILED, EXIT_ERROR, EXIT_OK, main
from claimfreq.contracts import paths
from claimfreq.contracts.model import (
    MLFLOW_EXPERIMENT_NAME,
    MODEL_ARTIFACT_PATH,
    RUN_TAGS,
    TAG_ALGORITHM,
    TAG_CANDIDATE_SET,
    TAG_DATA_HASH,
    TAG_GIT_COMMIT,
    TAG_SCHEMA_HASH,
    MetricName,
    MetricPrefix,
    file_md5,
)
from claimfreq.contracts.params import Params
from claimfreq.contracts.schema import coerce_features, schema_hash
from claimfreq.fakes import make_policies
from claimfreq.training import (
    ALGORITHM_CONSTANT,
    ALGORITHM_POISSON_GLM,
    PARAM_SEED,
    build_candidates,
    fit_candidate,
    train,
)

VAL_KEYS = {f"{MetricPrefix.VAL}{name}" for name in MetricName}


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
    assert len(children) == len(build_candidates(fast_params()))


def test_every_child_has_lineage_params_and_val_metrics(workspace):
    parent_id = train()
    expected_lineage = {
        TAG_GIT_COMMIT: os.environ["GIT_COMMIT"],
        TAG_DATA_HASH: file_md5(workspace / paths.TRAIN),
        TAG_SCHEMA_HASH: schema_hash(),
    }
    seed = fast_params().training.seed
    for run in _children(MlflowClient(), parent_id):
        tags = run.data.tags
        assert set(RUN_TAGS) <= set(tags)
        assert {key: tags[key] for key in RUN_TAGS} == expected_lineage
        assert tags[TAG_CANDIDATE_SET] == parent_id
        assert run.data.params[PARAM_SEED] == str(seed)
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
        model = mlflow.pyfunc.load_model(f"runs:/{run.info.run_id}/{MODEL_ARTIFACT_PATH}")
        predictions = np.asarray(model.predict(X), dtype=float)
        assert predictions.shape == (len(X),)
        assert (predictions > 0).all()


def test_fit_routes_exposure_weights():
    policies = make_policies(500, seed=4)
    X = coerce_features(policies)
    y = policies["ClaimNb"].to_numpy(float) / policies["Exposure"].to_numpy(float)
    w = policies["Exposure"].to_numpy(float)
    constant = build_candidates(fast_params())[0]
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


def test_logged_model_requirements_are_runtime_only(workspace):
    children = _children(MlflowClient(), train())
    for run in children:
        local = mlflow.artifacts.download_artifacts(
            f"runs:/{run.info.run_id}/{MODEL_ARTIFACT_PATH}/requirements.txt"
        )
        with open(local, encoding="utf-8") as f:
            names = {line.split("==")[0].strip() for line in f if line.strip()}
        assert {"scikit-learn", "skops", "numpy", "pandas"} <= names
        assert not names & {"fastapi", "psycopg", "pytest", "sqlalchemy"}


def test_glm_stays_bounded_at_the_extremes_the_api_accepts():
    policies = make_policies(8_000, seed=1)
    X = coerce_features(policies)
    y = policies["ClaimNb"].to_numpy(float) / policies["Exposure"].to_numpy(float)
    w = policies["Exposure"].to_numpy(float)
    glm = fit_candidate(build_candidates(fast_params())[1], X, y, w)
    # BonusMalus 350 and Density 50,000 are allowed by the API but far above the fake data.
    extreme = X.head(1).assign(BonusMalus=350, Density=50_000, DrivAge=18)
    prediction = float(glm.predict(coerce_features(extreme))[0])
    assert 0 < prediction < 5  # claims per policy-year; a linear BonusMalus term exceeds this


def test_drivage_bands_follow_params():
    params = Params.model_validate({"training": {"glm": {"drivage_edges": [25, 50]}}})
    glm = build_candidates(params)[1]
    assert glm.params["glm.drivage_edges"] == [25, 50]
    banding = glm.pipeline.named_steps["preprocess"].transformers[0][1].named_steps["band"]
    assert banding.kw_args == {"bins": [25, 50]}
