"""Tests for ``claimfreq evaluate-test``: the champion's ``test_*`` metrics on its own run.

They use the ``workspace`` fixture from ``conftest.py`` (fake train, validation, and test
splits; a temporary SQLite MLflow store) and the real ``train``, ``register``, and ``promote``.
"""

import mlflow
import numpy as np
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient

from claimfreq import metrics
from claimfreq.cli import EXIT_CHECK_FAILED, EXIT_ERROR, EXIT_OK, main
from claimfreq.contracts import paths
from claimfreq.contracts.model import (
    ALIAS_CHAMPION,
    REGISTERED_MODEL_NAME,
    TAG_SCHEMA_HASH,
    MetricName,
    MetricPrefix,
    file_md5,
    split_xy_w,
)
from claimfreq.contracts.schema import coerce_features, coerce_frame
from claimfreq.evaluation import TAG_TEST_DATA_HASH, evaluate_test
from claimfreq.registry import RegistryCheckError

VAL_KEYS = {f"{MetricPrefix.VAL}{name}" for name in MetricName}
TEST_KEYS = {f"{MetricPrefix.TEST}{name}" for name in MetricName}


@pytest.fixture
def champion(workspace):
    """Train, register, and promote version 1; return the client and that version."""
    assert main(["train"]) == EXIT_OK
    assert main(["register"]) == EXIT_OK
    assert main(["promote", "--version", "1"]) == EXIT_OK
    client = MlflowClient()
    return client, client.get_model_version_by_alias(REGISTERED_MODEL_NAME, ALIAS_CHAMPION)


def test_logs_the_champions_test_metrics_on_its_run(workspace, champion):
    client, version = champion
    result = evaluate_test()
    assert result.version == str(version.version) == "1"
    assert result.run_id == version.run_id

    run = client.get_run(version.run_id)
    # The run keeps its val_* metrics and gains exactly the three test_* metrics.
    assert set(run.data.metrics) == VAL_KEYS | TEST_KEYS
    assert set(result.metrics) == TEST_KEYS
    assert run.data.tags[TAG_TEST_DATA_HASH] == file_md5(workspace / paths.TEST)

    # The values are the champion version's own predictions on the test split.
    X, y, w = split_xy_w(coerce_frame(pd.read_parquet(workspace / paths.TEST)))
    model = mlflow.pyfunc.load_model(f"models:/{REGISTERED_MODEL_NAME}/{version.version}")
    expected = metrics.with_prefix(
        metrics.compute_all(y, np.asarray(model.predict(coerce_features(X)), float), w),
        MetricPrefix.TEST,
    )
    for key, value in expected.items():
        assert run.data.metrics[key] == pytest.approx(value)
        assert result.metrics[key] == pytest.approx(value)
    # Measured on another split than the val_* metrics, which stay as train logged them.
    name = MetricName.POISSON_DEVIANCE
    val = client.get_run(version.run_id).data.metrics[f"{MetricPrefix.VAL}{name}"]
    assert run.data.metrics[f"{MetricPrefix.TEST}{name}"] != pytest.approx(val)
    assert len(client.get_metric_history(version.run_id, f"{MetricPrefix.VAL}{name}")) == 1


def test_scores_the_champion_not_the_newer_challenger(workspace, champion):
    client, version = champion
    # A second candidate set: its best child becomes version 2, the challenger, unpromoted.
    assert main(["train"]) == EXIT_OK
    assert main(["register"]) == EXIT_OK
    challenger = client.get_model_version(REGISTERED_MODEL_NAME, "2")
    result = evaluate_test()
    assert result.run_id == version.run_id != challenger.run_id
    assert not TEST_KEYS & set(client.get_run(challenger.run_id).data.metrics)


def test_rerun_appends_a_step(workspace, champion):
    client, version = champion
    first = evaluate_test()
    second = evaluate_test()
    assert first.metrics == second.metrics
    key = f"{MetricPrefix.TEST}{MetricName.GINI}"
    assert len(client.get_metric_history(version.run_id, key)) == 2


def test_cli_prints_the_version_and_metrics(workspace, champion, capsys):
    capsys.readouterr()
    assert main(["evaluate-test"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "champion version 1 (1.0.0)" in out
    assert all(key in out for key in TEST_KEYS)


def test_exits_10_without_a_registered_model(workspace, capsys):
    assert main(["evaluate-test"]) == EXIT_CHECK_FAILED
    assert "no champion" in capsys.readouterr().err


def test_exits_10_when_versions_exist_but_none_is_champion(workspace, capsys):
    assert main(["train"]) == EXIT_OK
    assert main(["register"]) == EXIT_OK
    assert main(["evaluate-test"]) == EXIT_CHECK_FAILED
    assert "no champion" in capsys.readouterr().err


def test_refuses_a_champion_of_another_schema(workspace, champion):
    client, version = champion
    client.set_model_version_tag(REGISTERED_MODEL_NAME, version.version, TAG_SCHEMA_HASH, "old")
    with pytest.raises(RegistryCheckError, match="another feature schema"):
        evaluate_test()
    assert not TEST_KEYS & set(client.get_run(version.run_id).data.metrics)


def test_exits_10_on_a_broken_test_split(workspace, champion):
    broken = pd.read_parquet(workspace / paths.TEST).drop(columns=["DrivAge"])
    broken.to_parquet(workspace / paths.TEST, index=False)
    assert main(["evaluate-test"]) == EXIT_CHECK_FAILED


def test_exits_1_without_a_test_split(workspace, champion):
    (workspace / paths.TEST).unlink()
    assert main(["evaluate-test"]) == EXIT_ERROR


def test_warns_when_the_test_split_has_no_claims(workspace, champion, caplog):
    test = pd.read_parquet(workspace / paths.TEST).assign(ClaimNb=0)
    test.to_parquet(workspace / paths.TEST, index=False)
    with caplog.at_level("WARNING"):
        result = evaluate_test()
    assert "no claims" in caplog.text
    assert np.isnan(result.metrics[f"{MetricPrefix.TEST}{MetricName.GINI}"])
