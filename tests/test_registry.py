"""Tests for ``claimfreq register`` and ``claimfreq promote`` (ADR-0007).

They use the ``workspace`` fixture from ``conftest.py``: fake splits and a temporary SQLite
MLflow store with artifacts under ``tmp_path``. Assertions check values, not only keys, so a
wrong comparison, reference, or tag fails a test.
"""

import os
from types import SimpleNamespace

import mlflow
import numpy as np
import pandas as pd
import pytest
from mlflow.entities import RunStatus
from mlflow.tracking import MlflowClient

from claimfreq import metrics
from claimfreq.cli import EXIT_CHECK_FAILED, EXIT_OK, main
from claimfreq.contracts import paths
from claimfreq.contracts.model import (
    ALIAS_CHALLENGER,
    ALIAS_CHAMPION,
    MLFLOW_EXPERIMENT_NAME,
    MODEL_VERSION_TAGS,
    PREDICTED_FREQUENCY,
    REFERENCE_ARTIFACT_PATH,
    REFERENCE_COLUMNS,
    REGISTERED_MODEL_NAME,
    TAG_ALGORITHM,
    TAG_CANDIDATE_SET,
    TAG_DATA_HASH,
    TAG_GIT_COMMIT,
    TAG_SCHEMA_HASH,
    TAG_SELECTION_METRIC,
    TAG_SELECTION_METRIC_VALUE,
    TAG_SEMVER,
    MetricName,
    MetricPrefix,
    file_md5,
    is_better,
    split_xy_w,
)
from claimfreq.contracts.params import Params
from claimfreq.contracts.schema import coerce_features, coerce_frame, schema_hash
from claimfreq.fakes import make_policies
from claimfreq.registry import (
    CHAMPION_TAG_PREFIX,
    INCOMPARABLE,
    NO_CHAMPION,
    NO_CHAMPION_COMPARISON,
    TAG_BEATS_CHAMPION,
    TAG_CHAMPION_SEMVER,
    TAG_CHAMPION_VERSION,
    RegistryCheckError,
    next_semver,
    parse_semver,
    promote,
    register,
    select_best,
)
from claimfreq.training import ALGORITHM_POISSON_GLM, train

LINEAGE = {TAG_SCHEMA_HASH: "s1", TAG_DATA_HASH: "d1", TAG_ALGORITHM: "poisson_glm"}
DEVIANCE = MetricName.POISSON_DEVIANCE.value
VAL_DEVIANCE = f"{MetricPrefix.VAL}{DEVIANCE}"
# A params change that keeps data and algorithm, so the next version is a PATCH.
PATCH_PARAMS = Params.model_validate({"training": {"glm": {"alpha": 0.001}}})


def _client() -> MlflowClient:
    return MlflowClient()


def _version(number: str):
    return _client().get_model_version(REGISTERED_MODEL_NAME, number)


def _alias(alias: str):
    return _client().get_model_version_by_alias(REGISTERED_MODEL_NAME, alias)


def _validation(workspace) -> pd.DataFrame:
    return coerce_frame(pd.read_parquet(workspace / paths.VALIDATION))


def _predict(model_uri: str, frame: pd.DataFrame) -> np.ndarray:
    X, _, _ = split_xy_w(frame)
    model = mlflow.pyfunc.load_model(model_uri)
    return np.asarray(model.predict(coerce_features(X)), dtype=float)


# --- Semantic versions (pure functions) ---------------------------------------------------------


def test_first_version_is_1_0_0():
    assert next_semver(LINEAGE, None, None) == "1.0.0"


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        ({TAG_SCHEMA_HASH: "s2"}, "2.0.0"),  # input contract changed
        ({TAG_SCHEMA_HASH: "s2", TAG_DATA_HASH: "d2"}, "2.0.0"),  # MAJOR wins
        ({TAG_DATA_HASH: "d2"}, "1.3.0"),  # new data version
        ({TAG_ALGORITHM: "lightgbm"}, "1.3.0"),  # new algorithm
        ({}, "1.2.4"),  # same schema, data, and algorithm
    ],
)
def test_bump_against_the_champion(change, expected):
    champion = {**LINEAGE, TAG_SEMVER: "1.2.3"}
    assert next_semver({**LINEAGE, **change}, champion, "1.2.3") == expected


def test_bump_starts_from_the_highest_registered_version():
    # Champion 1.0.0; an unpromoted challenger already took 1.1.0, so the next MINOR is 1.2.0.
    champion = {**LINEAGE, TAG_SEMVER: "1.0.0"}
    assert next_semver({**LINEAGE, TAG_DATA_HASH: "d2"}, champion, "1.1.0") == "1.2.0"


def test_versions_without_a_champion_bump_minor_even_on_a_schema_change():
    assert next_semver(LINEAGE, None, "1.0.0") == "1.1.0"
    assert next_semver({**LINEAGE, TAG_SCHEMA_HASH: "s2"}, None, "1.0.0") == "1.1.0"


@pytest.mark.parametrize("value", ["1.0", "1.0.0.0", "v1.0.0", "1.x.0", ""])
def test_parse_semver_rejects_malformed_versions(value):
    with pytest.raises(ValueError):
        parse_semver(value)


# --- Selection (pure function) ------------------------------------------------------------------


def _child(name: str, **metric_values: float) -> SimpleNamespace:
    """A stand-in for an MLflow Run: select_best only reads ``data.metrics``."""
    return SimpleNamespace(name=name, data=SimpleNamespace(metrics=metric_values))


def _run_metrics(value: float | None) -> dict[str, float]:
    return {} if value is None else {VAL_DEVIANCE: value}


@pytest.mark.parametrize(
    ("values", "winner"),
    [
        ([0.5, 0.4, 0.45], "c1"),  # lowest deviance
        ([0.4, 0.4], "c0"),  # a tie keeps the earlier child (the baseline is logged first)
        ([float("nan"), 0.6], "c1"),  # NaN never wins
        ([0.6, float("nan")], "c0"),  # ... in either position
        ([None, 0.6], "c1"),  # a child without the metric is skipped
    ],
)
def test_select_best(values, winner):
    children = [_child(f"c{i}", **_run_metrics(v)) for i, v in enumerate(values)]
    assert select_best(children, DEVIANCE).name == winner


@pytest.mark.parametrize("values", [[], [None], [float("nan"), None]])
def test_select_best_without_a_scored_candidate_raises(values):
    children = [_child(f"c{i}", **_run_metrics(v)) for i, v in enumerate(values)]
    with pytest.raises(RegistryCheckError, match="no candidate"):
        select_best(children, DEVIANCE)


def test_register_without_a_scored_candidate_exits_10(workspace, capsys):
    client = _client()
    experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    parent = client.create_run(experiment.experiment_id)
    client.set_tag(parent.info.run_id, TAG_CANDIDATE_SET, parent.info.run_id)
    child = client.create_run(
        experiment.experiment_id, tags={"mlflow.parentRunId": parent.info.run_id}
    )
    client.set_terminated(child.info.run_id)
    client.set_terminated(parent.info.run_id)
    assert main(["register"]) == EXIT_CHECK_FAILED
    assert "no candidate" in capsys.readouterr().err


# --- register: the first version ----------------------------------------------------------------


def test_first_register_creates_challenger_1_0_0_with_correct_tags(workspace):
    parent_id = train()
    result = register()
    assert (result.version, result.semver, result.already_registered) == ("1", "1.0.0", False)
    assert result.candidate_set == parent_id
    # The GLM beats the constant baseline on the fake data, so it is selected.
    assert result.algorithm == ALGORITHM_POISSON_GLM

    version = _alias(ALIAS_CHALLENGER)
    assert str(version.version) == "1"
    assert version.run_id == result.run_id
    assert set(MODEL_VERSION_TAGS) <= set(version.tags)
    run = _client().get_run(result.run_id)
    assert version.tags == {
        **version.tags,
        TAG_SEMVER: "1.0.0",
        TAG_GIT_COMMIT: os.environ["GIT_COMMIT"],
        TAG_DATA_HASH: file_md5(workspace / paths.TRAIN),
        TAG_SCHEMA_HASH: schema_hash(),
        TAG_ALGORITHM: ALGORITHM_POISSON_GLM,
        TAG_SELECTION_METRIC: DEVIANCE,
        TAG_SELECTION_METRIC_VALUE: repr(run.data.metrics[VAL_DEVIANCE]),
    }


def test_first_register_logs_no_champion_comparison(workspace):
    train()
    run = _client().get_run(register().run_id)
    assert run.data.tags[TAG_CHAMPION_VERSION] == NO_CHAMPION
    assert run.data.tags[TAG_CHAMPION_SEMVER] == NO_CHAMPION
    assert run.data.tags[TAG_BEATS_CHAMPION] == NO_CHAMPION_COMPARISON


def test_reference_dataset_holds_the_models_predictions(workspace):
    train()
    result = register()
    local = mlflow.artifacts.download_artifacts(f"runs:/{result.run_id}/{REFERENCE_ARTIFACT_PATH}")
    reference = pd.read_parquet(local)
    validation = _validation(workspace)
    assert list(reference.columns) == REFERENCE_COLUMNS
    pd.testing.assert_frame_equal(
        reference.drop(columns=PREDICTED_FREQUENCY),
        validation[REFERENCE_COLUMNS[:-1]],
        check_dtype=False,
    )
    expected = _predict(f"models:/{REGISTERED_MODEL_NAME}/{result.version}", validation)
    np.testing.assert_allclose(reference[PREDICTED_FREQUENCY].to_numpy(), expected)
    # Predictions vary by policy: a constant would mean the wrong model was used.
    assert reference[PREDICTED_FREQUENCY].nunique() > 1


def test_registered_model_loads_by_alias(workspace):
    train()
    register()
    model = mlflow.pyfunc.load_model(f"models:/{REGISTERED_MODEL_NAME}@{ALIAS_CHALLENGER}")
    assert (model.predict(coerce_features(make_policies(10, seed=5))) > 0).all()


# --- register: idempotence and retries ----------------------------------------------------------


def test_register_twice_registers_nothing_new(workspace, capsys):
    train()
    assert main(["register"]) == EXIT_OK
    assert main(["register"]) == EXIT_OK
    assert "already version 1" in capsys.readouterr().out
    versions = _client().search_model_versions(f"name = '{REGISTERED_MODEL_NAME}'")
    assert len(versions) == 1


def test_a_retry_restores_a_challenger_alias_that_failed(workspace, monkeypatch):
    train()
    real = MlflowClient.set_registered_model_alias
    calls = {"n": 0}

    def fail_once(self, name, alias, version):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("tracking server hiccup")
        return real(self, name, alias, version)

    monkeypatch.setattr(MlflowClient, "set_registered_model_alias", fail_once)
    with pytest.raises(RuntimeError):
        register()
    retry = register()
    assert retry.already_registered
    assert str(_alias(ALIAS_CHALLENGER).version) == retry.version == "1"


def test_rerun_for_a_registered_run_is_a_noop_even_on_other_data(workspace):
    # The idempotence check runs before the data check (docs/contracts.md, Exit codes).
    train()
    register()
    make_policies(8_000, seed=12, id_start=1).to_parquet(workspace / paths.TRAIN, index=False)
    assert main(["register"]) == EXIT_OK


def test_register_on_other_data_than_the_train_run_exits_10(workspace, capsys):
    train()
    make_policies(8_000, seed=12, id_start=1).to_parquet(workspace / paths.TRAIN, index=False)
    assert main(["register"]) == EXIT_CHECK_FAILED
    assert "check out the matching data" in capsys.readouterr().err


def test_register_without_a_train_run_exits_10(workspace, capsys):
    assert main(["register"]) == EXIT_CHECK_FAILED
    assert "no FINISHED train run" in capsys.readouterr().err


def test_register_falls_back_past_a_failed_train_run_and_says_so(workspace, capsys):
    parent_id = train()
    client = _client()
    experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    # A newer train parent whose candidates crashed: it ends FAILED and must be skipped.
    failed = client.create_run(experiment.experiment_id, tags={TAG_CANDIDATE_SET: "x"})
    client.set_terminated(failed.info.run_id, RunStatus.to_string(RunStatus.FAILED))
    assert main(["register"]) == EXIT_OK
    out = capsys.readouterr().out
    assert f"falling back to candidate set {parent_id}" in out
    run_id = _alias(ALIAS_CHALLENGER).run_id
    assert client.get_run(run_id).data.tags["mlflow.parentRunId"] == parent_id


# --- register: comparison with the champion -----------------------------------------------------


def test_patch_retrain_compares_with_the_champion_on_the_same_holdout(workspace):
    train()
    register()
    promote(1)
    train(PATCH_PARAMS)
    result = register()
    assert result.semver == "1.0.1"

    run = _client().get_run(result.run_id)
    tags = run.data.tags
    assert tags[TAG_CHAMPION_VERSION] == "1"
    assert tags[TAG_CHAMPION_SEMVER] == "1.0.0"
    validation = _validation(workspace)
    _, y, w = split_xy_w(validation)
    champion_scores = metrics.compute_all(
        y, _predict(f"models:/{REGISTERED_MODEL_NAME}/1", validation), w
    )
    for name, value in champion_scores.items():
        assert float(tags[f"{CHAMPION_TAG_PREFIX}{MetricPrefix.VAL}{name}"]) == pytest.approx(value)
    beats = is_better(DEVIANCE, run.data.metrics[VAL_DEVIANCE], champion_scores[DEVIANCE])
    assert tags[TAG_BEATS_CHAMPION] == str(beats).lower()
    # The comparison is tags only: the run's metrics stay the model's own (see /model-info).
    assert all(key.startswith(MetricPrefix.VAL) for key in run.data.metrics)


def test_a_worse_challenger_does_not_beat_the_champion(workspace):
    train()
    register()
    promote(1)
    # A heavy penalty flattens the GLM towards the constant: worse validation deviance.
    train(Params.model_validate({"training": {"glm": {"alpha": 100.0}}}))
    run = _client().get_run(register().run_id)
    assert run.data.tags[TAG_BEATS_CHAMPION] == "false"


def test_retrain_on_new_data_is_a_minor(workspace):
    train()
    register()
    promote(1)
    make_policies(8_000, seed=11, id_start=1).to_parquet(workspace / paths.TRAIN, index=False)
    train()
    assert register().semver == "1.1.0"


def test_a_schema_change_is_major_and_incomparable(workspace):
    train()
    register()
    promote(1)
    # Pretend the champion was trained under another schema version.
    _client().set_model_version_tag(REGISTERED_MODEL_NAME, "1", TAG_SCHEMA_HASH, "old-schema")
    train(PATCH_PARAMS)
    result = register()
    assert result.semver == "2.0.0"
    tags = _client().get_run(result.run_id).data.tags
    assert tags[TAG_BEATS_CHAMPION] == INCOMPARABLE
    assert not any(key.startswith(f"{CHAMPION_TAG_PREFIX}{MetricPrefix.VAL}") for key in tags)


# --- promote ------------------------------------------------------------------------------------


def test_promote_moves_champion_and_clears_challenger(workspace, capsys):
    train()
    register()
    assert main(["promote", "--version", "1"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "champion is now version 1 (1.0.0); was no champion; challenger alias cleared" in out
    assert str(_alias(ALIAS_CHAMPION).version) == "1"
    assert _version("1").aliases == [ALIAS_CHAMPION]


def test_promote_keeps_a_challenger_that_points_elsewhere(workspace):
    train()
    register()
    promote(1)
    train(PATCH_PARAMS)
    register()  # version 2 is the challenger
    train(Params.model_validate({"training": {"glm": {"alpha": 0.01}}}))
    register()  # version 3 is the challenger
    result = promote(2)
    assert (str(result.target.version), str(result.previous.version)) == ("2", "1")
    assert not result.cleared_challenger
    assert str(_alias(ALIAS_CHALLENGER).version) == "3"


def test_a_retry_never_puts_challenger_on_the_champion(workspace):
    train()
    register()
    promote(1)
    assert register().already_registered
    assert _version("1").aliases == [ALIAS_CHAMPION]


def test_promote_a_missing_version_exits_10(workspace, capsys):
    train()
    register()
    assert main(["promote", "--version", "7"]) == EXIT_CHECK_FAILED
    assert "no version 7" in capsys.readouterr().err


def test_promote_before_any_registration_exits_10(workspace):
    assert main(["promote", "--version", "1"]) == EXIT_CHECK_FAILED
