"""Tests for ``claimfreq register`` and ``claimfreq promote`` (ADR-0007).

They use the ``workspace`` fixture from ``conftest.py``: fake splits and a temporary SQLite
MLflow store with artifacts under ``tmp_path``.
"""

import mlflow
import pandas as pd
import pytest
from mlflow.entities import RunStatus
from mlflow.tracking import MlflowClient

from claimfreq.cli import EXIT_CHECK_FAILED, EXIT_OK, main
from claimfreq.contracts import paths
from claimfreq.contracts.model import (
    ALIAS_CHALLENGER,
    ALIAS_CHAMPION,
    MLFLOW_EXPERIMENT_NAME,
    MODEL_VERSION_TAGS,
    REFERENCE_ARTIFACT_PATH,
    REFERENCE_COLUMNS,
    REGISTERED_MODEL_NAME,
    TAG_ALGORITHM,
    TAG_CANDIDATE_SET,
    TAG_DATA_HASH,
    TAG_SCHEMA_HASH,
    TAG_SELECTION_METRIC,
    TAG_SEMVER,
    MetricName,
)
from claimfreq.contracts.params import Params
from claimfreq.fakes import make_policies
from claimfreq.registry import (
    CHAMPION_METRIC_PREFIX,
    NO_CHAMPION,
    NO_CHAMPION_COMPARISON,
    TAG_BEATS_CHAMPION,
    TAG_CHAMPION_VERSION,
    next_semver,
    parse_semver,
    promote,
    register,
)
from claimfreq.training import ALGORITHM_POISSON_GLM, train

LINEAGE = {TAG_SCHEMA_HASH: "s1", TAG_DATA_HASH: "d1", TAG_ALGORITHM: "poisson_glm"}


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


def test_versions_without_a_champion_bump_minor():
    assert next_semver(LINEAGE, None, "1.0.0") == "1.1.0"


@pytest.mark.parametrize("value", ["1.0", "1.0.0.0", "v1.0.0", "1.x.0", ""])
def test_parse_semver_rejects_malformed_versions(value):
    with pytest.raises(ValueError):
        parse_semver(value)


# --- register -----------------------------------------------------------------------------------


def _client() -> MlflowClient:
    return MlflowClient()


def test_first_register_creates_challenger_1_0_0_with_every_tag(workspace):
    train()
    result = register()
    assert (result.version, result.semver, result.already_registered) == ("1", "1.0.0", False)
    # The GLM beats the constant baseline on the fake data, so it is selected.
    assert result.algorithm == ALGORITHM_POISSON_GLM

    client = _client()
    version = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, ALIAS_CHALLENGER)
    assert str(version.version) == "1"
    assert set(MODEL_VERSION_TAGS) <= set(version.tags)
    assert version.tags[TAG_SEMVER] == "1.0.0"
    assert version.tags[TAG_SELECTION_METRIC] == MetricName.POISSON_DEVIANCE
    assert version.run_id == result.run_id

    run = client.get_run(result.run_id)
    assert run.data.tags[TAG_CHAMPION_VERSION] == NO_CHAMPION
    assert run.data.tags[TAG_BEATS_CHAMPION] == NO_CHAMPION_COMPARISON


def test_register_logs_the_reference_dataset_on_the_models_run(workspace):
    train()
    result = register()
    local = mlflow.artifacts.download_artifacts(f"runs:/{result.run_id}/{REFERENCE_ARTIFACT_PATH}")
    reference = pd.read_parquet(local)
    validation = pd.read_parquet(workspace / paths.VALIDATION)
    assert list(reference.columns) == REFERENCE_COLUMNS
    assert len(reference) == len(validation)
    assert (reference["predicted_frequency"] > 0).all()


def test_registered_model_loads_by_alias(workspace):
    train()
    register()
    model = mlflow.pyfunc.load_model(f"models:/{REGISTERED_MODEL_NAME}@{ALIAS_CHALLENGER}")
    from claimfreq.contracts.schema import coerce_features

    assert (model.predict(coerce_features(make_policies(10, seed=5))) > 0).all()


def test_register_twice_registers_nothing_new(workspace, capsys):
    train()
    assert main(["register"]) == EXIT_OK
    assert main(["register"]) == EXIT_OK
    assert "already version 1" in capsys.readouterr().out
    versions = _client().search_model_versions(f"name = '{REGISTERED_MODEL_NAME}'")
    assert len(versions) == 1


def test_retrain_on_the_same_data_is_a_patch_and_compares_with_the_champion(workspace):
    train()
    register()
    promote(1)
    train(Params.model_validate({"training": {"glm": {"alpha": 0.001}}}))
    result = register()
    assert result.semver == "1.0.1"

    run = _client().get_run(result.run_id)
    assert run.data.tags[TAG_CHAMPION_VERSION] == "1"
    assert run.data.tags[TAG_BEATS_CHAMPION] in {"true", "false"}
    champion_keys = {key for key in run.data.metrics if key.startswith(CHAMPION_METRIC_PREFIX)}
    assert champion_keys == {f"{CHAMPION_METRIC_PREFIX}val_{name}" for name in MetricName}


def test_retrain_on_new_data_is_a_minor(workspace):
    train()
    register()
    promote(1)
    make_policies(8_000, seed=11, id_start=1).to_parquet(workspace / paths.TRAIN, index=False)
    train()
    assert register().semver == "1.1.0"


def test_register_skips_a_failed_train_run(workspace):
    parent_id = train()
    client = _client()
    experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    # A newer train parent whose candidates crashed: it ends FAILED and must be ignored.
    failed = client.create_run(experiment.experiment_id, tags={TAG_CANDIDATE_SET: "x"})
    client.set_terminated(failed.info.run_id, RunStatus.to_string(RunStatus.FAILED))
    result = register()
    assert client.get_run(result.run_id).data.tags["mlflow.parentRunId"] == parent_id


def test_register_without_a_train_run_exits_10(workspace, capsys):
    assert main(["register"]) == EXIT_CHECK_FAILED
    assert "no FINISHED train run" in capsys.readouterr().err


def test_register_on_other_data_than_the_train_run_exits_10(workspace, capsys):
    train()
    make_policies(8_000, seed=12, id_start=1).to_parquet(workspace / paths.TRAIN, index=False)
    assert main(["register"]) == EXIT_CHECK_FAILED
    assert "check out the matching data" in capsys.readouterr().err


# --- promote ------------------------------------------------------------------------------------


def test_promote_moves_the_champion_alias(workspace, capsys):
    train()
    register()
    assert main(["promote", "--version", "1"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "champion is now version 1 (1.0.0); was no champion" in out
    champion = _client().get_model_version_by_alias(REGISTERED_MODEL_NAME, ALIAS_CHAMPION)
    assert str(champion.version) == "1"


def test_promote_reports_the_previous_champion(workspace):
    train()
    register()
    promote(1)
    train(Params.model_validate({"training": {"glm": {"alpha": 0.001}}}))
    register()
    target, previous = promote(2)
    assert (str(target.version), str(previous.version)) == ("2", "1")


def test_promote_a_missing_version_exits_10(workspace, capsys):
    train()
    register()
    assert main(["promote", "--version", "7"]) == EXIT_CHECK_FAILED
    assert "no version 7" in capsys.readouterr().err


def test_promote_before_any_registration_exits_10(workspace, capsys):
    assert main(["promote", "--version", "1"]) == EXIT_CHECK_FAILED
