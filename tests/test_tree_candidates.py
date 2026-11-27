"""Tests for the tree candidates of ``train``: LightGBM and the FLAML search (ADR-0006).

The FLAML tests run a real search with a 2-second budget and at most 8 trials, so they check
what FLAML actually logged; ``test_flaml_sees_only_the_train_split`` replaces the search with a
recorder instead, to check its inputs exactly.
"""

import subprocess
import sys

import mlflow
import numpy as np
import pandas as pd
import pytest
from conftest import fast_params
from lightgbm import LGBMRegressor
from mlflow.tracking import MlflowClient
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from claimfreq import automl
from claimfreq.cli import EXIT_OK, main
from claimfreq.contracts import paths
from claimfreq.contracts.model import (
    MLFLOW_EXPERIMENT_NAME,
    TAG_ALGORITHM,
    TAG_CANDIDATE_SET,
    MetricName,
    MetricPrefix,
    split_xy_w,
)
from claimfreq.contracts.schema import coerce_features
from claimfreq.fakes import make_policies
from claimfreq.registry import logged_model_uri
from claimfreq.training import (
    ALGORITHM_FLAML_PREFIX,
    ALGORITHM_LIGHTGBM,
    CANDIDATE_FLAML,
    TAG_FLAML_BEST_LEARNER,
    TAG_FLAML_TRIALS,
    FlamlCandidate,
    build_tree_preprocess,
    model_requirements,
    train,
)

VAL_KEYS = {f"{MetricPrefix.VAL}{name}" for name in MetricName}
DEVIANCE = f"{MetricPrefix.VAL}{MetricName.POISSON_DEVIANCE}"
# The library estimator each FLAML learner refits to, and its Poisson setting.
LEARNER_ESTIMATORS = {
    "lgbm": (LGBMRegressor, "objective", "poisson"),
    "xgboost": (XGBRegressor, "objective", "count:poisson"),
    "xgb_limitdepth": (XGBRegressor, "objective", "count:poisson"),
    "histgb": (HistGradientBoostingRegressor, "loss", "poisson"),
}


def _children(client: MlflowClient, parent_id: str) -> dict[str, object]:
    """Return the direct children of ``parent_id`` by run name."""
    experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    runs = client.search_runs(
        [experiment.experiment_id], filter_string=f"tags.mlflow.parentRunId = '{parent_id}'"
    )
    return {run.info.run_name: run for run in runs}


def _load_pipeline(client: MlflowClient, run) -> Pipeline:
    return mlflow.sklearn.load_model(logged_model_uri(client, run))


def _requirements(client: MlflowClient, run) -> set[str]:
    model_id = logged_model_uri(client, run).removeprefix("models:/")
    location = mlflow.get_logged_model(model_id).artifact_location
    local = mlflow.artifacts.download_artifacts(artifact_uri=f"{location}/requirements.txt")
    with open(local, encoding="utf-8") as f:
        return {line.split("==")[0].strip() for line in f if line.strip()}


# --- LightGBM -----------------------------------------------------------------------------------


@pytest.fixture
def lightgbm_run(workspace):
    client = MlflowClient()
    parent_id = train(fast_params(candidates=["constant", "lightgbm"]))
    return client, _children(client, parent_id)


def test_lightgbm_is_a_poisson_model_that_beats_the_baseline(lightgbm_run):
    client, children = lightgbm_run
    run = children[ALGORITHM_LIGHTGBM]
    assert run.data.tags[TAG_ALGORITHM] == ALGORITHM_LIGHTGBM
    assert set(run.data.metrics) == VAL_KEYS
    assert run.data.metrics[DEVIANCE] < children["constant"].data.metrics[DEVIANCE]

    estimator = _load_pipeline(client, run)[-1]
    assert isinstance(estimator, LGBMRegressor)
    assert estimator.get_params()["objective"] == "poisson"
    assert estimator.get_params()["n_estimators"] == fast_params().training.lightgbm.n_estimators
    assert run.data.params["lightgbm.min_child_samples"] == "500"


def test_lightgbm_model_lists_lightgbm_but_not_flaml(lightgbm_run):
    client, children = lightgbm_run
    requirements = _requirements(client, children[ALGORITHM_LIGHTGBM])
    assert "lightgbm" in requirements
    assert not requirements & {"flaml", "xgboost"}
    assert "lightgbm" not in _requirements(client, children["constant"])


def test_lightgbm_predicts_unseen_categories(lightgbm_run):
    client, children = lightgbm_run
    model = mlflow.pyfunc.load_model(logged_model_uri(client, children[ALGORITHM_LIGHTGBM]))
    X = coerce_features(make_policies(20, seed=3))
    predictions = np.asarray(model.predict(X), dtype=float)
    assert (predictions > 0).all()


# --- FLAML: a real, short search ----------------------------------------------------------------


@pytest.fixture
def flaml_run(workspace):
    client = MlflowClient()
    parent_id = train(fast_params(candidates=["constant", "flaml"]))
    children = _children(client, parent_id)
    flaml_child = children[CANDIDATE_FLAML]
    trials = _children(client, flaml_child.info.run_id)
    return client, parent_id, children, flaml_child, trials


def test_flaml_child_is_a_flaml_free_poisson_pipeline(flaml_run):
    client, _, _, child, _ = flaml_run
    learner = child.data.tags[TAG_FLAML_BEST_LEARNER]
    assert learner in automl.LEARNERS
    assert child.data.tags[TAG_ALGORITHM] == f"{ALGORITHM_FLAML_PREFIX}{learner}"
    assert set(child.data.metrics) == VAL_KEYS

    estimator = _load_pipeline(client, child)[-1]
    estimator_class, key, poisson = LEARNER_ESTIMATORS[learner]
    assert type(estimator) is estimator_class
    assert estimator.get_params()[key] == poisson
    assert not type(estimator).__module__.startswith("flaml")

    requirements = _requirements(client, child)
    assert "flaml" not in requirements
    library = type(estimator).__module__.split(".")[0]
    assert (library in requirements) == (library in {"lightgbm", "xgboost"})


def test_flaml_trials_are_nested_under_the_flaml_child_only(flaml_run):
    _, parent_id, children, child, trials = flaml_run
    # The parent's children are exactly the candidates: no trial sits directly under it.
    assert set(children) == {"constant", CANDIDATE_FLAML}
    assert len(trials) == int(child.data.tags[TAG_FLAML_TRIALS]) >= 1
    for trial in trials.values():
        assert trial.data.tags["mlflow.parentRunId"] == child.info.run_id
        # Trials are not candidates: register searches train parents by this tag.
        assert TAG_CANDIDATE_SET not in trial.data.tags
        assert not any(key.startswith(MetricPrefix.VAL) for key in trial.data.metrics)
    assert parent_id not in {trial.data.tags["mlflow.parentRunId"] for trial in trials.values()}


def test_every_flaml_trial_used_a_poisson_objective(flaml_run):
    # custom_hp silently skips a learner missing from estimator_list (ADR-0006 gotcha).
    *_, trials = flaml_run
    for trial in trials.values():
        learner = trial.data.tags[automl.TAG_FLAML_LEARNER]
        _, key, poisson = LEARNER_ESTIMATORS[learner]
        assert trial.data.params[f"{automl.TRIAL_CONFIG_PREFIX}{key}"] == poisson
        assert trial.data.tags[TAG_ALGORITHM] == f"{ALGORITHM_FLAML_PREFIX}{learner}"


def test_full_candidate_set_trains_and_registers(workspace, capsys):
    client = MlflowClient()
    parent_id = train(fast_params(candidates=["constant", "poisson_glm", "lightgbm", "flaml"]))
    assert set(_children(client, parent_id)) == {
        "constant",
        "poisson_glm",
        ALGORITHM_LIGHTGBM,
        CANDIDATE_FLAML,
    }
    assert main(["register"]) == EXIT_OK
    assert "is the challenger" in capsys.readouterr().out


# --- FLAML: the search's inputs, with the search replaced ---------------------------------------


def test_flaml_sees_only_the_train_split(workspace):
    train_frame = pd.read_parquet(workspace / paths.TRAIN)
    X, y, w = split_xy_w(train_frame)
    seen = {}

    def recording_search(X_search, y_search, w_search, **settings):
        seen.update(X=X_search, y=y_search, w=w_search, **settings)
        return automl.SearchResult(
            learner="histgb",
            estimator=HistGradientBoostingRegressor(loss="poisson", max_iter=20),
            config={"loss": "poisson"},
            best_loss=0.5,
            trials=[],
        )

    params = fast_params(flaml_holdout_ratio=0.3)
    candidate = FlamlCandidate(params=params, search=recording_search)
    # The workspace experiment keeps the run's artifacts under tmp_path.
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    with mlflow.start_run() as run:
        fitted = candidate.fit(X, y, w, run.info.run_id)
    # Exactly the train split, preprocessed for trees; the validation split is never passed.
    np.testing.assert_array_equal(seen["X"], build_tree_preprocess().fit_transform(X))
    np.testing.assert_array_equal(seen["y"], y)
    np.testing.assert_array_equal(seen["w"], w)
    assert seen["holdout_ratio"] == 0.3
    assert seen["time_budget_s"] == params.training.flaml_time_budget_s
    assert seen["max_iter"] == params.training.flaml_max_iter
    assert seen["seed"] == params.training.seed
    assert fitted.algorithm == "flaml:histgb"
    # The best configuration is refitted on the whole train split.
    assert fitted.pipeline[-1].n_iter_ >= 1


# --- FLAML: every learner's refit path, one learner at a time --------------------------------


@pytest.mark.parametrize("learner", ["xgboost", "xgb_limitdepth", "histgb"])
def test_a_single_learner_search_refits_a_loadable_poisson_pipeline(
    workspace, monkeypatch, learner
):
    # On fake data the winner is always lgbm, so restrict the search to one learner to run the
    # refit -> skops log_model -> pyfunc load path for the others too (they win on real data).
    # LEARNERS and POISSON_HP are patched together, as run_search reads both.
    monkeypatch.setattr(automl, "LEARNERS", (learner,))
    monkeypatch.setattr(automl, "POISSON_HP", {learner: automl.POISSON_HP[learner]})
    client = MlflowClient()
    parent_id = train(fast_params(candidates=["constant", "flaml"]))
    child = _children(client, parent_id)[CANDIDATE_FLAML]
    assert child.data.tags[TAG_FLAML_BEST_LEARNER] == learner
    assert child.data.tags[TAG_ALGORITHM] == f"{ALGORITHM_FLAML_PREFIX}{learner}"

    estimator = _load_pipeline(client, child)[-1]
    estimator_class, key, poisson = LEARNER_ESTIMATORS[learner]
    assert type(estimator) is estimator_class
    assert estimator.get_params()[key] == poisson

    requirements = _requirements(client, child)
    assert "flaml" not in requirements
    assert ("xgboost" in requirements) == learner.startswith("xgb")
    assert "lightgbm" not in requirements

    model = mlflow.pyfunc.load_model(logged_model_uri(client, child))
    predictions = np.asarray(model.predict(coerce_features(make_policies(20, seed=3))), float)
    assert predictions.shape == (20,) and (predictions > 0).all()


def test_registry_and_evaluation_do_not_import_flaml():
    # register, promote, and evaluate-test import training for read_split; the api image has
    # no FLAML, so FLAML must load only when the FLAML candidate is fitted.
    code = (
        "import sys, claimfreq.registry, claimfreq.evaluation, claimfreq.training; "
        "print('flaml' in sys.modules)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


def test_model_requirements_follow_the_final_estimator():
    xgb = Pipeline([("model", XGBRegressor())])
    names = {line.split("==")[0] for line in model_requirements(xgb)}
    assert "xgboost" in names
    assert not names & {"lightgbm", "flaml"}
