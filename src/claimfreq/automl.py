"""Role B (Modeling): the FLAML AutoML search behind the ``flaml`` candidate (ADR-0006).

``train`` calls ``run_search`` on the preprocessed train split, then refits the best
configuration as a plain scikit-learn ``Pipeline`` (see ``training.FlamlCandidate``), so the
registered model never depends on FLAML. Decisions and gotchas from the ADR-0006 spike:

- The search covers ``lgbm``, ``xgboost``, ``xgb_limitdepth``, and ``histgb``. ``custom_hp``
  forces a Poisson objective on each of them; it applies only to learners in
  ``estimator_list``, so both come from the one ``POISSON_HP`` mapping and cannot drift apart.
- Trials are compared by exposure-weighted mean Poisson deviance (``holdout_deviance``) on a
  holdout that FLAML carves out of the **train** split. The validation split is never shown to
  FLAML, so FLAML's winner gets no head start when ``register`` compares ``val_*`` metrics.
- FLAML's own MLflow logging crashes on MLflow >= 3.14, so ``mlflow_logging=False``. Every
  trial is read back from FLAML's log (``log_type="all"``, not only improving trials) and
  replayed by ``log_trials`` as a run nested under the FLAML child run, never under the parent.
- ``retrain_full=False``: the best configuration is refitted on the whole train split by
  ``train`` itself, so FLAML need not spend its time budget on a refit.
"""

from __future__ import annotations

import math
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from flaml import AutoML
from flaml.automl.training_log import training_log_reader
from mlflow.entities import Metric, Param, RunStatus
from mlflow.tracking import MlflowClient
from sklearn.base import BaseEstimator, clone

from claimfreq import metrics
from claimfreq.contracts.model import TAG_ALGORITHM

# Learner -> the hyperparameter that makes it fit a Poisson model, as a FLAML `custom_hp`.
POISSON_HP: dict[str, dict[str, dict[str, str]]] = {
    "lgbm": {"objective": {"domain": "poisson"}},
    "xgboost": {"objective": {"domain": "count:poisson"}},
    "xgb_limitdepth": {"objective": {"domain": "count:poisson"}},
    "histgb": {"loss": {"domain": "poisson"}},
}
LEARNERS = tuple(POISSON_HP)

# Metric of each replayed trial run: FLAML's holdout loss. Not `val_`: this holdout is part of
# the train split, not the validation split, and trial runs are never candidates.
TRIAL_METRIC = "flaml_holdout_poisson_deviance"
# Tag of each trial run: the FLAML learner it tried.
TAG_FLAML_LEARNER = "flaml_learner"
# Prefix of a trial run's params: one param per hyperparameter of its configuration.
TRIAL_CONFIG_PREFIX = "config."
# Smallest prediction the metric accepts: Poisson deviance is infinite at 0.
_MIN_PREDICTION = 1e-9


@dataclass(frozen=True)
class SearchResult:
    """The outcome of one FLAML search."""

    learner: str  # the best FLAML learner, e.g. "lgbm"
    estimator: BaseEstimator  # unfitted library estimator with the best configuration
    config: dict[str, Any]  # the best configuration, as FLAML reports it
    best_loss: float  # its holdout Poisson deviance
    trials: list[dict[str, Any]]  # every trial record of FLAML's log


def holdout_deviance(
    X_val,
    y_val,
    estimator,
    labels,
    X_train,
    y_train,
    weight_val=None,
    weight_train=None,
    *args,
    **kwargs,
) -> tuple[float, dict[str, float]]:
    """FLAML custom metric: exposure-weighted mean Poisson deviance on FLAML's holdout.

    FLAML minimizes the first return value and logs the dict. The weights are the holdout's
    exposures, because ``fit`` receives ``sample_weight=Exposure``.
    """
    predictions = np.clip(np.asarray(estimator.predict(X_val), dtype=float), _MIN_PREDICTION, None)
    deviance = metrics.poisson_deviance(y_val, predictions, weight_val)
    return deviance, {"poisson_deviance": deviance}


def run_search(
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    *,
    time_budget_s: int,
    max_iter: int,
    holdout_ratio: float,
    seed: int,
) -> SearchResult:
    """Search Poisson tree models with FLAML on preprocessed features; return the best.

    ``X`` must already be numeric (the tree preprocessing of ``training``), so the refitted
    Pipeline applies exactly the same transformation in front of the returned estimator.
    """
    automl = AutoML()
    with tempfile.TemporaryDirectory() as tmp:
        log_file = Path(tmp) / "flaml.log"
        automl.fit(
            X,
            y,
            sample_weight=w,
            task="regression",
            metric=holdout_deviance,
            estimator_list=list(LEARNERS),
            custom_hp=POISSON_HP,
            time_budget=time_budget_s,
            max_iter=max_iter,
            eval_method="holdout",
            split_ratio=holdout_ratio,
            retrain_full=False,
            seed=seed,
            log_file_name=str(log_file),
            log_type="all",
            mlflow_logging=False,
            verbose=0,
        )
        with training_log_reader(str(log_file)) as reader:
            trials = [record.__dict__ for record in reader.records()]
    return SearchResult(
        learner=automl.best_estimator,
        estimator=clone(automl.model.estimator),
        config=dict(automl.best_config),
        best_loss=float(automl.best_loss),
        trials=trials,
    )


def log_trials(
    client: MlflowClient, experiment_id: str, parent_run_id: str, trials: list[dict[str, Any]]
) -> None:
    """Replay FLAML's trials as finished MLflow runs nested under ``parent_run_id``.

    Trial runs carry no ``candidate_set`` tag: they are not candidates, and ``register``
    searches for train parents by that tag.
    """
    for trial in trials:
        learner = trial["learner"]
        run = client.create_run(
            experiment_id,
            run_name=f"trial-{trial['record_id']}-{learner}",
            tags={
                "mlflow.parentRunId": parent_run_id,
                TAG_ALGORITHM: f"flaml:{learner}",
                TAG_FLAML_LEARNER: learner,
            },
        )
        loss = float(trial["validation_loss"])
        params = [Param(f"{TRIAL_CONFIG_PREFIX}{k}", str(v)) for k, v in trial["config"].items()]
        params.append(Param("sample_size", str(trial["sample_size"])))
        trial_metrics = []
        if math.isfinite(loss):  # a trial that failed to fit has an infinite loss
            trial_metrics.append(Metric(TRIAL_METRIC, loss, int(time.time() * 1000), 0))
        client.log_batch(run.info.run_id, metrics=trial_metrics, params=params)
        client.set_terminated(run.info.run_id, RunStatus.to_string(RunStatus.FINISHED))
