"""Role B (Modeling): the ``claimfreq evaluate-test`` command.

A human runs it after ``promote`` (docs/contracts.md, CLI). It scores the ``champion`` on the
test split, which no other command reads, and logs ``test_*`` metrics on the champion's run
(the ``train`` child run of the champion version), so ``/model-info`` reports them next to its
``val_*`` metrics.

- The champion is resolved once through the alias, then loaded by **version number**, so the
  metrics land on the run of the model that was actually scored even if ``promote`` runs
  meanwhile.
- No champion, or a champion trained on another feature schema, is a failed check (exit 10):
  its predictions on the current test split would mean nothing.
- So is a train split under ``DATA_DIR`` whose md5 is not the champion's ``data_hash``: the
  splits on disk then belong to another data version, whose test split may hold the
  champion's own training rows (the same guard as ``register``'s).
- The test split's md5 is logged as tag ``test_data_hash`` on the same run, so the ``test_*``
  metrics say which file they were measured on. Each run of ``evaluate-test`` logs its
  values at the next step (0, 1, ...), so the metric history keeps every evaluation and MLflow
  reports the latest.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import mlflow
import numpy as np
from mlflow.tracking import MlflowClient

from claimfreq import metrics
from claimfreq.contracts import env, paths, schema
from claimfreq.contracts.model import (
    REGISTERED_MODEL_NAME,
    TAG_DATA_HASH,
    TAG_SCHEMA_HASH,
    TAG_SEMVER,
    MetricPrefix,
    file_md5,
    split_xy_w,
)
from claimfreq.contracts.params import Params, load_params
from claimfreq.registry import RegistryCheckError, current_champion
from claimfreq.training import read_split

logger = logging.getLogger(__name__)

# Tag on the champion's run: md5 of the test split its `test_*` metrics were measured on.
TAG_TEST_DATA_HASH = "test_data_hash"


@dataclass(frozen=True)
class Evaluation:
    """The outcome of ``evaluate_test``."""

    version: str  # the champion's registry version
    semver: str  # its semantic version tag
    run_id: str  # its run, which now holds the test_* metrics
    metrics: dict[str, float]  # test_poisson_deviance, test_gini, test_calibration_ratio


def evaluate_test(params: Params | None = None) -> Evaluation:
    """Score the champion on ``splits/test.parquet`` and log ``test_*`` metrics on its run.

    Raises:
        RegistryCheckError: if there is no champion, its ``schema_hash`` is not the current
            feature schema's, or the train split's md5 is not its ``data_hash``.
        SchemaError: if the test split breaks the clean-stage schema.
        FileNotFoundError: if there is no test split under ``DATA_DIR``.
    """
    params = params or load_params()
    mlflow.set_tracking_uri(env.get(env.MLFLOW_TRACKING_URI))
    client = MlflowClient()
    champion = current_champion(client)
    if champion is None:
        raise RegistryCheckError(
            f"evaluate-test: {REGISTERED_MODEL_NAME} has no champion; run promote first"
        )
    if champion.tags.get(TAG_SCHEMA_HASH) != schema.schema_hash():
        raise RegistryCheckError(
            f"evaluate-test: champion version {champion.version} was trained on another feature "
            "schema than the current one; register and promote a model of this schema first"
        )

    train_md5 = file_md5(paths.data_path(paths.TRAIN))
    if champion.tags.get(TAG_DATA_HASH) != train_md5:
        raise RegistryCheckError(
            f"evaluate-test: {paths.TRAIN} under DATA_DIR has md5 {train_md5}, but champion "
            f"version {champion.version} was trained on {champion.tags.get(TAG_DATA_HASH)}; "
            "check out the champion's data version (dvc.lock) before evaluating"
        )

    test_path = paths.data_path(paths.TEST)
    X, y, w = split_xy_w(read_split(test_path, params))
    if not np.any(y > 0):
        logger.warning("test split has no claims: test_gini and test_calibration_ratio are NaN")
    model = mlflow.pyfunc.load_model(f"models:/{REGISTERED_MODEL_NAME}/{champion.version}")
    predictions = np.asarray(model.predict(schema.coerce_features(X)), dtype=float)
    scores = metrics.with_prefix(metrics.compute_all(y, predictions, w), MetricPrefix.TEST)

    for key, value in scores.items():
        # The next step, so repeated evaluations stay ordered in the metric history.
        step = len(client.get_metric_history(champion.run_id, key))
        client.log_metric(champion.run_id, key, value, step=step)
    client.set_tag(champion.run_id, TAG_TEST_DATA_HASH, file_md5(test_path))
    return Evaluation(
        version=str(champion.version),
        semver=champion.tags.get(TAG_SEMVER, ""),
        run_id=champion.run_id,
        metrics=scores,
    )
