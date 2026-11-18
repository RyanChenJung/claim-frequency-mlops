"""Model contract: registry names, MLflow keys, model I/O, and the constant baseline model.

Producers and consumers:

- Role B (Modeling) trains, logs, and registers models under these names, tags, and metric keys,
  and logs the reference dataset artifact.
- Role D (Serving and monitoring) loads ``models:/claim-frequency@champion``, calls ``predict``
  on a frame with exactly ``FEATURE_COLUMNS``, and loads the reference dataset from the
  champion's run.
- ``ConstantFrequencyModel`` is both Role B's baseline model (every candidate must beat it) and
  Role D's stub model before any real model exists.

Training convention (ADR-0002): the target is claim frequency ``y = ClaimNb / Exposure`` with
``sample_weight = Exposure``, on clean-stage frames.

Model artifact format (what ``train`` logs, ``register`` registers, and the API loads):

- The champion is a scikit-learn ``Pipeline`` with all preprocessing (for example categorical
  encoding) inside it, so it takes the raw rating factors.
- ``train`` logs every candidate's fitted Pipeline with ``mlflow.sklearn`` using skops
  serialization, as an MLflow logged model named ``MODEL_ARTIFACT_PATH`` linked to the
  candidate's child run; its ``skops_trusted_types`` are listed at log time. ``train`` and
  ``register`` run in separate containers, so ``register`` registers the best child's logged
  model (``models:/<model id>``) instead of refitting. The version's ``run_id`` is that child.
- The logged model has an MLflow signature inferred from the validation features, so pyfunc
  enforces the input schema (column names and dtypes of ``FEATURE_COLUMNS``).
- It is loaded with ``mlflow.pyfunc.load_model("models:/claim-frequency@champion")``.
- Its runtime libraries are a subset of {scikit-learn, lightgbm, xgboost}, which is why the
  ``api`` dependency group carries all three.
- The pyfunc model receives a features-only frame built with ``schema.coerce_features``.
"""

import hashlib
from enum import StrEnum
from os import PathLike
from typing import Protocol, Self

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted

from claimfreq.contracts.schema import CLAIM_NB, EXPOSURE, FEATURE_COLUMNS

# --- Registry and experiment --------------------------------------------------------------------

REGISTERED_MODEL_NAME = "claim-frequency"
MLFLOW_EXPERIMENT_NAME = "claim-frequency"
ALIAS_CHAMPION = "champion"  # served by the API; moved only by a human via `promote`
ALIAS_CHALLENGER = "challenger"  # set by `register` on every new best model

# --- Tags ---------------------------------------------------------------------------------------

# Run id of the `train` parent run, on the parent and on each of its child runs. `register` scopes
# its search to one candidate set (docs/contracts.md, Model: candidate scope).
TAG_CANDIDATE_SET = "candidate_set"
# Value of claimfreq.contracts.env.git_commit() at training time.
TAG_GIT_COMMIT = "git_commit"
# file_md5() of the train split file actually read. It equals the md5 that DVC 3 records for
# data/splits/train.parquet in dvc.lock, because `split` declares each split file as its own out.
TAG_DATA_HASH = "data_hash"
# Value of claimfreq.contracts.schema.schema_hash() at training time.
TAG_SCHEMA_HASH = "schema_hash"
# Next semantic version, computed by `register` against the champion (ADR-0007), e.g. "1.0.0".
TAG_SEMVER = "semver"
# Algorithm family, e.g. "constant", "poisson_glm", "lightgbm", "flaml:lgbm". A change bumps MINOR.
TAG_ALGORITHM = "algorithm"
# Metric base name used to pick the model (params.yaml training.selection_metric).
TAG_SELECTION_METRIC = "selection_metric"
# Its validation value, as a string (MLflow tag values are strings).
TAG_SELECTION_METRIC_VALUE = "selection_metric_value"

# Lineage tags on every MLflow training run.
RUN_TAGS = (TAG_GIT_COMMIT, TAG_DATA_HASH, TAG_SCHEMA_HASH)
# Tags on every registered model version.
MODEL_VERSION_TAGS = (
    TAG_SEMVER,
    TAG_SCHEMA_HASH,
    TAG_DATA_HASH,
    TAG_GIT_COMMIT,
    TAG_ALGORITHM,
    TAG_SELECTION_METRIC,
    TAG_SELECTION_METRIC_VALUE,
)

# --- Metric keys --------------------------------------------------------------------------------


class MetricName(StrEnum):
    """Base names of the three metrics computed by ``claimfreq.metrics`` (ADR-0002)."""

    POISSON_DEVIANCE = "poisson_deviance"  # lower is better; drives model selection by default
    GINI = "gini"  # higher is better
    CALIBRATION_RATIO = "calibration_ratio"  # 1.0 is perfect; below 1 means under-pricing


class MetricGoal(StrEnum):
    """Which values of a metric are better, in machine-readable form."""

    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"
    CLOSEST_TO_ONE = "closest_to_one"


METRIC_GOALS: dict[MetricName, MetricGoal] = {
    MetricName.POISSON_DEVIANCE: MetricGoal.MINIMIZE,
    MetricName.GINI: MetricGoal.MAXIMIZE,
    MetricName.CALIBRATION_RATIO: MetricGoal.CLOSEST_TO_ONE,
}


def is_better(metric: MetricName | str, candidate: float, incumbent: float) -> bool:
    """Return True if ``candidate`` is strictly better than ``incumbent`` on ``metric``.

    Role B's ``register`` uses it with ``training.selection_metric``, so the direction of every
    metric lives in one place (``METRIC_GOALS``). A tie is not better, and neither is any
    comparison that involves NaN. The result is a Python ``bool`` even for numpy inputs, such
    as the ``np.float64`` values of ``mlflow.search_runs``, whose comparisons give ``np.bool_``.
    """
    goal = METRIC_GOALS[MetricName(metric)]
    if goal is MetricGoal.MINIMIZE:
        return bool(candidate < incumbent)
    if goal is MetricGoal.MAXIMIZE:
        return bool(candidate > incumbent)
    return bool(abs(candidate - 1.0) < abs(incumbent - 1.0))


class MetricPrefix(StrEnum):
    """Prefix of an MLflow metric key. A full key is ``f"{prefix}{name}"``.

    For example ``val_poisson_deviance``. Keeping the split in the key name prevents a test
    or live metric from ever being used for model selection by accident.
    """

    VAL = "val_"  # validation split; the only split used for model selection
    TEST = "test_"  # final champion evaluation only (`evaluate-test`)
    LIVE = "live_"  # monitoring on logged traffic once ground truth has arrived


# --- Model artifact -----------------------------------------------------------------------------

# Name of the MLflow logged model holding the fitted Pipeline of each `train` child run (in
# MLflow 3, a logged model linked to the run). `register` registers it by its model id.
MODEL_ARTIFACT_PATH = "model"

# --- Reference dataset artifact -----------------------------------------------------------------

PREDICTED_FREQUENCY = "predicted_frequency"

# Logged by `register` on the model's MLflow run; loaded by `monitor` from the champion's run.
REFERENCE_ARTIFACT_PATH = "reference/reference.parquet"
# The validation split's features and targets, plus the model's prediction for each policy.
REFERENCE_COLUMNS = [*FEATURE_COLUMNS, CLAIM_NB, EXPOSURE, PREDICTED_FREQUENCY]

# --- Model I/O ----------------------------------------------------------------------------------


class FrequencyModel(Protocol):
    """What the API and the monitor need from any model, including an MLflow pyfunc model.

    Input: a DataFrame with exactly ``FEATURE_COLUMNS``, in canonical order and dtypes
    (``schema.DTYPES``), with no ``IDpol`` and no target columns. Build it with
    ``schema.coerce_features``, from API requests and from ``prediction_log`` rows alike.

    Output: a 1-D float array of length ``len(X)``, every value > 0, in predicted claims per
    policy-year (claim frequency).
    """

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return the predicted claim frequency of each row of ``X``."""
        ...


def split_xy_w(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Split a clean-stage frame into model inputs, target, and sample weight.

    Returns:
        ``(X, y, w)`` where ``X`` is ``df[FEATURE_COLUMNS]``, ``y`` is the observed claim
        frequency ``ClaimNb / Exposure``, and ``w`` is ``Exposure``. Weighting by exposure makes
        a policy observed for one year count twice as much as one observed for six months.
    """
    exposure = df[EXPOSURE].to_numpy(dtype=float)
    y = df[CLAIM_NB].to_numpy(dtype=float) / exposure
    return df[FEATURE_COLUMNS], y, exposure


def file_md5(path: str | PathLike[str]) -> str:
    """Return the md5 hex digest of a file's bytes: the ``data_hash`` lineage tag.

    ``train`` calls it on the train split file it actually read, so the tag describes the data
    the model saw. DVC 3 records the same plain content md5 for a file out in ``dvc.lock``.
    """
    digest = hashlib.md5()
    with open(path, "rb") as f:
        # Read in 1 MiB chunks, so a large split never has to fit in memory at once.
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ConstantFrequencyModel(RegressorMixin, BaseEstimator):
    """Baseline model that predicts the portfolio's mean claim frequency for every policy.

    ``fit`` stores the exposure-weighted mean of ``y``, which equals
    ``sum(ClaimNb) / sum(Exposure)``. Being a scikit-learn regressor, it can be logged with
    ``mlflow.sklearn`` like any other candidate.
    """

    def fit(self, X: pd.DataFrame, y: np.ndarray, sample_weight: np.ndarray | None = None) -> Self:
        """Fit the constant ``frequency_`` from frequencies ``y`` weighted by exposure."""
        frequency = float(np.average(np.asarray(y, dtype=float), weights=sample_weight))
        # The output contract requires strictly positive predictions (Poisson deviance is
        # infinite at 0), so a training set without any claim cannot produce a valid model.
        if not frequency > 0:
            raise ValueError("ConstantFrequencyModel needs at least one claim to fit")
        self.frequency_ = frequency
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return ``frequency_`` for every row of ``X``."""
        check_is_fitted(self)
        return np.full(len(X), self.frequency_, dtype=float)
