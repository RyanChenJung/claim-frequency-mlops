"""Role B (Modeling): the ``claimfreq train`` command.

``train`` fits every candidate on ``splits/train``, scores it on ``splits/validation``, and logs
one parent MLflow run per invocation with one child run per candidate (docs/contracts.md,
Model: candidate scope). Each child run carries:

- the candidate's hyperparameters, including ``training.seed``;
- ``val_*`` metrics from ``metrics.compute_all`` (ADR-0002);
- the lineage tags ``git_commit``, ``data_hash``, ``schema_hash``, and ``algorithm``;
- the fitted scikit-learn ``Pipeline`` at ``MODEL_ARTIFACT``, logged with ``mlflow.sklearn``
  and skops serialization, so ``register`` (a separate Airflow task, in a separate container)
  can register the winner without refitting it.

Every candidate is a ``Pipeline`` with all preprocessing inside it, so it takes the raw rating
factors of ``schema.FEATURE_COLUMNS``. It is fitted on claim frequency ``ClaimNb / Exposure``
with ``sample_weight = Exposure`` (``model.split_xy_w``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import skops.io
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    KBinsDiscretizer,
    OneHotEncoder,
    StandardScaler,
)

from claimfreq import metrics
from claimfreq.contracts import env, paths, schema
from claimfreq.contracts.model import (
    MLFLOW_EXPERIMENT_NAME,
    TAG_ALGORITHM,
    TAG_DATA_HASH,
    TAG_GIT_COMMIT,
    TAG_SCHEMA_HASH,
    ConstantFrequencyModel,
    MetricPrefix,
    file_md5,
    split_xy_w,
)
from claimfreq.contracts.params import Params, load_params

# Tag on the parent run and on every child run: the parent run id (docs/contracts.md, Model).
TAG_CANDIDATE_SET = "candidate_set"
# Artifact name of the fitted Pipeline in each child run. `register` loads it from here.
MODEL_ARTIFACT = "model"
# Name of the last Pipeline step of every candidate. `fit` routes `sample_weight` to it.
ESTIMATOR_STEP = "model"
# Run name of the parent run, so a person can spot `train` invocations in the MLflow UI.
PARENT_RUN_NAME = "train"

# Algorithm tag values (docs/contracts.md, Model version tags).
ALGORITHM_CONSTANT = "constant"
ALGORITHM_POISSON_GLM = "poisson_glm"

# Poisson GLM preprocessing. Driver age and vehicle age have a non-linear effect on claim
# frequency (young drivers and new cars claim more), so the GLM sees them as quantile bins.
# Density spans four orders of magnitude, so it enters on the log scale.
GLM_BINNED = ["VehAge", "DrivAge"]
GLM_LOG_SCALED = ["Density"]
GLM_SCALED = ["BonusMalus"]
GLM_ONE_HOT = ["VehPower", *schema.CATEGORICAL_FEATURES]
GLM_N_BINS = 10
GLM_ALPHA = 1e-4  # small L2 penalty: keeps the one-hot design well conditioned
GLM_MAX_ITER = 1000


@dataclass
class Candidate:
    """One model to train: its MLflow run name, ``algorithm`` tag, Pipeline, and params."""

    name: str
    algorithm: str
    pipeline: Pipeline
    params: dict[str, Any] = field(default_factory=dict)


def build_glm_pipeline() -> Pipeline:
    """Return the unfitted Poisson GLM Pipeline: preprocessing plus ``PoissonRegressor``.

    Only scikit-learn and numpy objects appear in it (no lambdas), so skops can serialize it.
    ``handle_unknown="ignore"`` maps a category unseen in training to all-zero indicators
    instead of failing at serving time.
    """
    preprocess = ColumnTransformer(
        [
            (
                "binned",
                KBinsDiscretizer(
                    n_bins=GLM_N_BINS,
                    encode="onehot",
                    strategy="quantile",
                    quantile_method="averaged_inverted_cdf",
                ),
                GLM_BINNED,
            ),
            (
                "log_scaled",
                Pipeline(
                    [
                        ("log", FunctionTransformer(np.log, feature_names_out="one-to-one")),
                        ("scale", StandardScaler()),
                    ]
                ),
                GLM_LOG_SCALED,
            ),
            ("scaled", StandardScaler(), GLM_SCALED),
            (
                "one_hot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                GLM_ONE_HOT,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    regressor = PoissonRegressor(alpha=GLM_ALPHA, max_iter=GLM_MAX_ITER, solver="newton-cholesky")
    return Pipeline([("preprocess", preprocess), (ESTIMATOR_STEP, regressor)])


def build_candidates(seed: int) -> list[Candidate]:
    """Return the candidates of this PR: the constant baseline and a Poisson GLM.

    ``seed`` is ``training.seed``. Neither candidate is random, but every candidate logs it, so
    runs stay comparable once random candidates (LightGBM, FLAML) join the set.
    """
    return [
        Candidate(
            name="constant",
            algorithm=ALGORITHM_CONSTANT,
            pipeline=Pipeline([(ESTIMATOR_STEP, ConstantFrequencyModel())]),
            params={"training.seed": seed},
        ),
        Candidate(
            name="poisson_glm",
            algorithm=ALGORITHM_POISSON_GLM,
            pipeline=build_glm_pipeline(),
            params={
                "training.seed": seed,
                "alpha": GLM_ALPHA,
                "max_iter": GLM_MAX_ITER,
                "solver": "newton-cholesky",
                "n_bins": GLM_N_BINS,
                "binned": ",".join(GLM_BINNED),
                "log_scaled": ",".join(GLM_LOG_SCALED),
                "scaled": ",".join(GLM_SCALED),
                "one_hot": ",".join(GLM_ONE_HOT),
            },
        ),
    ]


def read_split(path: Path, params: Params) -> pd.DataFrame:
    """Read a clean-stage split, validate it, and coerce it to the canonical dtypes.

    A ``SchemaError`` from ``validate_frame`` propagates, so ``claimfreq train`` exits 10.
    """
    df = pd.read_parquet(path)
    schema.validate_frame(df, "clean", params)
    return schema.coerce_frame(df)


def fit_candidate(candidate: Candidate, X: pd.DataFrame, y: np.ndarray, w: np.ndarray) -> Pipeline:
    """Fit the candidate's Pipeline with exposure weights routed to its final step."""
    return candidate.pipeline.fit(X, y, **{f"{ESTIMATOR_STEP}__sample_weight": w})


def skops_trusted_types(pipeline: Pipeline) -> list[str]:
    """Return the types skops would refuse to load from ``pipeline``, to trust them at log time.

    The list is computed from the fitted Pipeline itself, so it stays correct as candidates
    change (for example it includes ``claimfreq.contracts.model.ConstantFrequencyModel``).
    """
    return skops.io.get_untrusted_types(data=skops.io.dumps(pipeline))


def log_candidate(
    candidate: Candidate,
    train: tuple[pd.DataFrame, np.ndarray, np.ndarray],
    validation: tuple[pd.DataFrame, np.ndarray, np.ndarray],
    tags: dict[str, str],
) -> str:
    """Fit, score, and log one candidate as a child run of the active run; return its run id."""
    X_train, y_train, w_train = train
    X_val, y_val, w_val = validation
    with mlflow.start_run(run_name=candidate.name, nested=True) as run:
        mlflow.set_tags({**tags, TAG_ALGORITHM: candidate.algorithm})
        mlflow.log_params(candidate.params)
        pipeline = fit_candidate(candidate, X_train, y_train, w_train)
        predictions = pipeline.predict(X_val)
        scores = metrics.compute_all(y_val, predictions, w_val)
        mlflow.log_metrics(metrics.with_prefix(scores, MetricPrefix.VAL))
        mlflow.sklearn.log_model(
            pipeline,
            name=MODEL_ARTIFACT,
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_SKOPS,
            skops_trusted_types=skops_trusted_types(pipeline),
            signature=infer_signature(X_val, predictions),
        )
        return run.info.run_id


def train(params: Params | None = None) -> str:
    """Train every candidate and log them under one new parent run; return the parent run id.

    Reads ``splits/train.parquet`` and ``splits/validation.parquet`` under ``DATA_DIR`` and logs
    to ``MLFLOW_TRACKING_URI``, in experiment ``claim-frequency``.

    Raises:
        RuntimeError: if ``git_commit`` cannot be determined; no run is logged without lineage.
        SchemaError: if a split breaks the clean-stage schema.
    """
    params = params or load_params()
    # Lineage first: a run that cannot record where it came from must not be logged at all.
    train_path = paths.data_path(paths.TRAIN)
    lineage = {
        TAG_GIT_COMMIT: env.git_commit(),
        TAG_DATA_HASH: file_md5(train_path),
        TAG_SCHEMA_HASH: schema.schema_hash(),
    }
    train_xyw = split_xy_w(read_split(train_path, params))
    validation_xyw = split_xy_w(read_split(paths.data_path(paths.VALIDATION), params))

    mlflow.set_tracking_uri(env.get(env.MLFLOW_TRACKING_URI))
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    with mlflow.start_run(run_name=PARENT_RUN_NAME) as parent:
        parent_id = parent.info.run_id
        tags = {**lineage, TAG_CANDIDATE_SET: parent_id}
        mlflow.set_tags(tags)
        mlflow.log_param("training.seed", params.training.seed)
        for candidate in build_candidates(params.training.seed):
            log_candidate(candidate, train_xyw, validation_xyw, tags)
    return parent_id
