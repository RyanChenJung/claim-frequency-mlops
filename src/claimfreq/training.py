"""Role B (Modeling): the ``claimfreq train`` command.

``train`` fits every candidate on ``splits/train``, scores it on ``splits/validation``, and logs
one parent MLflow run per invocation with one child run per candidate (docs/contracts.md,
Model: candidate scope). Each child run carries:

- the candidate's hyperparameters, including ``training.seed``;
- ``val_*`` metrics from ``metrics.compute_all`` (ADR-0002);
- the lineage tags ``git_commit``, ``data_hash``, ``schema_hash``, plus ``algorithm`` and
  ``candidate_set``;
- the fitted scikit-learn ``Pipeline`` at ``MODEL_ARTIFACT_PATH``, logged with
  ``mlflow.sklearn`` and skops serialization, so ``register`` (a separate Airflow task, in a
  separate container) can register the winner without refitting it.

Every candidate is a ``Pipeline`` with all preprocessing inside it, so it takes the raw rating
factors of ``schema.FEATURE_COLUMNS``. It is fitted on claim frequency ``ClaimNb / Exposure``
with ``sample_weight = Exposure`` (``model.split_xy_w``). Tunable values come from the
``training`` section of ``params.yaml``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from importlib.metadata import version
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
    MODEL_ARTIFACT_PATH,
    TAG_ALGORITHM,
    TAG_CANDIDATE_SET,
    TAG_DATA_HASH,
    TAG_GIT_COMMIT,
    TAG_SCHEMA_HASH,
    ConstantFrequencyModel,
    MetricPrefix,
    file_md5,
    split_xy_w,
)
from claimfreq.contracts.params import GlmParams, Params, load_params

logger = logging.getLogger(__name__)

# Name of the last Pipeline step of every candidate. `fit` routes `sample_weight` to it.
ESTIMATOR_STEP = "model"
# Run name of the parent run, so a person can spot `train` invocations in the MLflow UI.
PARENT_RUN_NAME = "train"
# MLflow param key of the model seed, logged on the parent and on every child run.
PARAM_SEED = "training.seed"

# Algorithm tag values (docs/contracts.md, Model version tags).
ALGORITHM_CONSTANT = "constant"
ALGORITHM_POISSON_GLM = "poisson_glm"

# How each rating factor enters the Poisson GLM. The choices are modeling decisions, not tunable
# values; the tunable ones (penalty, solver, band edges, bin count) are in params.yaml.
# - DrivAge: explicit age bands, one-hot, so the young-driver effect is not blurred.
# - VehAge: quantile bins, one-hot (new cars claim more; the effect is not linear).
# - Density and BonusMalus: log, then standardized. On the log scale a coefficient is an
#   elasticity, so predictions stay bounded at the extremes the API accepts (BonusMalus 350,
#   Density 50,000) instead of exploding as a linear term would.
# - VehPower and the categorical factors: one-hot. An unseen category maps to all zeros.
GLM_BANDED = ["DrivAge"]
GLM_BINNED = ["VehAge"]
GLM_LOG_SCALED = ["Density", "BonusMalus"]
GLM_ONE_HOT = ["VehPower", *schema.CATEGORICAL_FEATURES]
VEHAGE_BIN_STRATEGY = "quantile"
VEHAGE_QUANTILE_METHOD = "averaged_inverted_cdf"
ONE_HOT_UNKNOWN = "ignore"

# Runtime libraries of a logged model, written to its requirements.txt instead of the whole uv
# project. MLflow adds its own `mlflow` pin. LightGBM and XGBoost join when #16 adds them.
MODEL_REQUIREMENTS = ("scikit-learn", "skops", "numpy", "pandas", "pyarrow")


@dataclass
class Candidate:
    """One model to train: its MLflow run name, ``algorithm`` tag, Pipeline, and params."""

    name: str
    algorithm: str
    pipeline: Pipeline
    params: dict[str, Any] = field(default_factory=dict)


def build_glm_pipeline(glm: GlmParams) -> Pipeline:
    """Return the unfitted Poisson GLM Pipeline: preprocessing plus ``PoissonRegressor``.

    Only scikit-learn and numpy objects appear in it (no lambdas), so skops can serialize it.
    """
    # np.digitize maps an age to the number of edges at or below it: 0 for 18-20, 1 for 21-24...
    banding = FunctionTransformer(np.digitize, kw_args={"bins": list(glm.drivage_edges)})
    preprocess = ColumnTransformer(
        [
            (
                "banded",
                Pipeline(
                    [
                        ("band", banding),
                        ("one_hot", OneHotEncoder(handle_unknown=ONE_HOT_UNKNOWN)),
                    ]
                ),
                GLM_BANDED,
            ),
            (
                "binned",
                KBinsDiscretizer(
                    n_bins=glm.vehage_bins,
                    encode="onehot",
                    strategy=VEHAGE_BIN_STRATEGY,
                    quantile_method=VEHAGE_QUANTILE_METHOD,
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
            (
                "one_hot",
                OneHotEncoder(handle_unknown=ONE_HOT_UNKNOWN, sparse_output=False),
                GLM_ONE_HOT,
            ),
        ],
        remainder="drop",
        sparse_threshold=0.0,
        verbose_feature_names_out=False,
    )
    regressor = PoissonRegressor(alpha=glm.alpha, max_iter=glm.max_iter, solver=glm.solver)
    return Pipeline([("preprocess", preprocess), (ESTIMATOR_STEP, regressor)])


def glm_logged_params(glm: GlmParams) -> dict[str, Any]:
    """Return the GLM's settings as MLflow params: tunable values and preprocessing choices."""
    return {
        **{f"glm.{key}": value for key, value in glm.model_dump().items()},
        "preprocess.banded": ",".join(GLM_BANDED),
        "preprocess.binned": ",".join(GLM_BINNED),
        "preprocess.binned_strategy": VEHAGE_BIN_STRATEGY,
        "preprocess.binned_quantile_method": VEHAGE_QUANTILE_METHOD,
        "preprocess.log_scaled": ",".join(GLM_LOG_SCALED),
        "preprocess.one_hot": ",".join(GLM_ONE_HOT),
        "preprocess.handle_unknown": ONE_HOT_UNKNOWN,
    }


def build_candidates(params: Params) -> list[Candidate]:
    """Return the candidates of this PR: the constant baseline and a Poisson GLM.

    Neither candidate is random, but every candidate logs ``training.seed``, so runs stay
    comparable once random candidates (LightGBM, FLAML) join the set.
    """
    seed = {PARAM_SEED: params.training.seed}
    glm = params.training.glm
    return [
        Candidate(
            name=ALGORITHM_CONSTANT,
            algorithm=ALGORITHM_CONSTANT,
            pipeline=Pipeline([(ESTIMATOR_STEP, ConstantFrequencyModel())]),
            params=seed,
        ),
        Candidate(
            name=ALGORITHM_POISSON_GLM,
            algorithm=ALGORITHM_POISSON_GLM,
            pipeline=build_glm_pipeline(glm),
            params={**seed, **glm_logged_params(glm)},
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


def model_requirements() -> list[str]:
    """Return the pinned runtime requirements of a logged model, from the installed versions."""
    return [f"{package}=={version(package)}" for package in MODEL_REQUIREMENTS]


def log_candidate(
    candidate: Candidate,
    train_data: tuple[pd.DataFrame, np.ndarray, np.ndarray],
    validation_data: tuple[pd.DataFrame, np.ndarray, np.ndarray],
    tags: dict[str, str],
) -> str:
    """Fit, score, and log one candidate as a child run of the active run; return its run id."""
    X_train, y_train, w_train = train_data
    X_val, y_val, w_val = validation_data
    with mlflow.start_run(run_name=candidate.name, nested=True) as run:
        mlflow.set_tags({**tags, TAG_ALGORITHM: candidate.algorithm})
        mlflow.log_params(candidate.params)
        pipeline = fit_candidate(candidate, X_train, y_train, w_train)
        predictions = pipeline.predict(X_val)
        scores = metrics.compute_all(y_val, predictions, w_val)
        mlflow.log_metrics(metrics.with_prefix(scores, MetricPrefix.VAL))
        mlflow.sklearn.log_model(
            pipeline,
            name=MODEL_ARTIFACT_PATH,
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_SKOPS,
            skops_trusted_types=skops_trusted_types(pipeline),
            signature=infer_signature(X_val, predictions),
            pip_requirements=model_requirements(),
        )
        return run.info.run_id


def train(params: Params | None = None) -> str:
    """Train every candidate and log them under one new parent run; return the parent run id.

    Reads ``splits/train.parquet`` and ``splits/validation.parquet`` under ``DATA_DIR`` and logs
    to ``MLFLOW_TRACKING_URI``, in experiment ``claim-frequency``. If a candidate raises, the
    parent run ends ``FAILED`` and ``register`` skips the whole candidate set.

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
    train_data = split_xy_w(read_split(train_path, params))
    validation_data = split_xy_w(read_split(paths.data_path(paths.VALIDATION), params))
    if not np.any(validation_data[1] > 0):
        # Selection still works (poisson_deviance stays finite, and is_better rejects NaN), but
        # val_gini and val_calibration_ratio are NaN without a single observed claim.
        logger.warning("validation split has no claims: val_gini and val_calibration_ratio are NaN")

    mlflow.set_tracking_uri(env.get(env.MLFLOW_TRACKING_URI))
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    with mlflow.start_run(run_name=PARENT_RUN_NAME) as parent:
        parent_id = parent.info.run_id
        tags = {**lineage, TAG_CANDIDATE_SET: parent_id}
        mlflow.set_tags(tags)
        mlflow.log_param(PARAM_SEED, params.training.seed)
        for candidate in build_candidates(params):
            log_candidate(candidate, train_data, validation_data, tags)
    return parent_id
