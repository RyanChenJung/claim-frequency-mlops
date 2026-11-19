"""Params contract: the typed, validated view of ``params.yaml``.

``params.yaml`` at the repository root holds every tunable value of the pipeline. DVC tracks it
as stage params, so changing a value reruns only the stages that depend on it. Each section is
owned by one role:

- ``split``, ``clean``: Role A (Data). Role A will add its own ``drift`` section later.
- ``training``: Role B (Modeling).
- ``traffic``, ``monitor``: Role D (Serving and monitoring).

The top-level ``seed`` is the data seed, used only by ``split`` and ``make-drift``. It is a
contract key, like any key read by more than one role. A role may add or change keys in its
own section in its own PR, updating this module, ``params.yaml``, and docs/contracts.md
together (docs/contracts.md, Params).

Commands load it with ``load_params()``. Unknown keys are rejected, so a typo in the file fails
loudly instead of silently falling back to a default. Integer, float, and boolean fields are
strict, so ``true`` is not accepted as 1, ``1`` is not accepted as true, and ``"0.2"`` is not
accepted as 0.2; a float field still accepts a YAML integer such as ``1``. ``params.yaml`` sets
every field explicitly (a test checks the keys); the defaults here serve tests and fakes.
"""

from pathlib import Path
from typing import Any, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

from claimfreq.contracts.api import MAX_BATCH_SIZE
from claimfreq.contracts.model import MetricName
from claimfreq.contracts.schema import DEFAULT_CLAIM_NB_CAP, DEFAULT_EXPOSURE_CAP

# Relative to the working directory: every command runs from the repo root.
PARAMS_PATH = Path("params.yaml")

# Largest seed that every random number generator in the stack accepts (numpy's legacy
# RandomState, and so scikit-learn's random_state, stop at 2**32 - 1).
SEED_MAX = 2**32 - 1


def _seed_field() -> Any:
    """Return a strict seed Field in [0, ``SEED_MAX``] with default 42."""
    return Field(default=42, strict=True, ge=0, le=SEED_MAX)


class _Section(BaseModel):
    """Base for every section: reject unknown keys."""

    model_config = ConfigDict(extra="forbid")


class SplitParams(_Section):
    """Split fractions, both relative to the full dataset (train gets the rest)."""

    test_fraction: float = Field(default=0.2, strict=True, gt=0, lt=1)
    validation_fraction: float = Field(default=0.1, strict=True, gt=0, lt=1)

    @model_validator(mode="after")
    def _leave_room_for_train(self) -> Self:
        if self.test_fraction + self.validation_fraction >= 1:
            raise ValueError("test_fraction + validation_fraction must be < 1")
        return self


class CleanParams(_Section):
    """Clean-stage caps: larger ``ClaimNb`` and ``Exposure`` values are clipped to these."""

    claim_nb_cap: int = Field(default=DEFAULT_CLAIM_NB_CAP, strict=True, ge=1)
    exposure_cap: float = Field(default=DEFAULT_EXPOSURE_CAP, strict=True, gt=0)


class GlmParams(_Section):
    """Poisson GLM candidate (``train``): penalty, solver, and how numeric factors enter."""

    # L2 penalty of PoissonRegressor; small, so it only keeps the one-hot design well conditioned.
    alpha: float = Field(default=1e-4, strict=True, ge=0)
    max_iter: int = Field(default=1000, strict=True, ge=1)
    solver: Literal["lbfgs", "newton-cholesky"] = "newton-cholesky"
    # Left edges of the DrivAge bands after the first band, which starts at the minimum age 18.
    # Fine bands at young ages keep the young-driver effect visible.
    drivage_edges: list[StrictInt] = Field(
        default_factory=lambda: [21, 25, 30, 40, 50, 60, 70], strict=True, min_length=1
    )
    # VehAge enters as this many quantile bins.
    vehage_bins: int = Field(default=10, strict=True, ge=2)

    @model_validator(mode="after")
    def _increasing_edges(self) -> Self:
        edges = self.drivage_edges
        if any(low >= high for low, high in zip(edges, edges[1:], strict=False)):
            raise ValueError("drivage_edges must be strictly increasing")
        return self


class LightgbmParams(_Section):
    """LightGBM Poisson candidate (``train``): fixed, hand-picked hyperparameters."""

    n_estimators: int = Field(default=200, strict=True, ge=1)
    learning_rate: float = Field(default=0.05, strict=True, gt=0)
    num_leaves: int = Field(default=15, strict=True, ge=2)
    # Large leaves: about 95% of policies have no claim, so small leaves fit noise.
    min_child_samples: int = Field(default=500, strict=True, ge=1)
    subsample: float = Field(default=0.8, strict=True, gt=0, le=1)
    colsample_bytree: float = Field(default=0.8, strict=True, gt=0, le=1)


# Candidate names accepted in `training.candidates`, in the order `train` logs them.
CANDIDATE_NAMES = ("constant", "poisson_glm", "lightgbm", "flaml")
CandidateName = Literal["constant", "poisson_glm", "lightgbm", "flaml"]


class TrainingParams(_Section):
    """Model selection and AutoML settings."""

    # Validation metric that picks the best run. Change it only based on validation results,
    # never after looking at test results (ADR-0002).
    selection_metric: MetricName = MetricName.POISSON_DEVIANCE
    flaml_time_budget_s: int = Field(default=300, strict=True, gt=0)
    # Upper bound on FLAML trials, so a fast machine cannot log thousands of trial runs.
    flaml_max_iter: int = Field(default=500, strict=True, ge=1)
    # Share of the train split FLAML holds out to compare its trials. The validation split is
    # never shown to FLAML: it stays unseen until every candidate is scored on it.
    flaml_holdout_ratio: float = Field(default=0.2, strict=True, gt=0, lt=1)
    # Candidates `train` fits; "constant" is required, since every candidate must beat it.
    candidates: list[CandidateName] = Field(default_factory=lambda: list(CANDIDATE_NAMES))
    # Model seed (AutoML search, model fitting). Separate from the data seed, so retraining with
    # another seed leaves the splits, and so dvc.lock, unchanged.
    seed: int = _seed_field()
    glm: GlmParams = Field(default_factory=GlmParams)
    lightgbm: LightgbmParams = Field(default_factory=LightgbmParams)

    @model_validator(mode="after")
    def _baseline_and_unique_candidates(self) -> Self:
        if "constant" not in self.candidates:
            raise ValueError("training.candidates must include the 'constant' baseline")
        if len(set(self.candidates)) != len(self.candidates):
            raise ValueError("training.candidates must not repeat a candidate")
        return self


class TrafficParams(_Section):
    """``send-traffic`` settings."""

    batch_size: int = Field(default=100, strict=True, ge=1, le=MAX_BATCH_SIZE)


class MonitorParams(_Section):
    """Thresholds of the ``monitor`` checks; failing any check that ran makes ``monitor`` exit 10.

    A check skipped for lack of data never fails the run (docs/contracts.md, What the monitor
    must catch).
    """

    # Fewest predictions in the window for the data drift and prediction drift checks to run;
    # with fewer, both are reported as "skipped (too few predictions)".
    min_predictions: int = Field(default=200, strict=True, ge=1)
    # The data drift check fails when the share of drifted rating factors is >= this value,
    # so 0.2 fails at 2 of the 9 rating factors (about 0.22).
    drift_share_max: float = Field(default=0.2, strict=True, gt=0, le=1)
    # When true, drift detected on predicted_frequency fails the prediction drift check.
    fail_on_prediction_drift: bool = Field(default=True, strict=True)
    # Acceptable range of the live calibration ratio (predicted / actual claims).
    calibration_min: float = Field(default=0.9, strict=True, gt=0)
    calibration_max: float = Field(default=1.1, strict=True, gt=0)
    # The live deviance check fails when live Poisson deviance / reference deviance exceeds this.
    live_deviance_max_ratio: float = Field(default=1.2, strict=True, gt=0)
    # Maximum share of requests rejected with 422.
    rejection_rate_max: float = Field(default=0.05, strict=True, ge=0, le=1)

    @model_validator(mode="after")
    def _ordered_calibration_range(self) -> Self:
        if self.calibration_min >= self.calibration_max:
            raise ValueError("calibration_min must be < calibration_max")
        return self


class Params(_Section):
    """All of ``params.yaml``."""

    # Data seed: used only by `split` and `make-drift`.
    seed: int = _seed_field()
    split: SplitParams = Field(default_factory=SplitParams)
    clean: CleanParams = Field(default_factory=CleanParams)
    training: TrainingParams = Field(default_factory=TrainingParams)
    traffic: TrafficParams = Field(default_factory=TrafficParams)
    monitor: MonitorParams = Field(default_factory=MonitorParams)


def load_params(path: Path | str = PARAMS_PATH) -> Params:
    """Load and validate ``params.yaml``; raises ``pydantic.ValidationError`` if invalid."""
    with open(path, encoding="utf-8") as f:
        return Params.model_validate(yaml.safe_load(f) or {})
