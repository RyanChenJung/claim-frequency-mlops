"""API contract: endpoint paths, request and response bodies, and error semantics.

Pydantic models only, with no FastAPI dependency, so that:

- Role D (Serving) uses them as FastAPI request and response models.
- Role D's ``send-traffic`` builds requests and parses responses with the same classes.
- Role C (Platform) knows the paths for health checks and smoke tests.

Endpoints:

- ``POST /predict``: body ``PredictRequest``, response ``PredictResponse`` (HTTP 200).
- ``GET /health``: readiness, response ``HealthResponse``. HTTP 200 with
  ``{"status": "ok", "model_loaded": true}`` once the champion is loaded; HTTP 503 with
  ``{"status": "degraded", "model_loaded": false}`` before that, so the compose healthcheck
  and ``send-traffic`` can wait for the 200.
- ``GET /model-info``: response ``ModelInfoResponse`` (HTTP 200). ``metrics`` omits any metric
  whose value is not finite (for example a NaN Gini), because JSON has no NaN.

Errors:

- **422** (invalid request): FastAPI's default validation-error body, unchanged::

      {"detail": [{"loc": ["body", "policies", 0, "DrivAge"],
                   "msg": "Input should be less than or equal to 100",
                   "type": "less_than_equal", ...}, ...]}

  ``detail`` is a list of objects with at least ``loc`` (list of str and int), ``msg`` (str),
  and ``type`` (str). Every 422 is also written to ``rejected_request_log``.
- **503** (model not loaded yet): ``/predict`` and ``/model-info`` return
  ``{"detail": "model not loaded"}`` until the champion has loaded at startup, and ``/health``
  returns its ``"degraded"`` body.
"""

from typing import Annotated, Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, FiniteFloat

from claimfreq.contracts.schema import PolicyFeatures

PATH_PREDICT = "/predict"
PATH_HEALTH = "/health"
PATH_MODEL_INFO = "/model-info"

MAX_BATCH_SIZE = 1000
MAX_BATCH_ID_LENGTH = 64
# prediction_log.policy_id is a BigInteger (signed int64) column.
MAX_POLICY_ID = 2**63 - 1

# A lowercase, hyphenated uuid4 string, as produced by str(uuid.uuid4()).
UuidStr = Annotated[
    str,
    Field(pattern=r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"),
]

# --- Request ------------------------------------------------------------------------------------


class PolicyInput(PolicyFeatures):
    """One policy to score: the 9 rating factors plus an optional ``policy_id``.

    ``policy_id`` is the ``IDpol`` of the policy, if the client has one. It is echoed in the
    response and logged, so delayed ground truth can be traced back to the policy.
    """

    policy_id: int | None = Field(default=None, strict=True, ge=1, le=MAX_POLICY_ID)


class PredictRequest(BaseModel):
    """Body of ``POST /predict``: 1 to ``MAX_BATCH_SIZE`` policies scored in one call."""

    model_config = ConfigDict(extra="forbid")

    # Identifies one `send-traffic` run, shared by every request of that run; logged with each
    # row, so `monitor --batch-id` can isolate the run.
    batch_id: str | None = Field(default=None, max_length=MAX_BATCH_ID_LENGTH)
    policies: list[PolicyInput] = Field(min_length=1, max_length=MAX_BATCH_SIZE)


# --- Responses ----------------------------------------------------------------------------------


class ModelRef(BaseModel):
    """The registered model version that produced a response."""

    name: str  # registered model name, "claim-frequency"
    version: str  # MLflow registry version number, as a string (e.g. "3")
    semver: str  # the version's semver tag (e.g. "1.2.0")


class Prediction(BaseModel):
    """The prediction for one policy, in the same position as in the request."""

    # Primary key of prediction_log, and the key that ground truth joins on.
    prediction_id: UuidStr
    policy_id: int | None
    # Expected claims per policy-year; inf and NaN are rejected, since JSON cannot carry them.
    predicted_frequency: float = Field(gt=0, allow_inf_nan=False)


class PredictResponse(BaseModel):
    """Body of a successful ``POST /predict``."""

    request_id: UuidStr  # one per HTTP request; shared by every row it logs
    model: ModelRef
    predictions: list[Prediction]  # in input order, one per policy


class HealthResponse(BaseModel):
    """Body of ``GET /health``: ``ok`` with HTTP 200 once ready, ``degraded`` with HTTP 503."""

    status: Literal["ok", "degraded"]  # "degraded" until the champion is loaded
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    """Body of ``GET /model-info``: the served model version and its lineage.

    The lineage fields are copied from the model version's tags (``contracts.model``).
    """

    name: str
    version: str  # MLflow registry version number, as a string
    semver: str
    alias: str  # the alias the API loaded, normally "champion"
    run_id: str  # MLflow run that produced the model
    trained_at: AwareDatetime  # start time of that run, timezone-aware UTC
    schema_hash: str
    data_hash: str
    git_commit: str
    algorithm: str
    # The run's logged metrics by full key, e.g. {"val_poisson_deviance": 0.31, ...}. Finite
    # values only: the API omits a metric whose value is NaN or infinite.
    metrics: dict[str, FiniteFloat]
