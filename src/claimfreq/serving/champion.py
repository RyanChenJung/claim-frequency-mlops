"""Load the champion model and its identity from the MLflow registry.

The API calls ``load_champion`` at startup. It returns the model and the ``ModelInfoResponse``
that ``/model-info`` serves. ``build_model_info`` turns the registry data into that response,
and needs no MLflow server, so it is easy to test.
"""

import math
from collections.abc import Mapping
from datetime import UTC, datetime

from claimfreq.contracts.api import ModelInfoResponse
from claimfreq.contracts.model import (
    ALIAS_CHAMPION,
    REGISTERED_MODEL_NAME,
    TAG_ALGORITHM,
    TAG_DATA_HASH,
    TAG_GIT_COMMIT,
    TAG_SCHEMA_HASH,
    TAG_SEMVER,
    FrequencyModel,
)

# The model version tags that /model-info needs. register (Role B) sets all of them.
REQUIRED_TAGS = (TAG_SEMVER, TAG_SCHEMA_HASH, TAG_DATA_HASH, TAG_GIT_COMMIT, TAG_ALGORITHM)


def build_model_info(
    version: str,
    tags: Mapping[str, str],
    run_id: str,
    run_start_time_ms: int,
    metrics: Mapping[str, float],
    alias: str = ALIAS_CHAMPION,
) -> ModelInfoResponse:
    """Build the ``/model-info`` body from one registered model version.

    Args:
        version: The registry version number, for example ``"3"``.
        tags: The model version's tags.
        run_id: The MLflow run that produced the model.
        run_start_time_ms: The run's start time, in milliseconds since the epoch (MLflow's unit).
        metrics: The run's logged metrics. Values that are NaN or infinite are left out,
            because JSON has no NaN.
        alias: The alias the API loaded.

    Raises:
        ValueError: if a required tag is missing.
    """
    missing = [tag for tag in REQUIRED_TAGS if tag not in tags]
    if missing:
        raise ValueError(f"model version {version} is missing tags: {', '.join(missing)}")
    return ModelInfoResponse(
        name=REGISTERED_MODEL_NAME,
        version=version,
        semver=tags[TAG_SEMVER],
        alias=alias,
        run_id=run_id,
        trained_at=datetime.fromtimestamp(run_start_time_ms / 1000, tz=UTC),
        schema_hash=tags[TAG_SCHEMA_HASH],
        data_hash=tags[TAG_DATA_HASH],
        git_commit=tags[TAG_GIT_COMMIT],
        algorithm=tags[TAG_ALGORITHM],
        metrics={key: value for key, value in metrics.items() if math.isfinite(value)},
    )


def load_champion(tracking_uri: str) -> tuple[FrequencyModel, ModelInfoResponse]:
    """Load ``claim-frequency@champion`` from MLflow, with its ``/model-info`` body.

    Raises any MLflow error, for example when no champion exists yet. The caller retries.
    """
    # Imported here, because MLflow is slow to import and tests with an injected model skip it.
    import mlflow
    from mlflow import MlflowClient

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    model_version = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, ALIAS_CHAMPION)
    run = client.get_run(model_version.run_id)
    # Load by version number, not by alias, so the model always matches the tags read above,
    # even if someone moves the alias between the two calls.
    model = mlflow.pyfunc.load_model(f"models:/{REGISTERED_MODEL_NAME}/{model_version.version}")
    info = build_model_info(
        version=str(model_version.version),
        tags=model_version.tags,
        run_id=model_version.run_id,
        run_start_time_ms=run.info.start_time,
        metrics=run.data.metrics,
    )
    return model, info
