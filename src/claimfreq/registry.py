"""Role B (Modeling): the ``claimfreq register`` and ``claimfreq promote`` commands.

``register`` (an Airflow task, right after ``train``):

1. Finds the most recent ``train`` parent run whose status is ``FINISHED`` and considers only its
   ``FINISHED`` direct children (docs/contracts.md, Model: candidate scope).
2. Picks the best child by ``val_<training.selection_metric>`` with ``is_better``. A child whose
   value is NaN or missing is never picked.
3. Registers that child's already-logged Pipeline (an MLflow logged model named ``model``,
   linked to the child run) as
   ``claim-frequency``, with every model-version tag and the alias ``challenger``. If that run
   is already registered, nothing is registered twice: the command reports it and exits 0.
4. Logs the reference dataset (validation split + predictions) on the child run.
5. Compares the challenger with the ``champion`` on the same holdout: the current validation
   split. The champion is re-scored on it, so both numbers come from the same data even when
   the champion was trained on an older data version. The comparison keys are below.

``promote --version N`` (a human, ADR-0007) moves the ``champion`` alias to version N.

Semantic version of a new challenger (ADR-0007), against the champion:

- no champion yet: ``1.0.0``;
- ``schema_hash`` differs: MAJOR (the input contract changed);
- otherwise ``data_hash`` or ``algorithm`` differs: MINOR (predictions change, code does not);
- otherwise: PATCH (same data and algorithm, for example a hyperparameter change).

The bump is applied to the highest ``semver`` registered so far, not to the champion's, so two
challengers registered before any promotion never share a version string.
"""

from __future__ import annotations

import math
import tempfile
from dataclasses import dataclass
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from mlflow.entities import Run, RunStatus
from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import (
    INVALID_PARAMETER_VALUE,
    RESOURCE_DOES_NOT_EXIST,
    ErrorCode,
)
from mlflow.tracking import MlflowClient

from claimfreq import metrics
from claimfreq.contracts import env, paths, schema
from claimfreq.contracts.model import (
    ALIAS_CHALLENGER,
    ALIAS_CHAMPION,
    MLFLOW_EXPERIMENT_NAME,
    MODEL_ARTIFACT_PATH,
    PREDICTED_FREQUENCY,
    REFERENCE_ARTIFACT_PATH,
    REFERENCE_COLUMNS,
    REGISTERED_MODEL_NAME,
    TAG_ALGORITHM,
    TAG_CANDIDATE_SET,
    TAG_DATA_HASH,
    TAG_GIT_COMMIT,
    TAG_SCHEMA_HASH,
    TAG_SELECTION_METRIC,
    TAG_SELECTION_METRIC_VALUE,
    TAG_SEMVER,
    MetricPrefix,
    file_md5,
    is_better,
    split_xy_w,
)
from claimfreq.contracts.params import Params, load_params
from claimfreq.training import read_split

FIRST_VERSION = "1.0.0"
# MLflow's string for a run that ended without an exception.
FINISHED = RunStatus.to_string(RunStatus.FINISHED)

# Comparison with the champion, logged on the challenger's (child) run. Role B's own keys: no
# other role reads them; they are for the person who decides on `promote`.
# Metrics: the champion re-scored on this validation split, e.g. champion_val_poisson_deviance.
CHAMPION_METRIC_PREFIX = "champion_"
# Tags: the champion's registry version and semver, or "none" when no champion exists yet.
TAG_CHAMPION_VERSION = "champion_version"
TAG_CHAMPION_SEMVER = "champion_semver"
NO_CHAMPION = "none"
# Tag: "true" if the challenger beats the champion on the selection metric, "false" if not,
# "no_champion" when there is nothing to compare with.
TAG_BEATS_CHAMPION = "beats_champion"
NO_CHAMPION_COMPARISON = "no_champion"


class RegistryCheckError(Exception):
    """A precondition of ``register`` or ``promote`` failed; the CLI exits 10 with the message."""


@dataclass(frozen=True)
class Registration:
    """What ``register`` did, for the CLI to print. ``version`` is the registry version."""

    version: str
    semver: str
    run_id: str
    algorithm: str
    already_registered: bool = False


# --- Semantic versions --------------------------------------------------------------------------


def parse_semver(value: str) -> tuple[int, int, int]:
    """Parse ``"MAJOR.MINOR.PATCH"`` into a tuple of ints; raise ``ValueError`` otherwise."""
    parts = value.split(".")
    if len(parts) != 3 or not all(part.isdigit() for part in parts):
        raise ValueError(f"not a semantic version: {value!r}")
    major, minor, patch = (int(part) for part in parts)
    return major, minor, patch


def next_semver(
    candidate_tags: dict[str, str],
    champion_tags: dict[str, str] | None,
    highest: str | None,
) -> str:
    """Return the semver of a new challenger (module docstring, ADR-0007).

    Args:
        candidate_tags: lineage tags of the challenger's run.
        champion_tags: model-version tags of the champion, or None if there is none.
        highest: the highest ``semver`` registered so far, or None if nothing is registered.
    """
    if champion_tags is None and highest is None:
        return FIRST_VERSION
    major, minor, patch = parse_semver(highest or champion_tags[TAG_SEMVER])
    if champion_tags is None:
        # Versions exist but none was ever promoted: the challenger replaces the last one.
        return f"{major}.{minor + 1}.0"
    if candidate_tags[TAG_SCHEMA_HASH] != champion_tags.get(TAG_SCHEMA_HASH):
        return f"{major + 1}.0.0"
    if any(candidate_tags[key] != champion_tags.get(key) for key in (TAG_DATA_HASH, TAG_ALGORITHM)):
        return f"{major}.{minor + 1}.0"
    return f"{major}.{minor}.{patch + 1}"


def highest_semver(versions: list[ModelVersion]) -> str | None:
    """Return the highest valid ``semver`` tag among registered versions, or None."""
    semvers = []
    for version in versions:
        try:
            semvers.append(parse_semver(version.tags.get(TAG_SEMVER, "")))
        except ValueError:
            continue
    if not semvers:
        return None
    return ".".join(str(part) for part in max(semvers))


# --- Selection ----------------------------------------------------------------------------------


def latest_finished_parent(client: MlflowClient, experiment_id: str) -> Run:
    """Return the most recent ``FINISHED`` train parent run; raise ``RegistryCheckError``."""
    # A parent carries candidate_set but, unlike its children, no mlflow.parentRunId tag.
    runs = client.search_runs(
        [experiment_id],
        filter_string=(f"tags.{TAG_CANDIDATE_SET} != '' and attributes.status = '{FINISHED}'"),
        order_by=["attributes.start_time DESC"],
    )
    for run in runs:
        if "mlflow.parentRunId" not in run.data.tags:
            return run
    raise RegistryCheckError("register: no FINISHED train run found; run `claimfreq train` first")


def finished_children(client: MlflowClient, experiment_id: str, parent_id: str) -> list[Run]:
    """Return the ``FINISHED`` direct children of ``parent_id``, oldest first."""
    return list(
        client.search_runs(
            [experiment_id],
            filter_string=(
                f"tags.mlflow.parentRunId = '{parent_id}' and attributes.status = '{FINISHED}'"
            ),
            order_by=["attributes.start_time ASC"],
        )
    )


def select_best(children: list[Run], selection_metric: str) -> Run:
    """Return the child with the best ``val_<selection_metric>``; raise if none has a value.

    ``is_better`` is strict, so on a tie the earlier child (the simpler candidate, since
    ``train`` logs the baseline first) wins. NaN values never win.
    """
    key = f"{MetricPrefix.VAL}{selection_metric}"
    best: Run | None = None
    for child in children:
        value = child.data.metrics.get(key)
        if value is None or math.isnan(value):
            continue
        if best is None or is_better(selection_metric, value, best.data.metrics[key]):
            best = child
    if best is None:
        raise RegistryCheckError(f"register: no candidate of the latest train run has {key}")
    return best


# --- Register -----------------------------------------------------------------------------------


# Error codes of get_model_version_by_alias when there is simply no champion: the registered
# model does not exist yet, or it has no `champion` alias. Any other error (for example the
# tracking server being unreachable) propagates, so it is never mistaken for "no champion".
_NO_CHAMPION_ERRORS = {
    ErrorCode.Name(RESOURCE_DOES_NOT_EXIST),
    ErrorCode.Name(INVALID_PARAMETER_VALUE),
}


def _champion(client: MlflowClient) -> ModelVersion | None:
    """Return the version with alias ``champion``, or None if the alias or model is missing."""
    try:
        return client.get_model_version_by_alias(REGISTERED_MODEL_NAME, ALIAS_CHAMPION)
    except MlflowException as error:
        if error.error_code in _NO_CHAMPION_ERRORS:
            return None
        raise


def logged_model_uri(client: MlflowClient, run: Run) -> str:
    """Return ``models:/<model_id>`` of the Pipeline that ``train`` logged in ``run``.

    In MLflow 3, ``mlflow.sklearn.log_model`` creates a logged model linked to the run (listed
    in the run's outputs) rather than a run artifact, so it is registered by its model id. The
    registered version's ``run_id`` is still this run, which holds the reference dataset.
    """
    for output in run.outputs.model_outputs:
        if mlflow.get_logged_model(output.model_id).name == MODEL_ARTIFACT_PATH:
            return f"models:/{output.model_id}"
    raise RegistryCheckError(
        f"register: run {run.info.run_id} has no logged model named {MODEL_ARTIFACT_PATH!r}"
    )


def _registered_versions(client: MlflowClient) -> list[ModelVersion]:
    return list(client.search_model_versions(f"name = '{REGISTERED_MODEL_NAME}'"))


def reference_frame(validation: pd.DataFrame, predictions: np.ndarray) -> pd.DataFrame:
    """Return the reference dataset: validation features and targets plus the predictions."""
    return validation.assign(**{PREDICTED_FREQUENCY: np.asarray(predictions, dtype=float)})[
        REFERENCE_COLUMNS
    ]


def _log_reference(client: MlflowClient, run_id: str, reference: pd.DataFrame) -> None:
    """Log ``reference`` on ``run_id`` at ``REFERENCE_ARTIFACT_PATH``."""
    artifact_dir, file_name = REFERENCE_ARTIFACT_PATH.rsplit("/", 1)
    with tempfile.TemporaryDirectory() as tmp:
        local = Path(tmp) / file_name
        reference.to_parquet(local, index=False)
        client.log_artifact(run_id, str(local), artifact_path=artifact_dir)


def _log_champion_comparison(
    client: MlflowClient,
    run_id: str,
    champion: ModelVersion | None,
    validation: pd.DataFrame,
    selection_metric: str,
    challenger_value: float,
) -> None:
    """Re-score the champion on ``validation`` and log the comparison on the challenger's run."""
    if champion is None:
        client.set_tag(run_id, TAG_CHAMPION_VERSION, NO_CHAMPION)
        client.set_tag(run_id, TAG_BEATS_CHAMPION, NO_CHAMPION_COMPARISON)
        return
    X, y, w = split_xy_w(validation)
    model = mlflow.pyfunc.load_model(f"models:/{REGISTERED_MODEL_NAME}@{ALIAS_CHAMPION}")
    predictions = np.asarray(model.predict(schema.coerce_features(X)), dtype=float)
    scores = metrics.compute_all(y, predictions, w)
    for name, value in metrics.with_prefix(scores, MetricPrefix.VAL).items():
        client.log_metric(run_id, f"{CHAMPION_METRIC_PREFIX}{name}", value)
    client.set_tag(run_id, TAG_CHAMPION_VERSION, str(champion.version))
    client.set_tag(run_id, TAG_CHAMPION_SEMVER, champion.tags.get(TAG_SEMVER, ""))
    beats = is_better(selection_metric, challenger_value, scores[selection_metric])
    client.set_tag(run_id, TAG_BEATS_CHAMPION, str(beats).lower())


def register(params: Params | None = None) -> Registration:
    """Register the best candidate of the latest finished ``train`` run as ``challenger``.

    Raises:
        RegistryCheckError: no finished train run, no scored candidate, or the validation and
            train splits under ``DATA_DIR`` are not the data that train run used.
    """
    params = params or load_params()
    selection_metric = params.training.selection_metric.value
    mlflow.set_tracking_uri(env.get(env.MLFLOW_TRACKING_URI))
    client = MlflowClient()
    experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    if experiment is None:
        raise RegistryCheckError("register: no train run found; run `claimfreq train` first")

    parent = latest_finished_parent(client, experiment.experiment_id)
    best = select_best(
        finished_children(client, experiment.experiment_id, parent.info.run_id), selection_metric
    )
    run_tags = best.data.tags
    # The reference and the comparison must be computed on the data this run was trained with.
    train_hash = file_md5(paths.data_path(paths.TRAIN))
    if train_hash != run_tags[TAG_DATA_HASH]:
        raise RegistryCheckError(
            f"register: {paths.data_path(paths.TRAIN)} has md5 {train_hash}, but the latest train "
            f"run used {run_tags[TAG_DATA_HASH]}; check out the matching data before registering"
        )

    versions = _registered_versions(client)
    for existing in versions:
        if existing.run_id == best.info.run_id:
            return Registration(
                version=str(existing.version),
                semver=existing.tags.get(TAG_SEMVER, ""),
                run_id=best.info.run_id,
                algorithm=run_tags[TAG_ALGORITHM],
                already_registered=True,
            )

    champion = _champion(client)
    semver = next_semver(
        run_tags, champion.tags if champion is not None else None, highest_semver(versions)
    )
    value = best.data.metrics[f"{MetricPrefix.VAL}{selection_metric}"]
    version_tags = {
        TAG_SEMVER: semver,
        TAG_SCHEMA_HASH: run_tags[TAG_SCHEMA_HASH],
        TAG_DATA_HASH: run_tags[TAG_DATA_HASH],
        TAG_GIT_COMMIT: run_tags[TAG_GIT_COMMIT],
        TAG_ALGORITHM: run_tags[TAG_ALGORITHM],
        TAG_SELECTION_METRIC: selection_metric,
        TAG_SELECTION_METRIC_VALUE: repr(float(value)),
    }

    validation = read_split(paths.data_path(paths.VALIDATION), params)
    _log_champion_comparison(
        client, best.info.run_id, champion, validation, selection_metric, value
    )
    model_uri = logged_model_uri(client, best)
    model = mlflow.pyfunc.load_model(model_uri)
    X_val, _, _ = split_xy_w(validation)
    predictions = model.predict(schema.coerce_features(X_val))
    _log_reference(client, best.info.run_id, reference_frame(validation, predictions))

    registered = mlflow.register_model(model_uri, REGISTERED_MODEL_NAME, tags=version_tags)
    client.set_registered_model_alias(REGISTERED_MODEL_NAME, ALIAS_CHALLENGER, registered.version)
    return Registration(
        version=str(registered.version),
        semver=semver,
        run_id=best.info.run_id,
        algorithm=run_tags[TAG_ALGORITHM],
    )


# --- Promote ------------------------------------------------------------------------------------


def promote(version: int) -> tuple[ModelVersion, ModelVersion | None]:
    """Move alias ``champion`` to ``version``; return the new and the previous champion.

    Raises:
        RegistryCheckError: if ``claim-frequency`` has no version ``version``.
    """
    mlflow.set_tracking_uri(env.get(env.MLFLOW_TRACKING_URI))
    client = MlflowClient()
    try:
        target = client.get_model_version(REGISTERED_MODEL_NAME, str(version))
    except MlflowException as error:
        if error.error_code == ErrorCode.Name(RESOURCE_DOES_NOT_EXIST):
            raise RegistryCheckError(
                f"promote: {REGISTERED_MODEL_NAME} has no version {version}"
            ) from None
        raise
    previous = _champion(client)
    client.set_registered_model_alias(REGISTERED_MODEL_NAME, ALIAS_CHAMPION, target.version)
    return target, previous
