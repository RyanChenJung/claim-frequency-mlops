"""Role B (Modeling): the ``claimfreq register`` and ``claimfreq promote`` commands.

``register`` (an Airflow task, right after ``train``):

1. Finds the most recent ``train`` parent run whose status is ``FINISHED`` and considers only its
   ``FINISHED`` direct children (docs/contracts.md, Model: candidate scope). If newer parents
   ended ``FAILED`` (or are still running), it falls back to that older candidate set and says so.
2. Picks the best child by ``val_<training.selection_metric>`` with ``is_better``. A child whose
   value is NaN or missing is never picked; on a tie the earlier child wins.
3. If that child is already a registered version, registers nothing and only makes sure the
   ``challenger`` alias points at it (unless it is the champion), so a retry repairs an alias
   call that failed. This check runs first, so a re-run is always a no-op.
4. Otherwise it checks that the train split under ``DATA_DIR`` is the file the run was trained
   on (md5 = the run's ``data_hash``), because steps 5 and 6 read the validation split.
5. Logs the comparison with the ``champion`` as tags on the child run (keys below): the champion
   version is re-scored on the current validation split, the same holdout as the challenger.
6. Logs the reference dataset (validation split + the challenger's predictions) on the child run.
7. Registers the child's logged model (an MLflow logged model named ``model``, linked to the
   child run) as ``claim-frequency``, with every model-version tag, and sets alias ``challenger``.

``promote --version N`` (a human, ADR-0007) moves the ``champion`` alias to version N. If the
``challenger`` alias pointed at N, it is removed: the challenger is the newest best model *not
yet promoted*.

Semantic version of a new version (ADR-0007):

- nothing registered yet: ``1.0.0``;
- versions exist but there is no champion: MINOR bump (the schema is not compared, since no
  client is served yet);
- with a champion, compared with it: ``schema_hash`` differs → MAJOR; otherwise ``data_hash``
  or ``algorithm`` differs → MINOR; otherwise PATCH (for example a hyperparameter change).

The bump is applied to the highest ``semver`` registered so far, not to the champion's, so two
challengers registered before a promotion never share a version string. Consequence: the first
champion is ``1.0.0`` only if it is the first version ever registered. Before the real first
training run, reset the registry with ``MlflowClient().delete_registered_model("claim-frequency")``
(runs and artifacts stay), and run ``register`` only once before the first ``promote``.
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

# Comparison with the champion, logged as **tags** on the challenger's (child) run. They are tags,
# not metrics, so the run's metrics stay the model's own: `/model-info` reports a run's metrics,
# and the metric-key rule allows only the val_, test_, and live_ prefixes. Role B's own keys: no
# other role reads them; they are for the person who decides on `promote`.
# The champion re-scored on this validation split, e.g. champion_val_poisson_deviance = "0.45".
CHAMPION_TAG_PREFIX = "champion_"
# The champion's registry version and semver, or "none" when no champion exists yet.
TAG_CHAMPION_VERSION = "champion_version"
TAG_CHAMPION_SEMVER = "champion_semver"
NO_CHAMPION = "none"
# "true" / "false": the challenger is / is not better on the selection metric (is_better).
# "no_champion": nothing to compare with. "incomparable": the schema changed, so the champion
# cannot score the new split (its signature would reject it).
TAG_BEATS_CHAMPION = "beats_champion"
NO_CHAMPION_COMPARISON = "no_champion"
INCOMPARABLE = "incomparable"


class RegistryCheckError(Exception):
    """A precondition of ``register`` or ``promote`` failed; the CLI exits 10 with the message."""


@dataclass(frozen=True)
class Registration:
    """What ``register`` did, for the CLI to print. ``version`` is the registry version."""

    version: str
    semver: str
    run_id: str
    algorithm: str
    # The train parent run whose candidates were considered.
    candidate_set: str
    # Number of newer train parents skipped because they did not finish (FAILED or running).
    skipped_parents: int = 0
    already_registered: bool = False


@dataclass(frozen=True)
class Promotion:
    """What ``promote`` did, for the CLI to print."""

    target: ModelVersion
    previous: ModelVersion | None
    # True if the challenger alias pointed at the promoted version and was removed.
    cleared_challenger: bool


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


def latest_finished_parent(client: MlflowClient, experiment_id: str) -> tuple[Run, int]:
    """Return the most recent ``FINISHED`` train parent and how many newer parents it skipped.

    Raises:
        RegistryCheckError: if no train parent run has finished.
    """
    # A parent carries candidate_set but, unlike its children, no mlflow.parentRunId tag.
    runs = client.search_runs(
        [experiment_id],
        filter_string=f"tags.{TAG_CANDIDATE_SET} != ''",
        order_by=["attributes.start_time DESC"],
    )
    skipped = 0
    for run in runs:
        if "mlflow.parentRunId" in run.data.tags:
            continue
        if run.info.status == FINISHED:
            return run, skipped
        skipped += 1
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


def current_champion(client: MlflowClient) -> ModelVersion | None:
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


def champion_comparison(
    champion: ModelVersion | None,
    validation: pd.DataFrame,
    selection_metric: str,
    challenger_value: float,
    challenger_schema_hash: str,
) -> dict[str, str]:
    """Return the comparison tags of a challenger against ``champion`` (keys above).

    The champion version is loaded by number (not through the alias, which a concurrent
    ``promote`` could move) and re-scored on ``validation``.
    """
    if champion is None:
        return {
            TAG_CHAMPION_VERSION: NO_CHAMPION,
            TAG_CHAMPION_SEMVER: NO_CHAMPION,
            TAG_BEATS_CHAMPION: NO_CHAMPION_COMPARISON,
        }
    tags = {
        TAG_CHAMPION_VERSION: str(champion.version),
        TAG_CHAMPION_SEMVER: champion.tags.get(TAG_SEMVER, ""),
    }
    if champion.tags.get(TAG_SCHEMA_HASH) != challenger_schema_hash:
        return {**tags, TAG_BEATS_CHAMPION: INCOMPARABLE}
    X, y, w = split_xy_w(validation)
    model = mlflow.pyfunc.load_model(f"models:/{REGISTERED_MODEL_NAME}/{champion.version}")
    predictions = np.asarray(model.predict(schema.coerce_features(X)), dtype=float)
    scores = metrics.compute_all(y, predictions, w)
    for name, value in metrics.with_prefix(scores, MetricPrefix.VAL).items():
        tags[f"{CHAMPION_TAG_PREFIX}{name}"] = repr(float(value))
    beats = is_better(selection_metric, challenger_value, scores[selection_metric])
    tags[TAG_BEATS_CHAMPION] = str(beats).lower()
    return tags


def _ensure_challenger(client: MlflowClient, version: ModelVersion) -> None:
    """Point ``challenger`` at ``version`` unless it is the champion; safe to repeat."""
    champion = current_champion(client)
    if champion is None or str(champion.version) != str(version.version):
        client.set_registered_model_alias(REGISTERED_MODEL_NAME, ALIAS_CHALLENGER, version.version)


def register(params: Params | None = None) -> Registration:
    """Register the best candidate of the latest finished ``train`` run as ``challenger``.

    Raises:
        RegistryCheckError: no finished train run, no scored candidate, a run without its
            logged model, or (for a run not registered yet) a train split under ``DATA_DIR``
            that is not the file the run was trained on.
    """
    params = params or load_params()
    selection_metric = params.training.selection_metric.value
    mlflow.set_tracking_uri(env.get(env.MLFLOW_TRACKING_URI))
    client = MlflowClient()
    experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    if experiment is None:
        raise RegistryCheckError(
            "register: no FINISHED train run found; run `claimfreq train` first"
        )

    parent, skipped = latest_finished_parent(client, experiment.experiment_id)
    best = select_best(
        finished_children(client, experiment.experiment_id, parent.info.run_id), selection_metric
    )
    run_tags = best.data.tags
    found = {
        "run_id": best.info.run_id,
        "algorithm": run_tags[TAG_ALGORITHM],
        "candidate_set": parent.info.run_id,
        "skipped_parents": skipped,
    }

    # Idempotent first: a retry registers nothing, but repairs the challenger alias.
    versions = _registered_versions(client)
    for existing in versions:
        if existing.run_id == best.info.run_id:
            _ensure_challenger(client, existing)
            return Registration(
                version=str(existing.version),
                semver=existing.tags.get(TAG_SEMVER, ""),
                already_registered=True,
                **found,
            )

    # The comparison and the reference read the validation split: it must be the data version
    # this run was trained with. The train split's md5 is the run's data_hash.
    train_hash = file_md5(paths.data_path(paths.TRAIN))
    if train_hash != run_tags[TAG_DATA_HASH]:
        raise RegistryCheckError(
            f"register: {paths.data_path(paths.TRAIN)} has md5 {train_hash}, but the train run "
            f"{parent.info.run_id} used {run_tags[TAG_DATA_HASH]}; check out the matching data "
            "before registering"
        )

    champion = current_champion(client)
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
    comparison = champion_comparison(
        champion, validation, selection_metric, value, run_tags[TAG_SCHEMA_HASH]
    )
    for key, tag_value in comparison.items():
        client.set_tag(best.info.run_id, key, tag_value)
    model_uri = logged_model_uri(client, best)
    model = mlflow.pyfunc.load_model(model_uri)
    X_val, _, _ = split_xy_w(validation)
    predictions = model.predict(schema.coerce_features(X_val))
    _log_reference(client, best.info.run_id, reference_frame(validation, predictions))

    registered = mlflow.register_model(model_uri, REGISTERED_MODEL_NAME, tags=version_tags)
    client.set_registered_model_alias(REGISTERED_MODEL_NAME, ALIAS_CHALLENGER, registered.version)
    return Registration(version=str(registered.version), semver=semver, **found)


# --- Promote ------------------------------------------------------------------------------------


def promote(version: int) -> Promotion:
    """Move alias ``champion`` to ``version``; clear ``challenger`` if it pointed there.

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
    previous = current_champion(client)
    client.set_registered_model_alias(REGISTERED_MODEL_NAME, ALIAS_CHAMPION, target.version)
    cleared = (
        ALIAS_CHALLENGER in client.get_model_version(REGISTERED_MODEL_NAME, target.version).aliases
    )
    if cleared:
        client.delete_registered_model_alias(REGISTERED_MODEL_NAME, ALIAS_CHALLENGER)
    return Promotion(target=target, previous=previous, cleared_challenger=cleared)
