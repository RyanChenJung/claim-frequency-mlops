"""Monitoring log contract: the tables of the ``monitoring`` database.

Role D's API writes ``prediction_log`` (every scored policy) and ``rejected_request_log``
(every HTTP 422). Role D's ``send-traffic`` writes ``ground_truth`` after the predictions, to
simulate claim outcomes that arrive later. Role D's ``monitor`` reads all three tables and
compares them with the reference dataset.

The connection URL comes from the ``MONITORING_DB_URL`` environment variable (see ``env.py``).
The tables are SQLAlchemy 2 Core ``Table`` objects that work on PostgreSQL (compose) and SQLite
(tests and local development).

Conventions:

- Every writer and reader connects with ``make_engine``, so SQLite enforces the foreign key
  just as PostgreSQL does.
- Every writer and reader also calls ``create_all`` before its first query: the API at startup,
  ``send-traffic`` and ``monitor`` when they start. It is safe to call repeatedly; the API
  creates the tables before it reports ready.
- Column names match the schema and the reference dataset: the feature columns in
  ``prediction_log`` (``DrivAge``, ...) and ``ClaimNb`` and ``Exposure`` in ``ground_truth``, so
  a query result lines up with the reference dataset without renaming. PostgreSQL folds
  unquoted identifiers to lower case, so raw SQL must quote them: ``SELECT "DrivAge" FROM
  prediction_log``. SQLAlchemy Core queries quote them automatically.
- Timestamps are timezone-aware UTC. SQLite does not store the offset, so values read back from
  SQLite are naive datetimes in UTC.
- The rejection rate for a time window is
  ``rejected requests / (rejected requests + distinct request_id in prediction_log)``.
"""

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    event,
)
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError, OperationalError, ProgrammingError

from claimfreq.contracts.api import PredictRequest, PredictResponse
from claimfreq.contracts.schema import CATEGORICAL_FEATURES, CLAIM_NB, EXPOSURE, FEATURE_COLUMNS

metadata = MetaData()

# Same names as the schema; integer rating factors are int64 in frames, hence BigInteger.
_feature_columns = [
    Column(name, String, nullable=False)
    if name in CATEGORICAL_FEATURES
    else Column(name, BigInteger, nullable=False)
    for name in FEATURE_COLUMNS
]

prediction_log = Table(
    "prediction_log",
    metadata,
    Column("prediction_id", String(36), primary_key=True),
    Column("request_id", String(36), nullable=False, index=True),
    Column("batch_id", String(64), nullable=True, index=True),
    Column("received_at", DateTime(timezone=True), nullable=False, index=True),
    Column("model_name", String, nullable=False),
    Column("model_version", String, nullable=False),
    Column("model_semver", String, nullable=False),
    Column("policy_id", BigInteger, nullable=True),
    *_feature_columns,
    Column("predicted_frequency", Float, nullable=False),
)
"""One row per scored policy, written by the API for every successful ``/predict``."""

rejected_request_log = Table(
    "rejected_request_log",
    metadata,
    Column("request_id", String(36), primary_key=True),
    Column("received_at", DateTime(timezone=True), nullable=False),
    # Set when the body is parseable JSON with a string `batch_id` of at most 64 characters;
    # otherwise NULL.
    Column("batch_id", String(64), nullable=True),
    Column("status_code", Integer, nullable=False),
    # The ``type`` of the first error in the 422 body, e.g. "extra_forbidden", "int_type",
    # "less_than_equal", or "json_invalid".
    Column("error_type", String, nullable=False),
    # The full 422 ``detail`` list, JSON-encoded.
    Column("error_detail", Text, nullable=False),
    # Length of the body's "policies" list when parseable; otherwise NULL.
    Column("n_policies", Integer, nullable=True),
)
"""One row per rejected request (HTTP 422). Feeds the rejection-rate signal."""

ground_truth = Table(
    "ground_truth",
    metadata,
    Column(
        "prediction_id",
        String(36),
        ForeignKey("prediction_log.prediction_id"),
        primary_key=True,
    ),
    Column(CLAIM_NB, Integer, nullable=False),
    Column(EXPOSURE, Float, nullable=False),
    Column("observed_at", DateTime(timezone=True), nullable=False),
    # Quoted, because PostgreSQL would otherwise fold the mixed-case names to lower case.
    CheckConstraint(f'"{CLAIM_NB}" >= 0', name="ck_ground_truth_claim_nb"),
    CheckConstraint(f'"{EXPOSURE}" > 0', name="ck_ground_truth_exposure"),
)
"""One row per scored policy whose outcome became known later (delayed ground truth)."""


def _enable_sqlite_foreign_keys(dbapi_connection: Any, _connection_record: Any) -> None:
    """Turn on foreign keys for one SQLite connection; SQLite leaves them off by default."""
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


def make_engine(url: str) -> Engine:
    """Create the engine for ``MONITORING_DB_URL``. Every writer and reader uses this.

    On SQLite it enables foreign keys on every new connection, so the ``ground_truth`` foreign
    key holds in tests and local development as it does on PostgreSQL.
    """
    engine = create_engine(url)
    if engine.dialect.name == "sqlite":
        event.listen(engine, "connect", _enable_sqlite_foreign_keys)
    return engine


def create_all(engine: Engine) -> None:
    """Create the three tables if they do not exist yet. Safe to call repeatedly.

    Two processes that start together can race: both see a table missing, both create it, and
    the slower one fails. SQLite reports the race as ``OperationalError`` (table already exists);
    PostgreSQL as ``ProgrammingError`` (duplicate table) or, while both transactions are still
    open, ``IntegrityError`` (duplicate row in its type catalog). The call is retried once per
    table: PostgreSQL creates all tables in one transaction, so one retry suffices there, but
    SQLite commits each ``CREATE TABLE`` on its own, so a loser can lose again on a later table.
    """
    attempts = len(metadata.sorted_tables)
    for attempt in range(attempts + 1):
        try:
            metadata.create_all(engine, checkfirst=True)
            return
        except (OperationalError, ProgrammingError, IntegrityError):
            if attempt == attempts:
                raise


def prediction_log_rows(
    request: PredictRequest, response: PredictResponse, received_at: datetime
) -> list[dict[str, Any]]:
    """Build the ``prediction_log`` rows for one successful ``/predict`` call.

    Pairs each policy with its prediction by position (the API returns predictions in input
    order). The result can be passed directly to ``connection.execute(insert(prediction_log),
    rows)``.

    Args:
        request: The validated request body.
        response: The response returned to the client.
        received_at: When the request arrived, as a timezone-aware datetime. It is stored in
            UTC. A naive datetime raises ``ValueError``, because its time zone would be a guess.
    """
    if received_at.utcoffset() is None:
        raise ValueError("received_at must be timezone-aware")
    received_at = received_at.astimezone(UTC)
    rows = []
    for policy, prediction in zip(request.policies, response.predictions, strict=True):
        rows.append(
            {
                "prediction_id": prediction.prediction_id,
                "request_id": response.request_id,
                "batch_id": request.batch_id,
                "received_at": received_at,
                "model_name": response.model.name,
                "model_version": response.model.version,
                "model_semver": response.model.semver,
                "policy_id": policy.policy_id,
                # mode="json" turns the category enums into plain strings for the driver.
                **policy.model_dump(mode="json", include=set(FEATURE_COLUMNS)),
                "predicted_frequency": prediction.predicted_frequency,
            }
        )
    return rows
