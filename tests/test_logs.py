"""Tests for the monitoring log tables, on in-memory SQLite."""

import json
import uuid
from datetime import UTC, datetime, timedelta, timezone

import pytest
from sqlalchemy import insert, inspect, select
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.schema import CreateTable

from claimfreq.contracts import logs
from claimfreq.contracts.api import ModelRef, Prediction, PredictRequest, PredictResponse
from claimfreq.contracts.logs import (
    create_all,
    ground_truth,
    make_engine,
    prediction_log,
    prediction_log_rows,
    rejected_request_log,
)
from claimfreq.contracts.schema import FEATURE_COLUMNS
from claimfreq.fakes import make_predict_request

NOW = datetime(2026, 11, 18, 12, 0, tzinfo=UTC)


@pytest.fixture
def engine():
    """An in-memory SQLite monitoring database with the three tables, foreign keys enforced."""
    engine = make_engine("sqlite://")
    create_all(engine)
    create_all(engine)  # safe to call repeatedly: every service calls it at startup
    return engine


def make_call(n: int = 3) -> tuple[PredictRequest, PredictResponse]:
    """Return a fake ``/predict`` request and a matching response with one prediction each."""
    request = make_predict_request(n)
    response = PredictResponse(
        request_id=str(uuid.uuid4()),
        model=ModelRef(name="claim-frequency", version="1", semver="1.0.0"),
        predictions=[
            Prediction(
                prediction_id=str(uuid.uuid4()), policy_id=p.policy_id, predicted_frequency=0.1
            )
            for p in request.policies
        ],
    )
    return request, response


@pytest.fixture
def logged_rows(engine):
    """Three ``prediction_log`` rows, already inserted into ``engine``."""
    rows = prediction_log_rows(*make_call(), NOW)
    with engine.begin() as connection:
        connection.execute(insert(prediction_log), rows)
    return rows


def test_create_all_survives_a_concurrent_first_start(monkeypatch):
    engine = make_engine("sqlite://")
    real_create_all = logs.metadata.create_all
    calls = []

    def racing_create_all(bind, **kwargs):
        """Lose the race once: another process creates the tables, so our CREATE fails."""
        calls.append(kwargs)
        real_create_all(bind, **kwargs)
        if len(calls) == 1:
            raise OperationalError("CREATE TABLE prediction_log", {}, Exception("already exists"))

    monkeypatch.setattr(logs.metadata, "create_all", racing_create_all)
    create_all(engine)
    assert calls == [{"checkfirst": True}, {"checkfirst": True}]
    assert set(inspect(engine).get_table_names()) == set(logs.metadata.tables)


def test_create_all_retries_once_per_table_then_gives_up(monkeypatch):
    # SQLite commits each CREATE TABLE alone, so a loser can lose once per table.
    calls = []

    def always_losing(bind, **kwargs):
        """Lose every race."""
        calls.append(kwargs)
        raise OperationalError("CREATE TABLE", {}, Exception("already exists"))

    monkeypatch.setattr(logs.metadata, "create_all", always_losing)
    with pytest.raises(OperationalError):
        create_all(make_engine("sqlite://"))
    assert len(calls) == len(logs.metadata.sorted_tables) + 1


def test_feature_columns_keep_schema_names():
    assert [c.name for c in prediction_log.columns if c.name in FEATURE_COLUMNS] == FEATURE_COLUMNS


def test_prediction_log_round_trip(engine, logged_rows):
    with engine.connect() as connection:
        stored = connection.execute(select(prediction_log)).mappings().all()
    assert len(stored) == 3
    first = dict(stored[0])
    # SQLite drops the UTC offset on read; compare the rest exactly.
    assert first.pop("received_at") == NOW.replace(tzinfo=None)
    expected = dict(logged_rows[0])
    expected.pop("received_at")
    assert first == expected


def test_received_at_is_stored_in_utc():
    paris = timezone(timedelta(hours=1))
    rows = prediction_log_rows(*make_call(1), datetime(2026, 11, 18, 13, 0, tzinfo=paris))
    assert rows[0]["received_at"] == NOW
    assert rows[0]["received_at"].tzinfo is UTC


def test_naive_received_at_is_rejected():
    with pytest.raises(ValueError, match="timezone-aware"):
        prediction_log_rows(*make_call(1), NOW.replace(tzinfo=None))


def test_ground_truth_insert_and_foreign_key(engine, logged_rows):
    with engine.begin() as connection:
        connection.execute(
            insert(ground_truth),
            {
                "prediction_id": logged_rows[0]["prediction_id"],
                "ClaimNb": 1,
                "Exposure": 0.5,
                "observed_at": NOW,
            },
        )
        assert connection.execute(select(ground_truth.c.ClaimNb)).scalar_one() == 1
    # make_engine turned SQLite's foreign keys on, so an unknown prediction_id fails.
    with pytest.raises(IntegrityError), engine.begin() as connection:
        connection.execute(
            insert(ground_truth),
            {
                "prediction_id": str(uuid.uuid4()),
                "ClaimNb": 0,
                "Exposure": 1.0,
                "observed_at": NOW,
            },
        )


def test_engine_replaces_a_pooled_connection_that_died(tmp_path):
    """A pooled connection that died, as when the database restarts, is replaced, not reused.

    The API keeps one engine for its whole life, so otherwise its first query after a Postgres
    restart fails, and that request gets HTTP 500.
    """
    engine = make_engine(f"sqlite:///{tmp_path / 'monitoring.db'}")
    create_all(engine)
    with engine.connect() as connection:
        pooled = connection.connection.dbapi_connection
    pooled.close()  # dies while it waits in the pool
    with engine.connect() as connection:
        assert connection.execute(select(prediction_log)).all() == []


@pytest.mark.parametrize(("claim_nb", "exposure"), [(-1, 1.0), (0, 0.0)])
def test_ground_truth_rejects_impossible_outcomes(engine, logged_rows, claim_nb, exposure):
    with pytest.raises(IntegrityError), engine.begin() as connection:
        connection.execute(
            insert(ground_truth),
            {
                "prediction_id": logged_rows[0]["prediction_id"],
                "ClaimNb": claim_nb,
                "Exposure": exposure,
                "observed_at": NOW,
            },
        )


def test_rejected_request_log_round_trip(engine):
    detail = [
        {"loc": ["body", "policies", 0, "DrivAge"], "msg": "too old", "type": "less_than_equal"}
    ]
    row = {
        "request_id": str(uuid.uuid4()),
        "received_at": NOW,
        "batch_id": "out_of_bounds-1",
        "status_code": 422,
        "error_type": "less_than_equal",
        "error_detail": json.dumps(detail),
        "n_policies": 100,
    }
    with engine.begin() as connection:
        connection.execute(insert(rejected_request_log), row)
        stored = connection.execute(select(rejected_request_log)).mappings().one()
    assert json.loads(stored["error_detail"]) == detail
    assert stored["n_policies"] == 100


def test_tables_compile_for_postgresql():
    """The compose stack uses PostgreSQL; mixed-case column names must come out quoted."""
    ddl = str(CreateTable(prediction_log).compile(dialect=postgresql.dialect()))
    assert '"DrivAge" BIGINT NOT NULL' in ddl
    assert "TIMESTAMP WITH TIME ZONE" in ddl
    ddl = str(CreateTable(ground_truth).compile(dialect=postgresql.dialect()))
    assert '"ClaimNb" INTEGER NOT NULL' in ddl
    assert 'CHECK ("ClaimNb" >= 0)' in ddl
