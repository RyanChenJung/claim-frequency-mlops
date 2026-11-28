"""Tests for ``claimfreq monitor`` on fakes: SQLite logs and ``make_reference_frame``.

There is one test per row of the scenario table in docs/contracts.md (What the monitor must
catch). Each window has 20,000 fake policies: on fewer, the calibration check is dominated by
noise (docs/contracts.md, Develop against fakes).
"""

import json
import uuid
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import insert
from test_publish import FakeWorkspace

from claimfreq import make_drift
from claimfreq.cli import EXIT_CHECK_FAILED, EXIT_OK, main
from claimfreq.contracts import env
from claimfreq.contracts.logs import (
    create_all,
    ground_truth,
    make_engine,
    prediction_log,
    rejected_request_log,
)
from claimfreq.contracts.model import PREDICTED_FREQUENCY
from claimfreq.contracts.params import MonitorParams, Params
from claimfreq.contracts.schema import (
    CLAIM_NB,
    EXPOSURE,
    FEATURE_COLUMNS,
    ID_COLUMN,
    coerce_features,
)
from claimfreq.fakes import TrueFrequencyModel, make_policies, make_reference_frame
from claimfreq.monitoring import publish
from claimfreq.monitoring import reference as reference_module
from claimfreq.monitoring.checks import (
    CALIBRATION,
    DATA_DRIFT,
    LIVE_DEVIANCE,
    PREDICTION_DRIFT,
    REJECTION_RATE,
    Status,
    check_calibration,
    run_checks,
)
from claimfreq.monitoring.window import Window, load_window

N_POLICIES = 20_000
REQUEST_SIZE = 100  # policies per logged request, like traffic.batch_size
T0 = datetime(2026, 12, 1, 12, 0, tzinfo=UTC)
PARAMS = Params()


@pytest.fixture(scope="module")
def reference():
    return make_reference_frame(N_POLICIES, seed=0)


@pytest.fixture(scope="module")
def test_split():
    """Fake policies that play the test split: same distribution as the reference."""
    return make_policies(N_POLICIES, seed=1, id_start=1_000_000)


@pytest.fixture
def engine(tmp_path):
    engine = make_engine(f"sqlite:///{tmp_path / 'monitoring.db'}")
    create_all(engine)
    yield engine
    engine.dispose()


def log_traffic(engine, policies, batch_id, received_at=T0, with_ground_truth=True):
    """Log ``policies`` as the API and send-traffic would: predictions, then ground truth."""
    predicted = TrueFrequencyModel().predict(coerce_features(policies))
    prediction_rows, truth_rows = [], []
    for start in range(0, len(policies), REQUEST_SIZE):
        request_id = str(uuid.uuid4())
        chunk = policies.iloc[start : start + REQUEST_SIZE]
        for (_, policy), value in zip(
            chunk.iterrows(), predicted[start : start + REQUEST_SIZE], strict=True
        ):
            prediction_id = str(uuid.uuid4())
            prediction_rows.append(
                {
                    "prediction_id": prediction_id,
                    "request_id": request_id,
                    "batch_id": batch_id,
                    "received_at": received_at,
                    "model_name": "claim-frequency",
                    "model_version": "0",
                    "model_semver": "0.0.0",
                    "policy_id": int(policy[ID_COLUMN]),
                    **{name: policy[name] for name in FEATURE_COLUMNS},
                    PREDICTED_FREQUENCY: float(value),
                }
            )
            truth_rows.append(
                {
                    "prediction_id": prediction_id,
                    CLAIM_NB: int(policy[CLAIM_NB]),
                    EXPOSURE: float(policy[EXPOSURE]),
                    "observed_at": received_at + timedelta(minutes=1),
                }
            )
    # Numpy ints are not accepted by every driver; cast the integer features.
    for row in prediction_rows:
        for name in FEATURE_COLUMNS:
            if isinstance(row[name], np.integer):
                row[name] = int(row[name])
    with engine.begin() as connection:
        connection.execute(insert(prediction_log), prediction_rows)
        if with_ground_truth:
            connection.execute(insert(ground_truth), truth_rows)


def log_rejections(engine, n_requests, batch_id, received_at=T0):
    """Log ``n_requests`` rejected requests, as the API does for every 422."""
    rows = [
        {
            "request_id": str(uuid.uuid4()),
            "received_at": received_at,
            "batch_id": batch_id,
            "status_code": 422,
            "error_type": "missing",
            "error_detail": json.dumps([{"type": "missing"}]),
            "n_policies": REQUEST_SIZE,
        }
        for _ in range(n_requests)
    ]
    with engine.begin() as connection:
        connection.execute(insert(rejected_request_log), rows)


def failed_checks(engine, reference, batch_id, params=PARAMS.monitor):
    data = load_window(engine, Window(batch_id=batch_id))
    report = run_checks(data, reference, params)
    return {check.name for check in report.failed}, report


# --- One test per scenario row ------------------------------------------------------------------


def test_clean_traffic_fails_nothing(engine, reference, test_split):
    log_traffic(engine, test_split, "clean-1")
    failed, report = failed_checks(engine, reference, "clean-1")
    assert failed == set()
    assert all(check.status is Status.PASSED for check in report.checks)


@pytest.mark.parametrize("scenario", ["schema_break", "out_of_bounds"])
def test_rejected_scenarios_fail_only_the_rejection_rate(engine, reference, scenario):
    # The API rejects every request of these files, so only rejections are logged.
    log_rejections(engine, N_POLICIES // REQUEST_SIZE, scenario)
    failed, report = failed_checks(engine, reference, scenario)
    assert failed == {REJECTION_RATE}
    skipped = {check.name for check in report.checks if check.status is Status.SKIPPED}
    assert skipped == {DATA_DRIFT, PREDICTION_DRIFT, CALIBRATION, LIVE_DEVIANCE}


def test_column_swap_fails_drift(engine, reference, test_split):
    log_traffic(engine, make_drift.column_swap(test_split), "column_swap-1")
    failed, _ = failed_checks(engine, reference, "column_swap-1")
    assert failed & {DATA_DRIFT, PREDICTION_DRIFT}
    # The live calibration ratio may also fail; nothing else may.
    assert failed <= {DATA_DRIFT, PREDICTION_DRIFT, CALIBRATION}


def test_portfolio_shift_fails_data_and_prediction_drift(engine, reference, test_split):
    shifted = make_drift.portfolio_shift(test_split, PARAMS, np.random.default_rng(0))
    log_traffic(engine, shifted, "portfolio_shift-1")
    failed, _ = failed_checks(engine, reference, "portfolio_shift-1")
    assert failed == {DATA_DRIFT, PREDICTION_DRIFT}


def test_concept_drift_fails_calibration_and_live_deviance(engine, reference, test_split):
    drifted = make_drift.concept_drift(test_split, PARAMS, np.random.default_rng(0))
    log_traffic(engine, drifted, "concept_drift-1")
    failed, _ = failed_checks(engine, reference, "concept_drift-1")
    assert failed == {CALIBRATION, LIVE_DEVIANCE}


# --- Skip rules ---------------------------------------------------------------------------------


def test_too_few_predictions_skips_the_drift_checks(engine, reference, test_split):
    log_traffic(engine, test_split.head(150), "small")
    _, report = failed_checks(engine, reference, "small")
    statuses = {check.name: check.status for check in report.checks}
    assert statuses[DATA_DRIFT] is Status.SKIPPED
    assert statuses[PREDICTION_DRIFT] is Status.SKIPPED
    assert report.snapshot is None


def test_no_ground_truth_skips_calibration_and_deviance(engine, reference, test_split):
    log_traffic(engine, test_split, "no-truth", with_ground_truth=False)
    _, report = failed_checks(engine, reference, "no-truth")
    statuses = {check.name: check.status for check in report.checks}
    assert statuses[CALIBRATION] is Status.SKIPPED
    assert statuses[LIVE_DEVIANCE] is Status.SKIPPED
    assert "no ground truth yet" in next(c for c in report.checks if c.name == CALIBRATION).detail


def test_prediction_drift_can_be_reported_without_failing(engine, reference, test_split):
    shifted = make_drift.portfolio_shift(test_split, PARAMS, np.random.default_rng(0))
    log_traffic(engine, shifted, "portfolio_shift-1")
    params = MonitorParams(fail_on_prediction_drift=False)
    failed, report = failed_checks(engine, reference, "portfolio_shift-1", params)
    assert failed == {DATA_DRIFT}
    detail = next(c for c in report.checks if c.name == PREDICTION_DRIFT).detail
    assert "ignored" in detail


def test_no_observed_claims_fails_calibration():
    labelled = pd.DataFrame({CLAIM_NB: [0.0, 0.0], EXPOSURE: [1.0, 0.5], PREDICTED_FREQUENCY: 0.1})
    assert check_calibration(labelled, PARAMS.monitor).status is Status.FAILED


# --- The window ---------------------------------------------------------------------------------


def test_window_filters_by_time_and_batch(engine, test_split):
    log_traffic(engine, test_split.head(100), "early", received_at=T0)
    log_traffic(engine, test_split.iloc[100:300], "late", received_at=T0 + timedelta(hours=1))
    log_rejections(engine, 3, "late", received_at=T0 + timedelta(hours=1))

    def sizes(window):
        data = load_window(engine, window)
        return len(data.predictions), data.n_rejected

    assert sizes(Window()) == (300, 3)
    # since is inclusive, until exclusive.
    assert sizes(Window(since=T0 + timedelta(hours=1))) == (200, 3)
    assert sizes(Window(until=T0 + timedelta(hours=1))) == (100, 0)
    assert sizes(Window(batch_id="early")) == (100, 0)
    # A naive time is read as UTC.
    assert sizes(Window(since=(T0 + timedelta(hours=1)).replace(tzinfo=None))) == (200, 3)


def test_window_joins_ground_truth_and_counts_requests(engine, test_split):
    log_traffic(engine, test_split.head(250), "b")
    data = load_window(engine, Window(batch_id="b"))
    assert data.n_accepted == 3  # 100 + 100 + 50 policies
    assert data.n_requests == 3
    assert len(data.with_ground_truth) == 250
    assert list(data.predictions[FEATURE_COLUMNS].dtypes) == list(
        coerce_features(test_split)[FEATURE_COLUMNS].dtypes
    )


def test_empty_window(engine):
    data = load_window(engine, Window(batch_id="nothing"))
    assert data.is_empty
    with pytest.raises(ValueError):
        run_checks(data, pd.DataFrame(), PARAMS.monitor)


# --- The CLI ------------------------------------------------------------------------------------


@pytest.fixture
def cli_db(tmp_path, monkeypatch, reference):
    """A SQLite MONITORING_DB_URL for the CLI, and a fake champion reference."""
    url = f"sqlite:///{tmp_path / 'cli.db'}"
    monkeypatch.setenv(env.MONITORING_DB_URL, url)
    calls = []

    def fake_load_reference(tracking_uri):
        calls.append(tracking_uri)
        return reference, "7"

    monkeypatch.setattr(reference_module, "load_reference", fake_load_reference)
    # No Evidently UI in unit tests: tests/test_publish.py covers the push.
    monkeypatch.setattr(publish, "_workspace", lambda url: FakeWorkspace())
    monkeypatch.delenv(env.ALERT_WEBHOOK_URL, raising=False)
    engine = make_engine(url)
    create_all(engine)
    yield engine, calls
    engine.dispose()


def test_cli_empty_window_exits_0_without_a_champion(cli_db, capsys):
    _, calls = cli_db
    assert main(["monitor", "--batch-id", "nothing"]) == EXIT_OK
    assert "no data in window" in capsys.readouterr().out
    assert calls == []


def test_cli_clean_traffic_exits_0(cli_db, test_split, capsys):
    engine, _ = cli_db
    log_traffic(engine, test_split, "clean-1")
    assert main(["monitor", "--batch-id", "clean-1"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "champion version 7" in out
    assert "monitor: calibration: passed" in out


def test_cli_failed_check_exits_10(cli_db, capsys):
    engine, _ = cli_db
    log_rejections(engine, 10, "schema_break-1")
    assert main(["monitor", "--batch-id", "schema_break-1"]) == EXIT_CHECK_FAILED
    captured = capsys.readouterr()
    assert "monitor: rejection_rate: failed" in captured.out
    assert "failed checks: rejection_rate" in captured.err
