"""Tests for the prediction API, with an injected model and SQLite. No MLflow is needed."""

import json
import math
import threading
import time
from datetime import UTC, datetime

import httpx
import numpy as np
import pandas as pd
import pytest
import uvicorn
from fastapi.testclient import TestClient
from sqlalchemy import func, insert, select

from claimfreq.contracts import env
from claimfreq.contracts.api import (
    PATH_HEALTH,
    PATH_MODEL_INFO,
    PATH_PREDICT,
    ModelInfoResponse,
    PredictResponse,
)
from claimfreq.contracts.logs import (
    ground_truth,
    make_engine,
    prediction_log,
    rejected_request_log,
)
from claimfreq.contracts.schema import FEATURE_COLUMNS
from claimfreq.fakes import TrueFrequencyModel, make_predict_request, true_frequency
from claimfreq.serving import app as app_module
from claimfreq.serving.app import create_app
from claimfreq.serving.champion import build_model_info

TAGS = {
    "semver": "1.2.0",
    "schema_hash": "aab3e74fe93a",
    "data_hash": "900150983cd24fb0d6963f7d28e17f72",
    "git_commit": "82538e9",
    "algorithm": "lightgbm",
}


@pytest.fixture
def db_url(tmp_path, monkeypatch):
    """Point MONITORING_DB_URL at a fresh SQLite file for one test."""
    url = f"sqlite:///{tmp_path / 'monitoring.db'}"
    monkeypatch.setenv(env.MONITORING_DB_URL, url)
    return url


@pytest.fixture
def client(db_url):
    """A test client for an app that serves the fake model."""
    with TestClient(create_app(model=TrueFrequencyModel())) as test_client:
        yield test_client


def count_rows(db_url, table):
    with make_engine(db_url).connect() as connection:
        return connection.execute(select(func.count()).select_from(table)).scalar_one()


def request_body(n=3, batch_id="test-batch"):
    request = make_predict_request(n)
    request.batch_id = batch_id
    return request.model_dump(mode="json")


# --- /predict -----------------------------------------------------------------------------------


def test_predict_returns_predictions_in_input_order(client):
    request = make_predict_request(5)
    response = client.post(PATH_PREDICT, json=request.model_dump(mode="json"))

    assert response.status_code == 200
    body = PredictResponse.model_validate(response.json())
    frame = pd.DataFrame([p.model_dump(mode="json") for p in request.policies])
    expected = true_frequency(frame[FEATURE_COLUMNS])
    assert [p.policy_id for p in body.predictions] == [p.policy_id for p in request.policies]
    np.testing.assert_allclose([p.predicted_frequency for p in body.predictions], expected)


def test_predict_reports_the_dev_identity(client):
    body = client.post(PATH_PREDICT, json=request_body()).json()
    assert body["model"] == {"name": "claim-frequency", "version": "0", "semver": "0.0.0"}


def test_predict_logs_every_policy(client, db_url):
    body = client.post(PATH_PREDICT, json=request_body(n=4)).json()

    with make_engine(db_url).connect() as connection:
        rows = connection.execute(select(prediction_log)).mappings().all()
    assert len(rows) == 4
    assert {row["prediction_id"] for row in rows} == {
        p["prediction_id"] for p in body["predictions"]
    }
    assert {row["request_id"] for row in rows} == {body["request_id"]}
    assert {row["batch_id"] for row in rows} == {"test-batch"}
    assert {row["model_semver"] for row in rows} == {"0.0.0"}


@pytest.fixture
def live_server(db_url):
    """Serve the app with a real uvicorn server in a thread, as the api container does.

    TestClient waits for a response's background tasks before it returns, so it cannot show
    when the log rows are written. A real server sends the 200 first, like in compose.
    """
    config = uvicorn.Config(
        create_app(model=TrueFrequencyModel()), host="127.0.0.1", port=0, log_level="warning"
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(500):
        if server.started:
            break
        time.sleep(0.01)
    assert server.started, "uvicorn did not start"
    # Port 0 lets the OS pick a free port; read back the one it picked.
    port = server.servers[0].sockets[0].getsockname()[1]
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(timeout=5)


def test_ground_truth_can_be_written_right_after_predict(live_server, db_url):
    """The rows are committed before the 200, so the ground_truth foreign key holds.

    This is the contract's logging-order rule, which send-traffic relies on. It writes ground
    truth the moment each 200 arrives, as send-traffic does, several times in a row.
    """
    engine = make_engine(db_url)
    with httpx.Client(base_url=live_server) as http:
        for _ in range(5):
            response = http.post(PATH_PREDICT, json=request_body(n=50))
            assert response.status_code == 200
            rows = [
                {
                    "prediction_id": p["prediction_id"],
                    "ClaimNb": 0,
                    "Exposure": 1.0,
                    "observed_at": datetime.now(UTC),
                }
                for p in response.json()["predictions"]
            ]
            # Raises IntegrityError if a prediction_log row is not committed yet.
            with engine.begin() as connection:
                connection.execute(insert(ground_truth), rows)
    engine.dispose()
    assert count_rows(db_url, ground_truth) == 250


def test_a_model_that_returns_a_bad_value_gives_500_and_logs_nothing(db_url):
    class NegativeModel:
        def predict(self, X):
            return np.full(len(X), -1.0)

    with TestClient(create_app(model=NegativeModel()), raise_server_exceptions=False) as client:
        response = client.post(PATH_PREDICT, json=request_body())
    assert response.status_code == 500
    assert count_rows(db_url, prediction_log) == 0


# --- 422 and rejected_request_log ----------------------------------------------------------------


def test_out_of_bounds_value_is_rejected_and_logged(client, db_url):
    body = request_body(n=2)
    body["policies"][0]["DrivAge"] = 250
    response = client.post(PATH_PREDICT, json=body)

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail[0]["loc"] == ["body", "policies", 0, "DrivAge"]
    assert detail[0]["type"] == "less_than_equal"

    with make_engine(db_url).connect() as connection:
        row = connection.execute(select(rejected_request_log)).mappings().one()
    assert row["status_code"] == 422
    assert row["error_type"] == "less_than_equal"
    assert row["batch_id"] == "test-batch"
    assert row["n_policies"] == 2
    assert json.loads(row["error_detail"]) == detail
    assert count_rows(db_url, prediction_log) == 0


def test_renamed_column_is_rejected_and_logged(client, db_url):
    body = request_body(n=1)
    policy = body["policies"][0]
    policy["DriverAge"] = policy.pop("DrivAge")
    response = client.post(PATH_PREDICT, json=body)

    assert response.status_code == 422
    assert count_rows(db_url, rejected_request_log) == 1


def test_invalid_json_is_rejected_and_logged_without_batch_id(client, db_url):
    response = client.post(
        PATH_PREDICT, content=b"{not json", headers={"Content-Type": "application/json"}
    )

    assert response.status_code == 422
    with make_engine(db_url).connect() as connection:
        row = connection.execute(select(rejected_request_log)).mappings().one()
    assert row["error_type"] == "json_invalid"
    assert row["batch_id"] is None
    assert row["n_policies"] is None


def test_too_long_batch_id_is_not_logged(client, db_url):
    response = client.post(PATH_PREDICT, json=request_body(batch_id="x" * 65))

    assert response.status_code == 422
    with make_engine(db_url).connect() as connection:
        row = connection.execute(select(rejected_request_log)).mappings().one()
    assert row["batch_id"] is None


# --- /health and /model-info ---------------------------------------------------------------------


def test_health_is_ok_with_a_model(client):
    response = client.get(PATH_HEALTH)
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_model_info_reports_the_dev_identity(client):
    info = ModelInfoResponse.model_validate(client.get(PATH_MODEL_INFO).json())
    assert (info.version, info.semver, info.alias) == ("0", "0.0.0", "dev")
    assert {info.run_id, info.schema_hash, info.data_hash, info.git_commit, info.algorithm} == {
        "dev"
    }


def test_model_info_reports_an_injected_model_ref(db_url):
    model_ref = build_model_info("3", TAGS, "run123", 1_700_000_000_000, {"val_gini": 0.3})
    with TestClient(create_app(model=TrueFrequencyModel(), model_ref=model_ref)) as client:
        assert client.get(PATH_MODEL_INFO).json()["semver"] == "1.2.0"
        body = client.post(PATH_PREDICT, json=request_body()).json()
    assert body["model"] == {"name": "claim-frequency", "version": "3", "semver": "1.2.0"}


# --- Before the champion has loaded --------------------------------------------------------------


@pytest.fixture
def no_champion(db_url, monkeypatch):
    """Make every champion load fail. Returns an event that is set after the first try."""
    tried = threading.Event()

    def fail(_tracking_uri):
        tried.set()
        raise RuntimeError("no champion yet")

    monkeypatch.setattr(app_module, "load_champion", fail)
    return tried


def test_without_a_champion_everything_is_503(no_champion, db_url):
    with TestClient(create_app()) as client:
        assert no_champion.wait(timeout=5)
        health = client.get(PATH_HEALTH)
        info = client.get(PATH_MODEL_INFO)
        predict = client.post(PATH_PREDICT, json=request_body())

    assert health.status_code == 503
    assert health.json() == {"status": "degraded", "model_loaded": False}
    assert info.status_code == 503
    assert info.json() == {"detail": "model not loaded"}
    assert predict.status_code == 503
    assert predict.json() == {"detail": "model not loaded"}
    assert count_rows(db_url, prediction_log) == 0


def test_the_app_retries_until_the_champion_loads(db_url, monkeypatch):
    monkeypatch.setattr(app_module, "CHAMPION_RETRY_SECONDS", 0.01)
    model_ref = build_model_info("1", {**TAGS, "semver": "1.0.0"}, "run1", 0, {})
    attempts = []
    loaded = threading.Event()

    def load_on_third_try(_tracking_uri):
        attempts.append(1)
        if len(attempts) < 3:
            raise RuntimeError("no champion yet")
        loaded.set()
        return TrueFrequencyModel(), model_ref

    monkeypatch.setattr(app_module, "load_champion", load_on_third_try)
    with TestClient(create_app()) as client:
        assert loaded.wait(timeout=5)
        # The state is set right after load_champion returns; give the thread a moment.
        for _ in range(100):
            if client.get(PATH_HEALTH).status_code == 200:
                break
            time.sleep(0.01)
        assert client.get(PATH_HEALTH).status_code == 200
        assert client.get(PATH_MODEL_INFO).json()["semver"] == "1.0.0"
    assert len(attempts) == 3


# --- build_model_info ---------------------------------------------------------------------------


def test_build_model_info_reads_tags_and_run():
    info = build_model_info("3", TAGS, "run123", 1_700_000_000_000, {"val_poisson_deviance": 0.31})
    assert info.version == "3"
    assert info.semver == "1.2.0"
    assert info.alias == "champion"
    assert info.trained_at == datetime.fromtimestamp(1_700_000_000, tz=UTC)
    assert info.metrics == {"val_poisson_deviance": 0.31}


def test_build_model_info_drops_non_finite_metrics():
    metrics = {"val_gini": math.nan, "val_poisson_deviance": 0.31, "val_x": math.inf}
    info = build_model_info("3", TAGS, "run123", 0, metrics)
    assert info.metrics == {"val_poisson_deviance": 0.31}


def test_build_model_info_names_missing_tags():
    tags = {key: value for key, value in TAGS.items() if key != "semver"}
    with pytest.raises(ValueError, match="semver"):
        build_model_info("3", tags, "run123", 0, {})
