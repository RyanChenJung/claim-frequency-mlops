"""Tests for ``claimfreq send-traffic`` against a fake API that behaves like Role D's.

``FakeAPI`` is an ``httpx.MockTransport`` handler. It answers ``GET /health``, validates each
``POST /predict`` body with the strict ``PredictRequest`` (so a float or string number gets a
422, as from the real API), and for every 200 it logs ``prediction_log_rows`` to the monitoring
database, as the real API does. That matters: ``ground_truth`` has a foreign key to
``prediction_log``, which ``make_engine`` enforces on SQLite too. The database is a file in
``tmp_path``, so the fake and ``send-traffic`` (each with its own engine) see the same rows.
Scripted statuses and exceptions exercise the retry and failure paths; ``sleep`` is a recorder,
so nothing really waits.
"""

import json
import re
import uuid
from datetime import UTC, datetime

import httpx
import pandas as pd
import pytest
from pydantic import ValidationError
from sqlalchemy import insert

from claimfreq import send_traffic as send_traffic_module
from claimfreq.cli import EXIT_ERROR, EXIT_OK, main
from claimfreq.contracts import logs, paths, schema
from claimfreq.contracts.api import (
    MAX_BATCH_ID_LENGTH,
    PATH_HEALTH,
    PATH_PREDICT,
    ModelRef,
    Prediction,
    PredictRequest,
    PredictResponse,
)
from claimfreq.contracts.params import Params, TrafficParams
from claimfreq.contracts.paths import DriftScenario
from claimfreq.fakes import make_policies
from claimfreq.make_drift import out_of_bounds, schema_break
from claimfreq.send_traffic import (
    HEALTH_INTERVAL_S,
    HEALTH_POLLS,
    RETRY_DELAYS_S,
    TrafficError,
    default_batch_id,
    make_body,
    send_traffic,
)

N_ROWS = 250
BATCH_SIZE = 100  # so 250 rows make batches of 100, 100, and 50
PARAMS = Params(traffic=TrafficParams(batch_size=BATCH_SIZE))
ID = schema.ID_COLUMN


class FakeAPI:
    """A stand-in for Role D's API, used as an ``httpx.MockTransport`` handler.

    Attributes:
        log: Log ``prediction_log`` rows for each 200, as the real API does.
        health: Statuses (or exceptions to raise) for the next ``/health`` calls.
        health_default: The ``/health`` status once ``health`` is used up.
        predict: Statuses (or exceptions to raise) for the next ``/predict`` calls; after
            that, real validation and scoring.
        reverse_ids: Return the predictions in reverse order, breaking the contract.
        calls: The path of every request received, in order.
    """

    def __init__(self, db_url: str, log: bool = True) -> None:
        self.engine = logs.make_engine(db_url)
        logs.create_all(self.engine)
        self.log = log
        self.health: list[int | Exception] = []
        self.health_default = 200
        self.predict: list[int | Exception] = []
        self.reverse_ids = False
        self.calls: list[str] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        self.calls.append(path)
        if path == PATH_HEALTH:
            return self._scripted(self.health, self.health_default)
        if path == PATH_PREDICT:
            return self._scripted(self.predict, 0) if self.predict else self._score(request)
        return httpx.Response(404, json={"detail": "Not Found"})

    @staticmethod
    def _scripted(script: list[int | Exception], default: int) -> httpx.Response:
        action = script.pop(0) if script else default
        if isinstance(action, Exception):
            raise action
        return httpx.Response(action, json={"detail": f"scripted {action}"})

    def _score(self, request: httpx.Request) -> httpx.Response:
        try:
            body = PredictRequest.model_validate_json(request.content)
        except ValidationError as error:
            return httpx.Response(422, json={"detail": json.loads(error.json())})
        response = PredictResponse(
            request_id=str(uuid.uuid4()),
            model=ModelRef(name="claim-frequency", version="0", semver="0.0.0"),
            predictions=[
                Prediction(
                    prediction_id=str(uuid.uuid4()),
                    policy_id=policy.policy_id,
                    predicted_frequency=0.1,
                )
                for policy in body.policies
            ],
        )
        if self.log:
            # Committed before the 200 is returned, as #17 is asked to guarantee.
            with self.engine.begin() as connection:
                rows = logs.prediction_log_rows(body, response, datetime.now(UTC))
                connection.execute(insert(logs.prediction_log), rows)
        if self.reverse_ids:
            response.predictions.reverse()
        return httpx.Response(200, content=response.model_dump_json())

    def table(self, name: str) -> pd.DataFrame:
        """Return one monitoring table as a frame."""
        with self.engine.connect() as connection:
            return pd.read_sql_table(name, connection)


class Sleeps(list):
    """A ``sleep`` that records each wait instead of waiting."""

    def __call__(self, seconds: float) -> None:
        self.append(seconds)


@pytest.fixture
def db_url(tmp_path, monkeypatch) -> str:
    """Point DATA_DIR and MONITORING_DB_URL into ``tmp_path``; return the database URL."""
    url = f"sqlite:///{tmp_path / 'monitoring.db'}"
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MONITORING_DB_URL", url)
    return url


@pytest.fixture
def holdout(db_url) -> pd.DataFrame:
    """Write a shuffled fake test split and its two broken drift files; return the split."""
    policies = make_policies(N_ROWS, seed=3, id_start=900_000)
    policies = policies.sample(frac=1, random_state=0).reset_index(drop=True)
    paths.data_path(paths.TEST).parent.mkdir(parents=True)
    policies.to_parquet(paths.data_path(paths.TEST), index=False)
    paths.data_path(paths.DRIFT_DIR).mkdir(parents=True)
    schema_break(policies).to_parquet(paths.drift_path(DriftScenario.SCHEMA_BREAK), index=False)
    out_of_bounds(policies).to_parquet(paths.drift_path(DriftScenario.OUT_OF_BOUNDS), index=False)
    return policies


@pytest.fixture
def api(db_url) -> FakeAPI:
    return FakeAPI(db_url)


@pytest.fixture
def client(api):
    with httpx.Client(transport=httpx.MockTransport(api), base_url="http://api.test") as c:
        yield c


@pytest.fixture
def sleeps() -> Sleeps:
    return Sleeps()


def run(source: str, client: httpx.Client, sleeps: Sleeps):
    """Run send_traffic with the test batch size, a fixed batch id, and no real waiting."""
    return send_traffic(source, batch_id="run-1", params=PARAMS, client=client, sleep=sleeps)


def test_default_batch_id_is_source_and_utc_time():
    moment = datetime(2026, 12, 1, 15, 30, 0, tzinfo=UTC)
    assert default_batch_id("portfolio_shift", moment) == "portfolio_shift-20261201T153000Z"
    longest = max(["test", *DriftScenario], key=len)
    assert len(default_batch_id(longest)) <= MAX_BATCH_ID_LENGTH


def test_body_posts_every_column_but_the_three_as_native_values(holdout):
    body = make_body(holdout.head(3), "run-1")
    assert body["batch_id"] == "run-1"
    first = body["policies"][0]
    assert list(first) == ["policy_id", *schema.FEATURE_COLUMNS]
    assert first["policy_id"] == holdout[ID].iloc[0]
    # Native Python values, so the strict API fields see exactly what the file holds.
    assert {type(v) for p in body["policies"] for v in p.values()} <= {int, str}
    # As is: a broken file's columns are posted unchanged.
    broken = make_body(schema_break(holdout).head(1), "run-1")["policies"][0]
    assert "DriverAge" in broken
    assert "DrivAge" not in broken


def test_scores_every_batch_and_writes_its_ground_truth(holdout, api, client, sleeps):
    counts = run("test", client, sleeps)

    assert (counts.batches, counts.scored_batches, counts.rejected_batches) == (3, 3, 0)
    assert counts.scored_policies == counts.ground_truth_rows == N_ROWS
    assert sleeps == []
    log = api.table("prediction_log")
    assert set(log["batch_id"]) == {"run-1"}
    # Each outcome belongs to the policy its prediction was made for.
    joined = api.table("ground_truth").merge(log[["prediction_id", "policy_id"]])
    assert len(joined) == N_ROWS
    expected = holdout.set_index(ID).loc[joined["policy_id"]]
    assert joined[schema.CLAIM_NB].tolist() == expected[schema.CLAIM_NB].tolist()
    assert joined[schema.EXPOSURE].tolist() == expected[schema.EXPOSURE].tolist()


@pytest.mark.parametrize("source", [DriftScenario.SCHEMA_BREAK, DriftScenario.OUT_OF_BOUNDS])
def test_a_broken_file_is_rejected_and_the_run_succeeds(holdout, api, client, sleeps, source):
    counts = run(source, client, sleeps)

    assert (counts.batches, counts.scored_batches, counts.rejected_batches) == (3, 0, 3)
    assert counts.rejected_policies == N_ROWS
    assert counts.ground_truth_rows == 0
    assert api.table("ground_truth").empty


def test_values_are_posted_without_coercion(holdout, client, sleeps):
    # A whole-number float is valid JSON but breaks the strict integer field, so it is rejected.
    path = paths.data_path(paths.TEST)
    holdout.assign(DrivAge=holdout["DrivAge"].astype(float)).to_parquet(path, index=False)

    counts = run("test", client, sleeps)

    assert counts.rejected_batches == 3
    assert counts.ground_truth_rows == 0


def test_waits_for_health_and_counts_connection_errors_as_not_ready(holdout, api, client, sleeps):
    api.health = [503, httpx.ConnectError("refused"), 503]

    counts = run("test", client, sleeps)

    assert sleeps == [HEALTH_INTERVAL_S] * 3
    assert api.calls[:5] == [PATH_HEALTH] * 4 + [PATH_PREDICT]
    assert counts.scored_batches == 3


def test_gives_up_when_health_never_returns_200(holdout, api, client, sleeps):
    api.health_default = 503

    with pytest.raises(TrafficError, match="API not ready"):
        run("test", client, sleeps)

    assert sleeps == [HEALTH_INTERVAL_S] * (HEALTH_POLLS - 1)
    assert api.calls == [PATH_HEALTH] * HEALTH_POLLS


def test_retries_connection_errors_timeouts_and_5xx(holdout, api, client, sleeps):
    api.predict = [503, httpx.ConnectError("refused"), httpx.ReadTimeout("slow")]

    counts = run("test", client, sleeps)

    assert sleeps == list(RETRY_DELAYS_S)
    assert counts.scored_batches == 3
    assert counts.ground_truth_rows == N_ROWS


def test_gives_up_after_the_retries(holdout, api, client, sleeps):
    api.predict = [500] * (len(RETRY_DELAYS_S) + 1)

    with pytest.raises(TrafficError, match=f"after {len(RETRY_DELAYS_S)} retries: HTTP 500"):
        run("test", client, sleeps)

    assert sleeps == list(RETRY_DELAYS_S)
    assert api.table("ground_truth").empty


def test_another_status_stops_the_run_without_retrying(holdout, api, client, sleeps):
    api.predict = [404]

    with pytest.raises(TrafficError, match="HTTP 404, expected 200 or 422"):
        run("test", client, sleeps)

    assert api.calls.count(PATH_PREDICT) == 1
    assert sleeps == []


def test_a_response_for_other_policies_stops_the_run(holdout, api, client, sleeps):
    api.reverse_ids = True

    with pytest.raises(TrafficError, match="does not match the batch sent"):
        run("test", client, sleeps)

    assert api.table("ground_truth").empty


def test_missing_prediction_log_rows_fail_with_a_hint(holdout, api, client, sleeps):
    # An API that does not log (or logs to another database): the foreign key rejects the write.
    api.log = False

    with pytest.raises(TrafficError) as error:
        run("test", client, sleeps)

    message = str(error.value)
    assert "FOREIGN KEY constraint failed" in message  # the database's own error
    assert "MONITORING_DB_URL" in message
    assert api.table("ground_truth").empty


def test_makes_and_closes_its_own_client_for_api_url(holdout, api, monkeypatch, sleeps):
    clients = []

    def make_client(base_url):
        clients.append(httpx.Client(transport=httpx.MockTransport(api), base_url=base_url))
        return clients[-1]

    monkeypatch.setattr(send_traffic_module, "make_client", make_client)
    monkeypatch.setenv("API_URL", "http://api.test:8000")

    send_traffic("test", params=PARAMS, sleep=sleeps)

    assert str(clients[0].base_url) == "http://api.test:8000"
    assert clients[0].is_closed


@pytest.fixture
def cli_api(holdout, api, monkeypatch) -> FakeAPI:
    """Route the CLI's own client to the fake API; return the fake."""
    transport = httpx.MockTransport(api)
    monkeypatch.setattr(
        send_traffic_module,
        "make_client",
        lambda base_url: httpx.Client(transport=transport, base_url=base_url),
    )
    return api


def test_cli_send_traffic_prints_the_counts_and_exits_0(cli_api, capsys):
    # params.yaml's traffic.batch_size is 100, so the 250 fake policies make 3 batches.
    assert main(["send-traffic", "--source", "test", "--batch-id", "cli-1"]) == EXIT_OK

    out = capsys.readouterr().out
    assert f"send-traffic: {N_ROWS} policies from " in out
    assert "batch_id=cli-1" in out
    assert (
        "send-traffic: 3 batches (3 scored with HTTP 200, 0 rejected with HTTP 422); "
        f"{N_ROWS} policies scored, 0 rejected; {N_ROWS} ground_truth rows"
    ) in out
    assert len(cli_api.table("ground_truth")) == N_ROWS


def test_cli_send_traffic_defaults_the_batch_id_and_exits_0_on_422(cli_api, capsys):
    assert main(["send-traffic", "--source", "schema_break"]) == EXIT_OK

    out = capsys.readouterr().out
    assert re.search(r"batch_id=schema_break-\d{8}T\d{6}Z,", out)
    assert "3 batches (0 scored with HTTP 200, 3 rejected with HTTP 422)" in out


def test_cli_send_traffic_exits_1_with_the_message_and_no_traceback(cli_api, capsys):
    cli_api.log = False

    assert main(["send-traffic", "--source", "test"]) == EXIT_ERROR

    err = capsys.readouterr().err
    assert err.startswith("send-traffic: error: ground_truth write failed")
    assert "MONITORING_DB_URL" in err
    assert "Traceback" not in err
