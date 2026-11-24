"""Role A (Data): the ``claimfreq send-traffic`` command.

``send-traffic`` replays one data source through the prediction API, as live traffic, and then
simulates the outcomes that arrive later (docs/contracts.md, CLI and Drift files):

1. It reads ``splits/test.parquet`` (``--source test``) or ``drift/<scenario>.parquet`` under
   ``DATA_DIR``, and waits until ``GET /health`` returns 200.
2. It posts the rows in file order, in batches of ``traffic.batch_size``, to ``POST /predict``.
   Each body has one ``batch_id`` for the whole run and every column except ``IDpol``,
   ``ClaimNb``, and ``Exposure``, as is: no coercion, so the broken files reach the API broken.
   ``IDpol`` is sent as ``policy_id``.
3. After each batch scored with HTTP 200, it writes that batch's ``ClaimNb`` and ``Exposure`` to
   ``ground_truth`` (``MONITORING_DB_URL``), keyed by the returned ``prediction_id``s. A batch
   rejected with 422 is counted and has no ground truth.

It succeeds when every batch got 200 or 422. Anything else raises ``TrafficError``, which the
CLI turns into exit code 1:

- a ``batch_id`` that is already in the monitoring logs, because ``monitor --batch-id`` would
  then mix two runs (a failed run re-sent under the same id would count its batches twice);

- a connection error, a timeout, or a 5xx (503 included) that persists after
  ``RETRY_DELAYS_S``;
- any other status (404, 413, ...), which is not retried;
- a 200 body that breaks the API contract, or whose ``policy_id``s differ from those sent;
- a ``ground_truth`` write that the database rejects, most likely because the API's
  ``prediction_log`` rows are not in this database.

A retried batch can be logged twice by the API: if the first attempt was logged but its response
was lost (a timeout or a 5xx after logging), the second attempt logs the batch again with new
``prediction_id``s. Only the rows of the answered attempt get ground truth.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
from pydantic import ValidationError
from sqlalchemy import Engine, func, insert, select
from sqlalchemy.exc import IntegrityError

from claimfreq.contracts import env, logs, paths, schema
from claimfreq.contracts.api import PATH_HEALTH, PATH_PREDICT, PredictResponse
from claimfreq.contracts.params import Params, load_params
from claimfreq.contracts.paths import DriftScenario

SOURCE_TEST = "test"
# Columns that are never posted: the id travels as policy_id, the targets go to ground_truth.
NOT_POSTED = [schema.ID_COLUMN, schema.CLAIM_NB, schema.EXPOSURE]

# Readiness wait: up to HEALTH_POLLS polls, HEALTH_INTERVAL_S apart (about 2 minutes). The API
# retries loading @champion every 30 s, so this allows several attempts. Polls are counted
# rather than timed, so a test with a no-op sleep finishes at once.
HEALTH_POLLS = 60
HEALTH_INTERVAL_S = 2.0
# Short for /health, so a hung API cannot stretch the wait far beyond 2 minutes; longer for
# /predict, which scores and logs a whole batch (httpx's own default is 5 s).
HEALTH_TIMEOUT = httpx.Timeout(5.0)
PREDICT_TIMEOUT = httpx.Timeout(30.0)
# Waits before each retry of a batch after a connection error, a timeout, or a 5xx.
RETRY_DELAYS_S = (1.0, 2.0, 4.0)

Sleep = Callable[[float], None]


class TrafficError(RuntimeError):
    """Raised when ``send-traffic`` cannot finish the run; the CLI exits 1 with the message."""


@dataclass
class TrafficCounts:
    """What one run sent and wrote, printed by the CLI."""

    batches: int = 0
    scored_batches: int = 0  # HTTP 200
    rejected_batches: int = 0  # HTTP 422
    scored_policies: int = 0
    rejected_policies: int = 0
    ground_truth_rows: int = 0


def source_path(source: str) -> Path:
    """Return the data file of ``source``: ``test`` or a drift scenario name."""
    if source == SOURCE_TEST:
        return paths.data_path(paths.TEST)
    return paths.drift_path(DriftScenario(source))


def default_batch_id(source: str, now: datetime | None = None) -> str:
    """Return ``<source>-<UTC time %Y%m%dT%H%M%SZ>``, e.g. ``portfolio_shift-20261201T153000Z``."""
    moment = (now or datetime.now(UTC)).astimezone(UTC)
    return f"{source}-{moment:%Y%m%dT%H%M%SZ}"


def make_client(base_url: str) -> httpx.Client:
    """Return the HTTP client for ``API_URL``. Tests replace this to inject a fake API."""
    return httpx.Client(base_url=base_url)


def make_body(batch: pd.DataFrame, batch_id: str) -> dict[str, Any]:
    """Return the ``POST /predict`` body for one batch of rows, with the values as is.

    Values become native Python ints, floats, and strs, and a missing value (NaN or None)
    becomes JSON ``null``: httpx refuses to encode NaN, and ``null`` lets the API reject the
    batch with a 422. The API's strict fields then accept or reject every value exactly as it
    is in the file.
    """
    ids = batch[schema.ID_COLUMN].tolist()
    frame = batch.drop(columns=NOT_POSTED).astype(object)
    features = frame.where(frame.notna(), None).to_dict("records")
    policies = [{"policy_id": i, **f} for i, f in zip(ids, features, strict=True)]
    return {"batch_id": batch_id, "policies": policies}


def batches(df: pd.DataFrame, size: int) -> Iterator[pd.DataFrame]:
    """Yield consecutive slices of ``df`` with ``size`` rows; the last one may be shorter."""
    for start in range(0, len(df), size):
        yield df.iloc[start : start + size]


def wait_for_health(client: httpx.Client, sleep: Sleep = time.sleep) -> None:
    """Poll ``GET /health`` until it returns 200; raise ``TrafficError`` after ``HEALTH_POLLS``.

    A connection error or a timeout counts as "not ready yet", like a 503.
    """
    for poll in range(HEALTH_POLLS):
        try:
            if client.get(PATH_HEALTH, timeout=HEALTH_TIMEOUT).status_code == 200:
                return
        except httpx.TransportError:
            pass
        if poll < HEALTH_POLLS - 1:
            sleep(HEALTH_INTERVAL_S)
    waited = (HEALTH_POLLS - 1) * HEALTH_INTERVAL_S
    raise TrafficError(
        f"API not ready at {client.base_url} after {HEALTH_POLLS} polls of {PATH_HEALTH} "
        f"(about {waited:.0f} s); is the API running, and is there a @champion?"
    )


def post_batch(
    client: httpx.Client, body: dict[str, Any], sleep: Sleep = time.sleep
) -> httpx.Response:
    """Post one batch, retrying connection errors, timeouts, and 5xx after ``RETRY_DELAYS_S``.

    Returns the first response below 500. Raises ``TrafficError`` when the retries run out.
    """
    for delay in (*RETRY_DELAYS_S, None):
        try:
            response = client.post(PATH_PREDICT, json=body, timeout=PREDICT_TIMEOUT)
        except httpx.TransportError as error:
            # Every httpx timeout and connection error is a TransportError.
            problem = f"{type(error).__name__}: {error}"
        else:
            if response.status_code < 500:
                return response
            problem = f"HTTP {response.status_code}"
        if delay is None:
            raise TrafficError(
                f"POST {PATH_PREDICT} failed after {len(RETRY_DELAYS_S)} retries: {problem}"
            )
        sleep(delay)
    raise AssertionError("unreachable")  # the loop always returns or raises


def parse_scored(response: httpx.Response, policy_ids: list[int]) -> PredictResponse:
    """Parse a 200 body and check it answers the batch sent, in order; else ``TrafficError``."""
    try:
        scored = PredictResponse.model_validate_json(response.content)
    except ValidationError as error:
        raise TrafficError(f"HTTP 200 body breaks the API contract: {error}") from error
    returned = [prediction.policy_id for prediction in scored.predictions]
    if returned != policy_ids:
        raise TrafficError(
            f"HTTP 200 body does not match the batch sent: {len(returned)} predictions for "
            f"{len(policy_ids)} policies, or policy_ids out of order"
        )
    return scored


def check_batch_id_unused(engine: Engine, batch_id: str) -> None:
    """Raise ``TrafficError`` if ``batch_id`` already has rows in the monitoring logs.

    Both tables count: a run of ``schema_break`` or ``out_of_bounds`` leaves rows only in
    ``rejected_request_log``.
    """
    with engine.connect() as connection:
        used = {
            table.name: connection.execute(
                select(func.count()).select_from(table).where(table.c.batch_id == batch_id)
            ).scalar_one()
            for table in (logs.prediction_log, logs.rejected_request_log)
        }
    if any(used.values()):
        rows = ", ".join(f"{n} in {name}" for name, n in used.items())
        raise TrafficError(
            f"batch_id={batch_id} is already used ({rows}), so `monitor --batch-id` would mix "
            "two runs. Use a new --batch-id, or omit it for <source>-<UTC time>."
        )


def write_ground_truth(engine: Engine, scored: PredictResponse, batch: pd.DataFrame) -> int:
    """Write one scored batch's ``ClaimNb`` and ``Exposure`` to ``ground_truth`` atomically.

    Rows pair with predictions by position, which ``parse_scored`` has checked. Returns the row
    count. Raises ``TrafficError`` when the database rejects the rows (``IntegrityError``).
    """
    observed_at = datetime.now(UTC)
    outcomes = zip(batch[schema.CLAIM_NB].tolist(), batch[schema.EXPOSURE].tolist(), strict=True)
    rows = [
        {
            "prediction_id": prediction.prediction_id,
            schema.CLAIM_NB: claim_nb,
            schema.EXPOSURE: exposure,
            "observed_at": observed_at,
        }
        for prediction, (claim_nb, exposure) in zip(scored.predictions, outcomes, strict=True)
    ]
    try:
        with engine.begin() as connection:
            connection.execute(insert(logs.ground_truth), rows)
    except IntegrityError as error:
        # Not retried: a missing prediction_log row will not appear by waiting a moment, and a
        # check-constraint or duplicate-key failure will not either.
        raise TrafficError(
            f"ground_truth write failed ({error.orig}). Most likely the API's prediction_log "
            "rows are not in this database: is MONITORING_DB_URL the same one the API logs to, "
            f"and does the API commit before returning 200? ({engine.url.render_as_string()})"
        ) from error
    return len(rows)


def _send_batch(
    client: httpx.Client,
    engine: Engine,
    batch: pd.DataFrame,
    batch_id: str,
    counts: TrafficCounts,
    sleep: Sleep,
) -> None:
    """Post one batch, write its ground truth if it was scored, and update ``counts``.

    ``counts`` changes only once the batch is fully handled, so after a failure it describes
    the batches that were completed.
    """
    body = make_body(batch, batch_id)
    response = post_batch(client, body, sleep)
    if response.status_code == 200:
        scored = parse_scored(response, [p["policy_id"] for p in body["policies"]])
        written = write_ground_truth(engine, scored, batch)
        counts.scored_batches += 1
        counts.scored_policies += len(batch)
        counts.ground_truth_rows += written
    elif response.status_code == 422:
        counts.rejected_batches += 1
        counts.rejected_policies += len(batch)
    else:
        raise TrafficError(
            f"POST {PATH_PREDICT} returned HTTP {response.status_code}, expected 200 or "
            f"422: {response.text[:200]}"
        )
    counts.batches += 1


def send_traffic(
    source: str,
    batch_id: str | None = None,
    params: Params | None = None,
    client: httpx.Client | None = None,
    sleep: Sleep = time.sleep,
) -> TrafficCounts:
    """Replay ``source`` through the API and write ground truth for every scored batch.

    Args:
        source: ``test`` or a drift scenario name.
        batch_id: The id of this run; ``default_batch_id(source)`` when omitted.
        params: Loaded ``params.yaml``; read from the repo root when omitted.
        client: An HTTP client whose ``base_url`` is the API (FastAPI's ``TestClient`` works);
            ``make_client(API_URL)`` when omitted, closed at the end.
        sleep: Called with each wait in seconds; tests pass a recorder so nothing really waits.

    Returns:
        The counts of the run.

    Raises:
        TrafficError: on any failure the module docstring lists; the CLI exits 1.
    """
    params = params or load_params()
    path = source_path(source)
    df = pd.read_parquet(path)
    batch_id = batch_id or default_batch_id(source)
    size = params.traffic.batch_size

    own_client = client is None
    if client is None:
        client = make_client(env.get(env.API_URL) or "")
    engine = logs.make_engine(env.get(env.MONITORING_DB_URL) or "")
    counts = TrafficCounts()
    try:
        logs.create_all(engine)
        check_batch_id_unused(engine, batch_id)
        print(
            f"send-traffic: {len(df)} policies from {path} in batches of {size}, "
            f"batch_id={batch_id}, API {client.base_url}"
        )
        wait_for_health(client, sleep)
        n_batches = math.ceil(len(df) / size)
        try:
            for batch in batches(df, size):
                _send_batch(client, engine, batch, batch_id, counts, sleep)
        except TrafficError as error:
            # Say how far the run got, so a half-finished run is not re-sent under the same id.
            raise TrafficError(
                f"{error} (after {counts.batches} of {n_batches} batches; "
                f"{counts.ground_truth_rows} ground_truth rows already written under "
                f"batch_id={batch_id}; re-run with a new --batch-id)"
            ) from error
    finally:
        engine.dispose()
        if own_client:
            client.close()
    return counts
