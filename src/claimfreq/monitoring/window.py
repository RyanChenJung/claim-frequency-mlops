"""Read one time window of the monitoring logs.

The window applies to ``received_at`` in ``prediction_log`` and ``rejected_request_log``.
``since`` is inclusive and ``until`` exclusive; either can be left open. ``batch_id`` keeps only
the requests of one ``send-traffic`` run. Ground truth is joined by ``prediction_id``, whatever
its ``observed_at`` (docs/contracts.md, DAGs).
"""

from dataclasses import dataclass
from datetime import UTC, datetime

import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.engine import Engine

from claimfreq.contracts.logs import ground_truth, prediction_log, rejected_request_log
from claimfreq.contracts.model import PREDICTED_FREQUENCY
from claimfreq.contracts.schema import CLAIM_NB, EXPOSURE, FEATURE_COLUMNS, coerce_features

REQUEST_ID = "request_id"
# The columns of WindowData.predictions, in this order.
PREDICTION_COLUMNS = [REQUEST_ID, *FEATURE_COLUMNS, PREDICTED_FREQUENCY, CLAIM_NB, EXPOSURE]


def _to_utc(value: datetime | None) -> datetime | None:
    """Return ``value`` in UTC. A naive time is read as UTC."""
    if value is None:
        return None
    if value.utcoffset() is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


@dataclass(frozen=True)
class Window:
    """Which logged requests ``monitor`` looks at."""

    since: datetime | None = None
    until: datetime | None = None
    batch_id: str | None = None

    def __post_init__(self) -> None:
        # frozen=True, so set the UTC values through object.__setattr__.
        object.__setattr__(self, "since", _to_utc(self.since))
        object.__setattr__(self, "until", _to_utc(self.until))

    def describe(self) -> str:
        """A short text for log lines, e.g. ``since=... until=... batch_id=...``."""
        since = self.since.isoformat() if self.since else "unbounded"
        until = self.until.isoformat() if self.until else "unbounded"
        return f"since={since} until={until} batch_id={self.batch_id or 'any'}"

    def filters(self, table) -> list:
        """The SQL conditions for ``table``, which has ``received_at`` and ``batch_id``."""
        conditions = []
        if self.since is not None:
            conditions.append(table.c.received_at >= self.since)
        if self.until is not None:
            conditions.append(table.c.received_at < self.until)
        if self.batch_id is not None:
            conditions.append(table.c.batch_id == self.batch_id)
        return conditions


@dataclass(frozen=True)
class WindowData:
    """The logs of one window.

    ``predictions`` has one row per scored policy, with ``PREDICTION_COLUMNS``. ``ClaimNb`` and
    ``Exposure`` are NaN for the rows that have no ground truth yet.
    """

    predictions: pd.DataFrame
    n_rejected: int

    @property
    def n_accepted(self) -> int:
        """Number of successful ``/predict`` requests (distinct ``request_id``)."""
        return int(self.predictions[REQUEST_ID].nunique())

    @property
    def n_requests(self) -> int:
        """Number of requests, accepted or rejected."""
        return self.n_accepted + self.n_rejected

    @property
    def is_empty(self) -> bool:
        """True when the window has no row in either log table."""
        return self.n_requests == 0

    @property
    def with_ground_truth(self) -> pd.DataFrame:
        """The predictions whose claim outcome is known."""
        return self.predictions[self.predictions[CLAIM_NB].notna()]


def load_window(engine: Engine, window: Window) -> WindowData:
    """Read the predictions, their ground truth, and the rejection count for ``window``."""
    query = (
        select(
            prediction_log.c[REQUEST_ID],
            *(prediction_log.c[name] for name in FEATURE_COLUMNS),
            prediction_log.c[PREDICTED_FREQUENCY],
            ground_truth.c[CLAIM_NB],
            ground_truth.c[EXPOSURE],
        )
        .select_from(prediction_log.outerjoin(ground_truth))
        .where(*window.filters(prediction_log))
    )
    count_rejected = (
        select(func.count())
        .select_from(rejected_request_log)
        .where(*window.filters(rejected_request_log))
    )
    with engine.connect() as connection:
        rows = connection.execute(query).mappings().all()
        n_rejected = connection.execute(count_rejected).scalar_one()

    frame = pd.DataFrame(rows, columns=PREDICTION_COLUMNS)
    predictions = pd.concat(
        [
            frame[[REQUEST_ID]],
            # The same model-input dtypes as the API and the reference dataset.
            coerce_features(frame[FEATURE_COLUMNS]),
            frame[[PREDICTED_FREQUENCY, CLAIM_NB, EXPOSURE]].astype("float64"),
        ],
        axis=1,
    )
    return WindowData(predictions=predictions, n_rejected=int(n_rejected))
