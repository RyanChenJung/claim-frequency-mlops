"""Window arguments of the `monitor` DAG task (docs/contracts.md, DAGs).

This module imports only the standard library, so pytest checks it on the host without
Airflow. The `monitor` DAG exposes ``monitor_args`` as a Jinja macro and renders its result
into the task's command line.

Rules:
- A scheduled run monitors its data interval: ``--since data_interval_start --until
  data_interval_end``.
- A manual run uses the ``since`` and ``until`` params when given. With neither, it covers the
  last 24 hours; with only one, the other side is 24 hours away (``until`` defaults to now).
- A ``batch_id`` param adds ``--batch-id``, which restricts the check to one send-traffic run.
"""

from __future__ import annotations

import shlex
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from typing import Any

# How far back a manual run looks when no window is given.
DEFAULT_LOOKBACK = timedelta(hours=24)


def _iso(value: datetime | str) -> str:
    """Render a datetime as ISO 8601; pass strings from params through unchanged.

    The CLI parses the string and reads a naive time as UTC, so params need no conversion.
    """
    return value.isoformat() if isinstance(value, datetime) else str(value)


def _parse(value: datetime | str) -> datetime:
    """Turn a param string into a UTC-aware datetime, so the 24-hour default can be derived."""
    if isinstance(value, datetime):
        parsed = value
    else:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def monitor_args(
    run_type: Any,
    params: Mapping[str, Any],
    data_interval_start: datetime | None,
    data_interval_end: datetime | None,
    now: datetime | None = None,
) -> str:
    """Return the `claimfreq monitor` arguments for one DAG run, shell-quoted.

    Args:
        run_type: The DAG run's type. Airflow passes ``dag_run.run_type``, a string enum whose
            value is ``"scheduled"`` for scheduled runs.
        params: The run's params: optional ``since``, ``until`` (ISO strings) and ``batch_id``.
        data_interval_start: The run's data interval start (may be None for a manual run).
        data_interval_end: The run's data interval end (may be None for a manual run).
        now: The current time, injectable for tests; defaults to now in UTC.
    """
    now = now or datetime.now(UTC)
    since, until = params.get("since"), params.get("until")

    if not since and not until:
        scheduled = str(getattr(run_type, "value", run_type)) == "scheduled"
        if scheduled and data_interval_start and data_interval_end:
            if data_interval_start >= data_interval_end:
                # A zero-length interval would make monitor see no data and exit 0, a silent
                # pass. The DAG must use a data-interval timetable (see monitor.py).
                raise ValueError(
                    f"empty data interval [{data_interval_start}, {data_interval_end}): "
                    "schedule the monitor DAG with CronDataIntervalTimetable"
                )
            since, until = data_interval_start, data_interval_end
        else:
            since, until = now - DEFAULT_LOOKBACK, now
    elif not until:
        until = now
    elif not since:
        since = _parse(until) - DEFAULT_LOOKBACK

    args = ["--since", _iso(since), "--until", _iso(until)]
    if params.get("batch_id"):
        args += ["--batch-id", str(params["batch_id"])]
    return shlex.join(args)
