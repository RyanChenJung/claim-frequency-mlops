"""Tests for the `monitor` DAG's window arguments (dags/monitor_window.py), without Airflow."""

import shlex
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

# The DAG folder is not a package; Airflow puts it on sys.path, and so does this test.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dags"))

from monitor_window import monitor_args  # noqa: E402

NOW = datetime(2026, 12, 1, 15, 30, tzinfo=UTC)
START = datetime(2026, 12, 1, 14, 0, tzinfo=UTC)
END = datetime(2026, 12, 1, 15, 0, tzinfo=UTC)


class RunType(str):
    """Stand-in for Airflow's DagRunType, a str enum with a ``value``."""

    @property
    def value(self) -> str:
        """Return the plain string, as an enum member would."""
        return str(self)


def args(run_type, params=None, start=None, end=None) -> list[str]:
    """Render monitor_args and split it back into a list, as the shell would."""
    return shlex.split(monitor_args(run_type, params or {}, start, end, now=NOW))


def test_scheduled_run_uses_its_data_interval():
    assert args(RunType("scheduled"), start=START, end=END) == [
        "--since", START.isoformat(), "--until", END.isoformat(),
    ]  # fmt: skip


@pytest.mark.parametrize("run_type", ["manual", RunType("manual"), RunType("scheduled")])
def test_run_without_interval_or_params_covers_the_last_24_hours(run_type):
    # A manual run, or a scheduled one without an interval, falls back to the last 24 hours.
    assert args(run_type) == [
        "--since", "2026-11-30T15:30:00+00:00", "--until", NOW.isoformat(),
    ]  # fmt: skip


def test_scheduled_run_with_an_empty_interval_fails_loudly():
    # Airflow 3's plain cron schedules give start == end; monitoring that would pass silently.
    with pytest.raises(ValueError, match="empty data interval"):
        args(RunType("scheduled"), start=END, end=END)


def test_manual_run_ignores_the_data_interval():
    assert args("manual", start=START, end=END)[1] == "2026-11-30T15:30:00+00:00"


def test_params_win_over_the_data_interval():
    params = {"since": "2026-12-01T00:00:00Z", "until": "2026-12-01T06:00:00Z"}
    assert args("scheduled", params, START, END) == [
        "--since", "2026-12-01T00:00:00Z", "--until", "2026-12-01T06:00:00Z",
    ]  # fmt: skip


def test_only_since_runs_until_now():
    assert args("manual", {"since": "2026-12-01T00:00:00"}) == [
        "--since", "2026-12-01T00:00:00", "--until", NOW.isoformat(),
    ]  # fmt: skip


def test_only_until_looks_back_24_hours():
    # A naive param is read as UTC, as the CLI does.
    assert args("manual", {"until": "2026-12-01T06:00:00"}) == [
        "--since", "2026-11-30T06:00:00+00:00", "--until", "2026-12-01T06:00:00",
    ]  # fmt: skip


def test_batch_id_is_added_and_quoted():
    rendered = monitor_args("manual", {"batch_id": "portfolio_shift-1; rm"}, None, None, now=NOW)
    assert shlex.split(rendered)[-2:] == ["--batch-id", "portfolio_shift-1; rm"]
    assert "'portfolio_shift-1; rm'" in rendered  # quoted, so it stays one argument


def test_empty_batch_id_is_ignored():
    assert "--batch-id" not in args("manual", {"batch_id": None})
    assert "--batch-id" not in args("manual", {"batch_id": ""})
