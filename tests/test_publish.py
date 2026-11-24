"""Tests for publishing monitor results: the Evidently UI push and the Discord alert.

The Evidently UI and the webhook are faked, so no network is needed.
"""

import json
import urllib.error
import uuid
from datetime import UTC, datetime

import pandas as pd
import pytest

from claimfreq.cli import EXIT_CHECK_FAILED, EXIT_ERROR, EXIT_OK, main
from claimfreq.contracts import env
from claimfreq.monitoring import checks, publish, reference, window
from claimfreq.monitoring.checks import CheckResult, MonitorReport, Status
from claimfreq.monitoring.window import Window, WindowData

SECRET_URL = "https://discord.example/api/webhooks/123/secret-token"


class FakeProject:
    def __init__(self, name):
        self.id = uuid.uuid4()
        self.name = name


class FakeWorkspace:
    """Stands in for ``evidently.ui.workspace.RemoteWorkspace``."""

    def __init__(self):
        self.projects = []
        self.runs = []

    def search_project(self, project_name):
        return [project for project in self.projects if project.name == project_name]

    def create_project(self, name, description=None):
        project = FakeProject(name)
        self.projects.append(project)
        return project

    def add_run(self, project_id, run, name=None):
        self.runs.append((project_id, run, name))


@pytest.fixture
def workspace(monkeypatch):
    fake = FakeWorkspace()
    monkeypatch.setattr(publish, "_workspace", lambda url: fake)
    return fake


@pytest.fixture
def posted(monkeypatch):
    """Record every webhook request instead of sending it."""
    requests = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def fake_urlopen(request, timeout):
        requests.append(request)
        return Response()

    monkeypatch.setattr(publish.urllib.request, "urlopen", fake_urlopen)
    return requests


FAILED = CheckResult(checks.DATA_DRIFT, Status.FAILED, "2 of 9 rating factors drifted")


# --- push_snapshot ------------------------------------------------------------------------------


def test_push_creates_the_project_once_and_adds_one_run_per_call(workspace):
    first = publish.push_snapshot("http://ui", "snapshot-1", Window(batch_id="clean-1"))
    second = publish.push_snapshot("http://ui", "snapshot-2", Window(batch_id="column_swap-1"))

    assert first == second
    assert [project.name for project in workspace.projects] == ["claim-frequency"]
    assert [(run, name) for _, run, name in workspace.runs] == [
        ("snapshot-1", "clean-1"),
        ("snapshot-2", "column_swap-1"),
    ]


def test_snapshot_name_falls_back_to_the_window():
    since = datetime(2026, 12, 1, 13, 0, tzinfo=UTC)
    assert publish.snapshot_name(Window(since=since)) == "window from 2026-12-01T13:00:00+00:00"
    assert publish.snapshot_name(Window()) == "all traffic"


# --- The alert ----------------------------------------------------------------------------------


def test_alert_message_names_the_failed_checks_and_the_batch():
    message = publish.alert_message([FAILED], Window(batch_id="column_swap-1"))
    assert "1 check(s) failed" in message
    assert "Batch: column_swap-1" in message
    assert "data_drift: failed (2 of 9 rating factors drifted)" in message


def test_alert_message_fits_discord():
    long = CheckResult(checks.DATA_DRIFT, Status.FAILED, "x" * 5000)
    assert len(publish.alert_message([long], Window())) == publish.DISCORD_MAX_LENGTH


def test_send_alert_posts_a_discord_body(posted):
    publish.send_alert(SECRET_URL, "hello")
    (request,) = posted
    assert request.full_url == SECRET_URL
    assert request.get_method() == "POST"
    assert json.loads(request.data) == {"content": "hello"}
    assert request.get_header("Content-type") == "application/json"


# --- The CLI ------------------------------------------------------------------------------------


@pytest.fixture
def fake_monitor(monkeypatch, workspace, posted):
    """Make ``claimfreq monitor`` return a chosen report, without a database or a champion."""
    state = {"report": None}

    def run(failed, snapshot="snapshot"):
        checks_ = [FAILED] if failed else [CheckResult(checks.DATA_DRIFT, Status.PASSED, "ok")]
        state["report"] = MonitorReport(checks=checks_, snapshot=snapshot)

    class OneRequest(WindowData):
        """A window that reports one request, so the CLI goes on to the checks."""

        @property
        def n_requests(self):
            return 1

    monkeypatch.setattr(window, "load_window", lambda engine, w: OneRequest(pd.DataFrame(), 0))
    monkeypatch.setattr(reference, "load_reference", lambda uri: (pd.DataFrame(), "7"))
    monkeypatch.setattr(checks, "run_checks", lambda data, ref, params: state["report"])
    monkeypatch.setenv(env.MONITORING_DB_URL, "sqlite://")
    monkeypatch.delenv(env.ALERT_WEBHOOK_URL, raising=False)
    return run


def test_cli_pushes_the_snapshot_and_alerts_on_failure(
    fake_monitor, workspace, posted, monkeypatch
):
    monkeypatch.setenv(env.ALERT_WEBHOOK_URL, SECRET_URL)
    fake_monitor(failed=True)
    assert main(["monitor", "--batch-id", "column_swap-1"]) == EXIT_CHECK_FAILED
    assert len(workspace.runs) == 1
    assert len(posted) == 1
    assert "column_swap-1" in json.loads(posted[0].data)["content"]


def test_cli_does_not_alert_when_every_check_passes(fake_monitor, workspace, posted, monkeypatch):
    monkeypatch.setenv(env.ALERT_WEBHOOK_URL, SECRET_URL)
    fake_monitor(failed=False)
    assert main(["monitor"]) == EXIT_OK
    assert len(workspace.runs) == 1
    assert posted == []


def test_cli_without_a_webhook_logs_only(fake_monitor, posted, capsys):
    fake_monitor(failed=True)
    assert main(["monitor"]) == EXIT_CHECK_FAILED
    assert posted == []
    assert "ALERT_WEBHOOK_URL is not set" in capsys.readouterr().out


def test_cli_skips_the_push_without_a_snapshot(fake_monitor, workspace, capsys):
    fake_monitor(failed=True, snapshot=None)
    assert main(["monitor"]) == EXIT_CHECK_FAILED
    assert workspace.runs == []
    assert "no Evidently snapshot" in capsys.readouterr().out


def test_cli_push_error_exits_1_when_the_checks_pass(fake_monitor, monkeypatch, capsys):
    def broken(url):
        raise ConnectionError("Evidently UI is down")

    monkeypatch.setattr(publish, "_workspace", broken)
    fake_monitor(failed=False)
    assert main(["monitor"]) == EXIT_ERROR
    assert "could not push the snapshot" in capsys.readouterr().err


def test_cli_publish_errors_keep_exit_10_and_hide_the_webhook_url(
    fake_monitor, monkeypatch, capsys
):
    def broken(url):
        raise ConnectionError("Evidently UI is down")

    def refuse(request, timeout):
        raise urllib.error.URLError(f"refused {request.full_url}")

    monkeypatch.setattr(publish, "_workspace", broken)
    monkeypatch.setattr(publish.urllib.request, "urlopen", refuse)
    monkeypatch.setenv(env.ALERT_WEBHOOK_URL, SECRET_URL)
    fake_monitor(failed=True)

    assert main(["monitor"]) == EXIT_CHECK_FAILED
    captured = capsys.readouterr()
    assert "could not send the alert: URLError" in captured.err
    assert "secret-token" not in captured.out + captured.err
