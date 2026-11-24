"""Publish monitor results: push the Evidently snapshot, and send the alert.

- ``push_snapshot`` adds one snapshot per run to the ``claim-frequency`` project of the
  Evidently UI at ``EVIDENTLY_UI_URL``. On the first run it creates the project and its
  dashboard: three line charts, with one point per monitor run.
- ``send_alert`` posts a Discord webhook body ``{"content": "..."}`` to ``ALERT_WEBHOOK_URL``.
  The URL is a secret: it lives only in ``.env``, and this module never prints it.
"""

import json
import urllib.request
from typing import Any

from claimfreq.contracts.model import PREDICTED_FREQUENCY
from claimfreq.contracts.schema import FEATURE_COLUMNS
from claimfreq.monitoring.checks import CheckResult
from claimfreq.monitoring.window import Window

EVIDENTLY_PROJECT = "claim-frequency"
EVIDENTLY_PROJECT_DESCRIPTION = "Drift of the served claim frequency model (claimfreq monitor)"
# Discord rejects a message longer than 2000 characters.
DISCORD_MAX_LENGTH = 2000
WEBHOOK_TIMEOUT_S = 10


def _workspace(url: str) -> Any:
    """Connect to the Evidently UI at ``url``. Tests replace this function."""
    # Imported here, because Evidently is slow to import.
    from evidently.ui.workspace import RemoteWorkspace

    return RemoteWorkspace(url)


def snapshot_name(window: Window) -> str:
    """The name of a run in the Evidently UI: the batch id, else the window start."""
    if window.batch_id:
        return window.batch_id
    if window.since:
        return f"window from {window.since.isoformat()}"
    return "all traffic"


def dashboard_panels() -> list[Any]:
    """The dashboard charts, built from the metrics of ``checks.run_drift_report``.

    1. The share of drifted rating factors: the data drift check.
    2. The drift score of ``predicted_frequency``: the prediction drift check.
    3. The drift score of each rating factor, to see which ones moved.
    """
    # Imported here, because Evidently is slow to import.
    from evidently.sdk.models import PanelMetric
    from evidently.sdk.panels import line_plot_panel

    # Evidently labels each DriftedColumnsCount result with its column list, written like a
    # Python list, so the filter picks the rating factors and skips predicted_frequency.
    share = PanelMetric(
        legend="share of drifted rating factors",
        metric="DriftedColumnsCount",
        metric_labels={"columns": str(FEATURE_COLUMNS), "value_type": "share"},
    )
    prediction = PanelMetric(
        legend=PREDICTED_FREQUENCY,
        metric="ValueDrift",
        metric_labels={"column": PREDICTED_FREQUENCY},
    )
    per_factor = [
        PanelMetric(legend=column, metric="ValueDrift", metric_labels={"column": column})
        for column in FEATURE_COLUMNS
    ]
    return [
        line_plot_panel(
            "Share of drifted rating factors",
            [share],
            description="The data drift check fails at monitor.drift_share_max.",
            size="half",
        ),
        line_plot_panel(
            "Prediction drift score",
            [prediction],
            description="Evidently's drift score of predicted_frequency.",
            size="half",
        ),
        line_plot_panel(
            "Drift score per rating factor",
            per_factor,
            description="Evidently's drift score of each rating factor.",
        ),
    ]


def push_snapshot(url: str, snapshot: Any, window: Window) -> str:
    """Push ``snapshot`` to the ``claim-frequency`` project, and return the project id.

    The project is looked up by name and created if it does not exist yet. If its dashboard
    has no charts yet, the charts of ``dashboard_panels`` are added.
    """
    workspace = _workspace(url)
    projects = workspace.search_project(EVIDENTLY_PROJECT)
    if projects:
        project = projects[0]
    else:
        project = workspace.create_project(
            EVIDENTLY_PROJECT, description=EVIDENTLY_PROJECT_DESCRIPTION
        )
    # Only when empty, so charts that someone added or changed in the UI are kept.
    if not project.dashboard.model().panels:
        for panel in dashboard_panels():
            project.dashboard.add_panel(panel)
    workspace.add_run(project.id, snapshot, name=snapshot_name(window))
    return str(project.id)


def alert_message(failed: list[CheckResult], window: Window) -> str:
    """The alert text: which checks failed, and for which batch or window."""
    lines = [
        f"Claim frequency monitor: {len(failed)} check(s) failed",
        f"Batch: {window.batch_id or 'any'}",
        f"Window: {window.describe()}",
        *(f"- {check.line()}" for check in failed),
    ]
    message = "\n".join(lines)
    if len(message) > DISCORD_MAX_LENGTH:
        message = message[: DISCORD_MAX_LENGTH - 3] + "..."
    return message


def send_alert(webhook_url: str, message: str) -> None:
    """Post ``message`` to a Discord webhook. Raises on a network error or an HTTP error."""
    request = urllib.request.Request(
        webhook_url,
        data=json.dumps({"content": message}).encode(),
        headers={
            "Content-Type": "application/json",
            # Discord blocks Python's default User-Agent.
            "User-Agent": "claimfreq-monitor",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=WEBHOOK_TIMEOUT_S):
        pass
