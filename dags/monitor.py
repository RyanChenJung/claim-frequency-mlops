"""`monitor` DAG: one `claimfreq monitor` task per run (docs/contracts.md, DAGs).

- Hourly, paused at creation; pause it for the demo and trigger it by hand instead.
- Optional run params: ``batch_id`` (one send-traffic run) and ``since`` / ``until``
  (ISO 8601). The window rules live in monitor_window.py.
- ``retries=0``: exit 10 means a check failed, and retrying would only repeat the alert.
"""

from __future__ import annotations

from datetime import UTC, datetime

from airflow.sdk import DAG, Param
from airflow.timetables.interval import CronDataIntervalTimetable
from claimfreq_tasks import EVIDENTLY, MLFLOW, monitoring_secrets, trainer_task
from monitor_window import monitor_args

with DAG(
    dag_id="monitor",
    description="Check live traffic against the reference dataset; fail on drift",
    # Hourly, with real data intervals: the 02:00 run covers [01:00, 02:00). A plain
    # "@hourly" in Airflow 3 gives zero-length intervals (start == end), which would make
    # every scheduled run check an empty window and pass silently.
    schedule=CronDataIntervalTimetable("0 * * * *", timezone="UTC"),
    start_date=datetime(2026, 11, 17, tzinfo=UTC),
    catchup=False,
    max_active_runs=1,
    tags=["claimfreq"],
    doc_md=__doc__,
    params={
        "batch_id": Param(
            None,
            type=["null", "string"],
            maxLength=64,
            description="Only this send-traffic run (its batch_id)",
        ),
        "since": Param(None, type=["null", "string"], description="ISO 8601, inclusive"),
        "until": Param(None, type=["null", "string"], description="ISO 8601, exclusive"),
    },
    user_defined_macros={"monitor_args": monitor_args},
):
    trainer_task(
        "monitor",
        "claimfreq monitor {{ monitor_args(dag_run.run_type, params,"
        " data_interval_start, data_interval_end) }}",
        environment={**MLFLOW, **EVIDENTLY},
        private_environment=monitoring_secrets(),
    )
