"""`monitor` DAG: check live traffic, and retrain when drift is detected (docs/contracts.md, DAGs).

Tasks:

    monitor ──> retrain_on_drift ──> monitor_failed

- ``monitor`` runs `claimfreq monitor` in the trainer image, under `timeout -v 30m`. Exit 10
  means a check failed (drift or decay detected); exit 1 or 3 means a crash or a stub, and
  124 a run that hung until the timeout ended it.
- ``retrain_on_drift`` runs only when ``monitor`` failed. It triggers the ``training`` DAG
  only for exit 10 and only when the ``retrain_on_drift`` param is on; otherwise it is
  skipped. Retraining yields a new challenger; promotion stays manual (ADR-0007).
- ``monitor_failed`` fails whenever ``monitor`` failed, so the DAG run itself turns red (the
  alert) even after a successful retraining trigger. When ``monitor`` passes, it is skipped
  and the run succeeds.
- Hourly, paused at creation. Unpause it before triggering it from the CLI: Airflow leaves every
  run of a paused DAG queued, manual ones included.
- Optional run params: ``batch_id`` (one send-traffic run), ``since`` / ``until`` (ISO
  8601; the window rules live in monitor_window.py), and ``retrain_on_drift`` (default on).
- ``retries=0`` everywhere: a failed check would only repeat the alert.
"""

from __future__ import annotations

from datetime import UTC, datetime

from airflow.providers.standard.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.sdk import DAG, Param, TriggerRule, task
from airflow.sdk.exceptions import AirflowFailException, AirflowSkipException
from airflow.timetables.interval import CronDataIntervalTimetable
from claimfreq_tasks import EVIDENTLY, MLFLOW, monitoring_secrets, trainer_task
from monitor_window import monitor_args
from retrain_policy import EXIT_CODE_XCOM_KEY, should_retrain


class RetrainOnDriftOperator(TriggerDagRunOperator):
    """Trigger the training DAG only if the monitor task failed with exit 10.

    The monitor task's exit code comes from XCom (``TrainerOperator``); any other code, or
    ``retrain_on_drift=false``, skips this task instead of triggering.
    """

    def execute(self, context):
        """Decide from the monitor's exit code and the run params, then trigger or skip."""
        exit_code = context["ti"].xcom_pull(task_ids="monitor", key=EXIT_CODE_XCOM_KEY)
        enabled = context["params"].get("retrain_on_drift", True)
        if not should_retrain(exit_code, enabled):
            raise AirflowSkipException(
                f"monitor exit code {exit_code}, retrain_on_drift={enabled}: not retraining"
            )
        self.log.info("monitor exit code %s: drift detected, triggering training", exit_code)
        return super().execute(context)


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
        "retrain_on_drift": Param(
            True, type="boolean", description="Trigger the training DAG when a check fails"
        ),
    },
    user_defined_macros={"monitor_args": monitor_args},
):
    monitor = trainer_task(
        "monitor",
        # `timeout` ends a run that hangs, for example on an Evidently UI that accepts the
        # connection but never answers. The container exits 124, which never retrains, and
        # the next hourly run can start; a normal run takes seconds. `-v` logs the timeout
        # before the 124. The checks may already have run: a hang in the Evidently push
        # drops a failed check's alert and its retraining. Airflow's
        # execution_timeout would leave the container running: DockerOperator then fails to
        # remove it while it runs (auto_remove="success"), and that error replaces the timeout.
        "timeout -v 30m claimfreq monitor {{ monitor_args(dag_run.run_type, params,"
        " data_interval_start, data_interval_end) }}",
        environment={**MLFLOW, **EVIDENTLY},
        private_environment=monitoring_secrets(),
    )

    retrain_on_drift = RetrainOnDriftOperator(
        task_id="retrain_on_drift",
        trigger_dag_id="training",
        # One training run per drifting monitor run; the training DAG runs one at a time.
        trigger_run_id="drift__{{ run_id }}",
        conf={"triggered_by": "monitor", "monitor_run_id": "{{ run_id }}"},
        wait_for_completion=False,
        trigger_rule=TriggerRule.ALL_FAILED,
        retries=0,
    )

    @task(trigger_rule=TriggerRule.ONE_FAILED, retries=0)
    def monitor_failed() -> None:
        """Fail the DAG run whenever monitoring failed, so the alert stays red."""
        raise AirflowFailException("monitor failed: see the monitor task's log")

    monitor >> retrain_on_drift
    [monitor, retrain_on_drift] >> monitor_failed()
