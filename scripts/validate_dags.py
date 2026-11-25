"""Parse dags/ with Airflow's own loader; fail on import errors or a changed DAG shape.

Runs inside the Airflow image, as the CI step "Validate Airflow DAGs" does:

    docker run --rm -e HOST_REPO_DIR=/ci \\
      -v "$PWD/dags:/opt/airflow/dags:ro" -v "$PWD/scripts:/scripts:ro" \\
      apache/airflow:3.4.0-python3.12 python /scripts/validate_dags.py

The expected task lists mirror docs/contracts.md (DAGs), so a DAG edit that drifts from the
contract fails CI. It also exercises the drift-triggered retraining decision with stubbed
Docker and trigger calls, since a live exit 10 needs a real `monitor`.
"""

import sys
from unittest import mock

from airflow.dag_processing.dagbag import DagBag

DAG_FOLDER = "/opt/airflow/dags"
EXPECTED_TASKS = {
    "training": ["dvc_repro", "train", "register"],
    "monitor": ["monitor", "retrain_on_drift", "monitor_failed"],
}
EXPECTED_TRIGGER_RULES = {
    ("monitor", "retrain_on_drift"): "all_failed",
    ("monitor", "monitor_failed"): "one_failed",
}


class FakeTaskInstance:
    """Minimal stand-in for the runtime task instance: records and serves one XCom."""

    def __init__(self, pulled=None):
        self.pushed = {}
        self.pulled = pulled

    def xcom_push(self, key, value):
        """Record a pushed value."""
        self.pushed[key] = value

    def xcom_pull(self, task_ids, key):
        """Return the canned upstream value."""
        return self.pulled


def check_retrain_wiring(monitor) -> list[str]:
    """Exercise TrainerOperator and RetrainOnDriftOperator with Docker and triggering stubbed."""
    from airflow.providers.docker.exceptions import DockerContainerFailedException
    from airflow.providers.docker.operators.docker import DockerOperator
    from airflow.providers.standard.operators.trigger_dagrun import TriggerDagRunOperator
    from airflow.sdk.exceptions import AirflowSkipException

    problems = []
    monitor_task = monitor.get_task("monitor")
    failure = DockerContainerFailedException("Docker container failed: {'StatusCode': 10}")
    ti = FakeTaskInstance()
    with mock.patch.object(DockerOperator, "execute", side_effect=failure):
        try:
            monitor_task.execute({"ti": ti})
            problems.append("TrainerOperator swallowed the container failure")
        except DockerContainerFailedException:
            pass
    if ti.pushed.get("exit_code") != 10:
        problems.append(f"TrainerOperator pushed {ti.pushed}, expected exit_code=10")

    retrain = monitor.get_task("retrain_on_drift")
    if retrain.trigger_dag_id != "training":
        problems.append(f"retrain_on_drift triggers {retrain.trigger_dag_id!r}, not 'training'")
    cases = [
        (10, True, True),
        (10, False, False),
        (3, True, False),
        (1, True, False),
        (124, True, False),
    ]
    for exit_code, enabled, expect_trigger in cases:
        ti = FakeTaskInstance(pulled=exit_code)
        context = {"ti": ti, "params": {"retrain_on_drift": enabled}}
        with mock.patch.object(TriggerDagRunOperator, "execute") as trigger:
            try:
                retrain.execute(context)
                triggered = trigger.called
            except AirflowSkipException:
                triggered = False
        if triggered != expect_trigger:
            problems.append(
                f"retrain_on_drift with exit {exit_code}, enabled={enabled}: "
                f"triggered={triggered}, expected {expect_trigger}"
            )
    return problems


def main() -> int:
    """Return 0 when every DAG imports and matches the expected tasks, else 1."""
    # The dag-processor puts the DAG bundle folder on sys.path while parsing, which is how
    # training.py and monitor.py import claimfreq_tasks and monitor_window. A bare DagBag
    # does not, so do the same here.
    sys.path.insert(0, DAG_FOLDER)
    bag = DagBag(dag_folder=DAG_FOLDER)
    problems = [f"import error in {path}:\n{error}" for path, error in bag.import_errors.items()]
    for dag_id, tasks in EXPECTED_TASKS.items():
        dag = bag.dags.get(dag_id)
        if dag is None:
            problems.append(f"missing DAG {dag_id!r}")
            continue
        found = [task.task_id for task in dag.topological_sort()]
        if found != tasks:
            problems.append(f"DAG {dag_id!r} has tasks {found}, expected {tasks}")
        retried = [task.task_id for task in dag.tasks if task.retries]
        if retried:
            problems.append(f"DAG {dag_id!r} retries {retried}; the contract says retries=0")
    # A plain cron schedule gives zero-length data intervals in Airflow 3; monitor needs real
    # ones, or every scheduled run checks an empty window (dags/monitor.py).
    monitor = bag.dags.get("monitor")
    timetable = type(monitor.timetable).__name__ if monitor is not None else None
    if timetable not in (None, "CronDataIntervalTimetable"):
        problems.append(f"monitor uses {timetable}, not a data-interval timetable")
    if monitor is not None:
        # A hung run must end by itself, or every later hourly run waits behind it.
        command = monitor.get_task("monitor").command
        if not command.startswith("timeout "):
            problems.append(f"monitor runs {command!r}, not under `timeout`")
        for (dag_id, task_id), rule in EXPECTED_TRIGGER_RULES.items():
            trigger_rule = bag.dags[dag_id].get_task(task_id).trigger_rule
            found_rule = str(getattr(trigger_rule, "value", trigger_rule))
            if found_rule != rule:
                problems.append(
                    f"{dag_id}.{task_id} trigger_rule {found_rule!r}, expected {rule!r}"
                )
        problems += check_retrain_wiring(monitor)
    for problem in problems:
        print(problem, file=sys.stderr)
    if not problems:
        print(f"OK: {sorted(bag.dag_ids)} imported with the expected tasks")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
