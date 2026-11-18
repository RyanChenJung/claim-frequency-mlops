"""Parse dags/ with Airflow's own loader; fail on import errors or a changed DAG shape.

Runs inside the Airflow image, as the CI step "Validate Airflow DAGs" does:

    docker run --rm -e HOST_REPO_DIR=/ci \\
      -v "$PWD/dags:/opt/airflow/dags:ro" -v "$PWD/scripts:/scripts:ro" \\
      apache/airflow:3.4.0-python3.12 python /scripts/validate_dags.py

The expected task lists mirror docs/contracts.md (DAGs), so a DAG edit that drifts from the
contract fails CI.
"""

import sys

from airflow.dag_processing.dagbag import DagBag

DAG_FOLDER = "/opt/airflow/dags"
EXPECTED_TASKS = {
    "training": ["dvc_repro", "train", "register"],
    "monitor": ["monitor"],
}


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
    for problem in problems:
        print(problem, file=sys.stderr)
    if not problems:
        print(f"OK: {sorted(bag.dag_ids)} imported with the expected tasks")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
