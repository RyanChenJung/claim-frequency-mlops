"""`training` DAG: dvc repro → train → register (docs/contracts.md, DAGs; ADR-0007).

- Manual trigger only. It trains on whatever data version the checked-out dvc.lock points to;
  to train on another version, check out that git revision and trigger the DAG.
- `train` scores every candidate on validation (val_* metrics), so there is no separate
  evaluate task. `promote` and `evaluate-test` are run by a human afterwards.
- The `dvc repro` exit code is DVC's own; a stub stage shows as a DVC failure, not exit 3.
"""

from __future__ import annotations

from datetime import UTC, datetime

from airflow.sdk import DAG
from claimfreq_tasks import MLFLOW, trainer_task

with DAG(
    dag_id="training",
    description="Rebuild data with DVC, train candidates, register the best as challenger",
    schedule=None,
    start_date=datetime(2026, 11, 1, tzinfo=UTC),
    catchup=False,
    max_active_runs=1,
    tags=["claimfreq"],
    doc_md=__doc__,
):
    dvc_repro = trainer_task("dvc_repro", "dvc repro")
    train = trainer_task("train", "claimfreq train", environment=MLFLOW)
    register = trainer_task("register", "claimfreq register", environment=MLFLOW)

    dvc_repro >> train >> register
