# Run everything locally with docker compose; no cloud deployment

Status: accepted (2026-11-17)

The syllabus requires a README that reproduces the entire pipeline locally, and we have about three weeks. Cloud accounts, costs, and IAM permissions would add risk without adding grading value. One `docker compose` stack therefore runs everything:

- Postgres, with the `airflow`, `mlflow`, and `monitoring` databases
- the MLflow server
- the Airflow API server, scheduler, and DAG processor
- the Evidently UI
- the FastAPI `api` service

DockerOperator launches the `trainer` image on demand. One demo machine with 16 GB of RAM runs the full stack for the presentation. Developers run only the pieces they need.

## Consequences

- Cloud deployment and Kubernetes (`KubernetesPodOperator` instead of `docker.sock`) are future work for the slides.
