# Orchestrate with Airflow using DockerOperator and LocalExecutor

Status: accepted (2026-11-17)

We use Apache Airflow 3 because the course covers it, it is the industry default, and its DAG view documents the pipeline well. Airflow has two problems for this project:

- **Weight:** the official compose file runs Redis, Celery workers, and more. Airflow's docs ask for at least 4 GB of Docker memory, ideally 8 GB.
- **Dependency conflicts:** Airflow pins many packages, which clash with MLflow, LightGBM, FLAML, and Evidently when installed in the same environment.

We address them as follows:

- Airflow runs with `LocalExecutor` and a trimmed compose file: API server, scheduler, and DAG processor, with no Redis or Celery.
- **Every task runs our own `trainer` image through DockerOperator.** Airflow's environment never imports ML code.
- The DAGs are thin. They call CLI commands of the ML package, which runs without Airflow: `docker compose run trainer <command>`.

## Considered options

- **Prefect 3:** lighter and pure Python, but the team preferred Airflow for course alignment.
- **CeleryExecutor:** too heavy for a laptop that also runs MLflow, the API, and monitoring.
- **A custom Airflow image with the ML dependencies installed:** reintroduces the dependency conflicts.
- **`ExternalPythonOperator`:** isolates packages, but not the operating system or system libraries.

## Consequences

- The `airflow-scheduler` container mounts `/var/run/docker.sock`, which grants host-level Docker access. It is the only one, because with `LocalExecutor` the tasks run there. This is acceptable for local development. In production we would use `KubernetesPodOperator` instead, which we list as future work.
- DockerOperator mounts refer to host paths or named volumes, and the launched container must join the compose network to reach MLflow and Postgres.
- Containers isolate dependencies, not memory. Docker Desktop still needs about 8 GB, and only the demo machine has to run the full stack.
- The DAG author and the ML authors agree only on CLI commands and file paths (the CLI contract), so they can work in parallel.
