# MLflow tracking server and model registry (Role C; docs/contracts.md, Images).
# The official image has no PostgreSQL driver, so this adds psycopg 3, the same driver the
# project uses (MONITORING_DB_URL is postgresql+psycopg://). Keep the image tag equal to the
# `mlflow` / `mlflow-skinny` client version in uv.lock (Runtime → Version coupling).
FROM ghcr.io/mlflow/mlflow:v3.18.0
RUN pip install --no-cache-dir "psycopg[binary]==3.3.9"
