-- Runs once, on the first start of an empty Postgres volume (docker-entrypoint-initdb.d).
-- One server, three databases (docs/contracts.md, Environment variables):
--   airflow     Airflow metadata
--   mlflow      MLflow tracking and model registry
--   monitoring  prediction_log, rejected_request_log, ground_truth (contracts/logs.py)
CREATE DATABASE airflow;
CREATE DATABASE mlflow;
CREATE DATABASE monitoring;
