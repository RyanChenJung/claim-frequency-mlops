# Use FLAML for AutoML, not H2O

Status: accepted (2026-11-17)

The team has used both FLAML and H2O. A spike on 2026-11-17 used synthetic Poisson data with exposure and ran FLAML end to end: training, MLflow logging and registration, loading through pyfunc without FLAML installed, and predicting from a FastAPI Docker container. Versions tested: Python 3.12, flaml 2.8.0, MLflow 3.18, LightGBM 4.7, XGBoost 2.1.

H2O's modelling covers our needs: it supports a Poisson distribution and `weights_column`. It loses on deployment:

- It needs Java in both the training image and the serving image (`h2o.jar` alone is 279 MB).
- MLflow's `h2o` flavor starts an H2O JVM at inference time.
- It officially supports Python only up to 3.11, with numpy < 2.

## Decision

- Search `lgbm`, `xgboost`, `xgb_limitdepth`, and `histgb`, forcing a Poisson objective through `custom_hp`:
  - `objective: poisson` for LightGBM
  - `objective: count:poisson` for both XGBoost learners
  - `loss: poisson` for HistGB
- Use a custom metric, exposure-weighted mean Poisson deviance, and pass `sample_weight=Exposure`.
- Set `mlflow_logging=False` with `log_type="all"`, then replay each trial from the FLAML log as a nested MLflow run.
- Refit the best configuration as a plain scikit-learn Pipeline that does not depend on FLAML, and register it with `mlflow.sklearn`. The serving image then needs only scikit-learn, LightGBM, pandas, and skops: about 185 MB compressed and about 180 MiB of memory when idle.

## Consequences (gotchas found in the spike)

- FLAML's built-in MLflow logging crashes at the end of `fit()` on MLflow ≥ 3.14, because skops became the default sklearn serialization. This is why we replay trials manually.
- `custom_hp` applies only to learners listed in `estimator_list`. Any learner left out would silently train with squared error.
- The default `log_type="better"` records only improving trials.
- skops does not trust LightGBM types by default. Pass `skops_trusted_types` when logging.
- `python:3.12-slim` lacks `libgomp1`, which LightGBM needs.
- `flaml[automl]` pins `xgboost<3`.

## Slide-ready text

> **Why FLAML, not H2O?** Both handle Poisson targets with exposure weights. H2O needs a Java runtime in both training and serving images, and its MLflow flavor starts a JVM at inference time; it also lacks official Python 3.12 support. FLAML's best configuration refits into a plain scikit-learn pipeline, so our serving container stays small and has no AutoML dependency at all.
