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

## Consequences found in integration (2026-11-21)

- **FLAML results do not repeat across runs**, even with the same data, code, and `training.seed`. With a time budget, FLAML chooses the next learner and when to grow the sample from measured trial times, so machine load changes the search. Six real-data runs picked xgboost, xgb_limitdepth, lgbm, and histgb as the winner; lowering `flaml_max_iter` does not fix it. The data and the fixed candidates (`constant`, `poisson_glm`, `lightgbm`) are deterministic: only the FLAML search moves.
- **The registry, not retraining, pins a model.** To serve the same model again, promote its registered version; re-running `train` gives a different FLAML candidate.
- **A retrain on an unchanged `data_hash` can still become `challenger`**, bumped MINOR (another learner won) or PATCH (the same learner won), even when it is worse than the champion. A small `beats_champion=true` margin on the same `data_hash` (for example 0.00035 in deviance) is search noise. Promotion stays a human decision (ADR-0007), and whoever runs `promote` should read the margin with that in mind.
- **If repeatable search is ever needed:** `time_budget=-1` with a trial cap makes FLAML count trials instead of seconds (about 75 trials matched today's 5 minutes in one test), at the cost of no time limit on a slower machine. We keep the time budget for M2.

## Slide-ready text

> **Why FLAML, not H2O?** Both handle Poisson targets with exposure weights. H2O needs a Java runtime in both training and serving images, and its MLflow flavor starts a JVM at inference time; it also lacks official Python 3.12 support. FLAML's best configuration refits into a plain scikit-learn pipeline, so our serving container stays small and has no AutoML dependency at all.
