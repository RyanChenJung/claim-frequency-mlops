# No feature store

Status: accepted (2026-11-17)

Neither the project requirements nor the rubric require a feature store, and the problems a feature store solves mostly do not exist here. We solve the one that does, training–serving skew, by logging preprocessing and the model as a single MLflow model (one scikit-learn Pipeline). Training and serving then execute the same transformation code.

A feature store solves four problems:

| Problem | Applies to us? |
|---|---|
| Training–serving skew | Yes. Solved by the single-pipeline model |
| Low-latency online lookup by entity key | No. The API receives the full policy record |
| Point-in-time-correct historical features | No. The data is one snapshot with no timestamps |
| Feature reuse across models and teams | No. We serve one model |

## Consequences

- The API accepts raw rating factors, never precomputed features.
- A feature store becomes worthwhile when a severity model shares these features, or when real-time features such as telematics are added. We list it as future work and as the lowest-priority stretch goal.

## Slide-ready text

> **Why no feature store?** Feature stores solve four problems: training–serving skew, low-latency online lookup by entity key, point-in-time-correct historical features, and feature reuse across models. Our API receives the full policy record, the data is a single snapshot without timestamps, and we serve one model — so only skew applies. We eliminate skew structurally by packaging preprocessing and the model as a single MLflow model, so training and serving execute the same code. A feature store becomes worthwhile once we add a severity model that shares these features, or real-time telematics features.
