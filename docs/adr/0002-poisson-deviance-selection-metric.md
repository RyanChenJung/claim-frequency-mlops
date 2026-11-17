# Select models by exposure-weighted Poisson deviance; always report Gini and calibration

Status: accepted (2026-11-17)

About 95% of policies have zero claims, so accuracy and RMSE say little; predicting zero for everyone looks "95% accurate". We therefore fit claim frequency (`ClaimNb / Exposure`) with `Exposure` as the sample weight.

**Model selection uses exposure-weighted mean Poisson deviance on the validation split.** It is the proper scoring rule for count data and the loss a Poisson GLM minimizes.

Every run also logs two metrics that are reported but do not drive selection:

- **Gini**, the ranking power on the Lorenz curve.
- **Calibration ratio**, total predicted claims divided by total actual claims.

They carry the business story:

- **Gini:** poor ranking invites adverse selection. A competitor that ranks risks better undercuts us on good drivers, and we keep the bad ones.
- **Calibration ratio:** a ratio below 1 means we under-price the portfolio as a whole.

## Consequences

- The selection metric is one config value (`selection_metric`). If the team pivots the business story, it changes that value instead of the pipeline.
- The selection metric may only be changed based on validation results. Choosing it after seeing test results would leak the test set into model selection.
- The same evaluation module computes these metrics in training and in production monitoring. In monitoring, the calibration ratio is the signal for concept drift.
