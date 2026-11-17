# claim-frequency-mlops

End-to-end MLOps pipeline that predicts the expected number of motor insurance claims per policy per year, built on the French motor third-party liability dataset `freMTPL2freq`.

Final project for **ADSP 32021 Machine Learning Operations and Inference Engineering** (University of Chicago, Autumn 2026), Group 2.

> **Status:** scope agreed; implementation starting. Setup and run instructions will be added as the pipeline is built.
>
> - [Project scope](docs/scope.md): what we build, what we don't, roles, and timeline
> - [Architecture decision records](docs/adr/): why each major choice was made
> - [Glossary](GLOSSARY.md): shared insurance and MLOps vocabulary

## Problem

Predict annual claim frequency (`ClaimNb / Exposure`) from driver, vehicle, and region attributes, so an insurer can price premiums accurately and detect shifts in portfolio risk.

## Data

- Source: [OpenML dataset 41214](https://www.openml.org/d/41214) (`freMTPL2freq`), CC0 license, 678,013 policies, 12 columns.
- OpenML 41214 contains the same 678,013 policies (identical `IDpol` set and order) with identical `ClaimNb` and rating factors as the [Kaggle copy](https://www.kaggle.com/datasets/karansarpal/fremtpl2-french-motor-tpl-insurance-claims) named in our proposal; both derive from the pre-2022 [CASdatasets](https://github.com/dutangc/CASdatasets) `freMTPL2freq`. Differences: OpenML encodes `Region` as INSEE codes (`R11`–`R94`), while the Kaggle copy uses region names and merges R41 (Lorraine) into "Nord-Pas-de-Calais"; Kaggle's `Exposure` is rounded to 9 decimals.

## Planned stack

| Concern | Tool |
|---|---|
| Orchestration | Apache Airflow (DockerOperator) |
| Data versioning | DVC, with a DagsHub remote |
| Experiment tracking and model registry | MLflow |
| AutoML | FLAML |
| Serving | FastAPI in Docker |
| Monitoring | Evidently |

## License

[MIT](LICENSE)
