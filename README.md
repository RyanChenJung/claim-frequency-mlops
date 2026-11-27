# claim-frequency-mlops

End-to-end MLOps pipeline that predicts the expected number of motor insurance claims per policy per year, built on the French motor third-party liability dataset `freMTPL2freq`.

Final project for **ADSP 32021 Machine Learning Operations and Inference Engineering** (University of Chicago, Autumn 2026), Group 2.

> **Status:** platform in place (CI, compose, trainer image, Airflow DAGs); the data, modeling, serving, and monitoring commands are being implemented. Steps in [Run it locally](#run-it-locally) say which issue each pending step waits for.
>
> - [Project scope](docs/scope.md): what we build, what we don't, roles, and timeline
> - [Contracts](docs/contracts.md): every handoff between roles, and the [dev setup](docs/contracts.md#dev-setup)
> - [Architecture decision records](docs/adr/): why each major choice was made
> - [Glossary](GLOSSARY.md): shared insurance and MLOps vocabulary
> - [Challenges log](docs/challenges.md): engineering problems we hit and how we fixed them

## Problem

Predict annual claim frequency (`ClaimNb / Exposure`) from driver, vehicle, and region attributes, so an insurer can price premiums accurately and detect shifts in portfolio risk.

## Data

- Source: [OpenML dataset 41214](https://www.openml.org/d/41214) (`freMTPL2freq`), CC0 license, 678,013 policies, 12 columns.
- OpenML 41214 contains the same 678,013 policies (identical `IDpol` set and order) with identical `ClaimNb` and rating factors as the [Kaggle copy](https://www.kaggle.com/datasets/karansarpal/fremtpl2-french-motor-tpl-insurance-claims) named in our proposal; both derive from the pre-2022 [CASdatasets](https://github.com/dutangc/CASdatasets) `freMTPL2freq`. Differences: OpenML encodes `Region` as INSEE codes (`R11`–`R94`), while the Kaggle copy uses region names and merges R41 (Lorraine) into "Nord-Pas-de-Calais"; Kaggle's `Exposure` is rounded to 9 decimals.
- Datasheet and EDA: [docs/datasheet.md](docs/datasheet.md); plots and summary numbers in [docs/eda/](docs/eda/).

## Stack

| Concern | Tool | Where |
|---|---|---|
| Orchestration | Apache Airflow 3.4 (DockerOperator) | <http://localhost:8080> |
| Data versioning | DVC, with a DagsHub remote | `dvc.yaml`, `dvc.lock` |
| Experiment tracking and model registry | MLflow 3.18 | <http://localhost:5001> |
| AutoML | FLAML | inside `claimfreq train` |
| Serving | FastAPI in Docker | <http://localhost:8000/docs> |
| Monitoring | Evidently | <http://localhost:8001> |

The architecture diagram is tracked in [#24](https://github.com/RyanChenJung/claim-frequency-mlops/issues/24).

## Run it locally

These steps reproduce the whole pipeline on one machine, from a fresh clone. Steps marked **⏳** wait for the issue named next to them; everything else works today.

### 0. Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/), with at least **8 GB** of memory in its settings. The full stack (Postgres, MLflow, Airflow, the API, and the Evidently UI) uses about 2.5 GB, plus 1–2 GB for the trainer container while `train` runs.
- [uv](https://docs.astral.sh/uv/getting-started/installation/) **0.13.9 or newer within 0.13**. An older uv stops with a version error; `uv self update` fixes it. uv installs Python 3.12 itself.
- git. On macOS, also `brew install libomp` (LightGBM and XGBoost need it).

### 1. Clone and check

```bash
git clone https://github.com/RyanChenJung/claim-frequency-mlops.git
cd claim-frequency-mlops
uv sync
uv run pytest
```

Every later command runs from this folder (the repo root).

### 2. Configure `.env`

```bash
cp .env.example .env
```

Fill in `.env`. Git ignores it, so never put these values in an issue or PR:

- `POSTGRES_PASSWORD`: any value of letters and digits.
- For Airflow:
  - `HOST_REPO_DIR`: this folder's absolute path (the output of `pwd`).
  - `AIRFLOW_FERNET_KEY`: the output of `python3 -c "import base64, os; print(base64.urlsafe_b64encode(os.urandom(32)).decode())"`.
  - `AIRFLOW_JWT_SECRET`: the output of `python3 -c "import secrets; print(secrets.token_urlsafe(64))"`.
- On Linux only, set `AIRFLOW_UID` to the output of `id -u`.

### 3. Build the data

```bash
uv run dvc repro
```

This downloads OpenML 41214 and runs `ingest → clean → split → make-drift`, the stages in `dvc.yaml`. Graders need no account for it: DagsHub denies anonymous reads even on public repositories ([ADR-0004](docs/adr/0004-dvc-with-dagshub-remote.md)), so `dvc repro` rebuilds every file from the public source instead of pulling it. A second `uv run dvc repro` skips all four stages as unchanged.

Teammates with DagsHub access can instead fetch the versioned data with `uv run dvc pull`, after setting their token ([#13](https://github.com/RyanChenJung/claim-frequency-mlops/issues/13)).

### 4. Start the services

```bash
docker compose up -d                            # postgres and mlflow
docker compose --profile tools build trainer    # the image every pipeline step runs in
docker compose --profile airflow up -d          # adds Airflow
```

- MLflow: <http://localhost:5001>.
- Airflow: <http://localhost:8080>, with no login. It is only reachable from this machine.
- Both DAGs start paused, and a paused DAG runs nothing: a manual trigger just waits in the queue. Unpause them once:

```bash
docker compose exec airflow-scheduler airflow dags unpause training
docker compose exec airflow-scheduler airflow dags unpause monitor
```

`unpause` and `pause` print the old `is_paused` value, even when they worked. To see the current state, run `docker compose exec airflow-scheduler airflow dags list` and read its `is_paused` column.

From then on, `monitor` also runs by itself, once right away and then at every full hour, each time on all of the previous hour's traffic. So the batches you send in steps 8 and 9 are checked again at the next full hour, pooled with the rest of that hour's traffic. Like a manual run, a scheduled run that fails a check triggers `training`. Until `monitor` is implemented ([#20](https://github.com/RyanChenJung/claim-frequency-mlops/issues/20)), every one of these runs fails with exit 3.

### 5. Train and register ⏳ [#15](https://github.com/RyanChenJung/claim-frequency-mlops/issues/15), [#16](https://github.com/RyanChenJung/claim-frequency-mlops/issues/16)

```bash
docker compose exec airflow-scheduler airflow dags trigger training
```

The `training` DAG runs `dvc_repro → train → register`. In MLflow, the candidates appear under the `claim-frequency` experiment, and the best one is registered as model `claim-frequency` with alias `challenger`. On the full data, `train` takes about 6 minutes, because FLAML searches for `training.flaml_time_budget_s` (300 s in `params.yaml`).

### 6. Promote and serve ⏳ [#15](https://github.com/RyanChenJung/claim-frequency-mlops/issues/15), [#17](https://github.com/RyanChenJung/claim-frequency-mlops/issues/17), [#22](https://github.com/RyanChenJung/claim-frequency-mlops/issues/22)

Pick the version number `N` of the challenger in MLflow (Models → `claim-frequency`). Promotion is a human decision ([ADR-0007](docs/adr/0007-semver-and-human-promotion.md)). To compare the challenger with the champion, open the version's source run: `register` logs the comparison there as tags, against the version that was champion when it ran (`champion_version`). `beats_champion` is `true`, `false`, `no_champion`, or `incomparable` (the schema changed), and `champion_val_<metric>` is that champion re-scored on the same validation split.

A later retrain on unchanged data (the same `data_hash` as the champion), such as the drift-triggered one in step 9, still registers a new version: MINOR if the winner's `algorithm` differs from the champion's, PATCH if not. It can also report `beats_champion` as `true` by a small margin. FLAML's search has a time budget, so each run can end with a slightly different model. A small margin on the same `data_hash` is noise, not an improvement.

```bash
docker compose run --rm trainer claimfreq promote --version N
docker compose restart api                      # the API loads @champion at startup
curl -s localhost:8000/model-info               # the first champion reports semver 1.0.0
```

`promote` moves `champion` to version N and, if `challenger` pointed at N, removes `challenger`, so `challenger` never names the current champion.

The API's interactive docs are at <http://localhost:8000/docs>.

### 7. Evaluate on the test set ⏳ [#28](https://github.com/RyanChenJung/claim-frequency-mlops/issues/28)

```bash
docker compose run --rm trainer claimfreq evaluate-test
```

This logs `test_*` metrics on the champion's run. The test set is used only here, and as clean traffic in the next step.

### 8. Clean traffic: no alert ⏳ [#20](https://github.com/RyanChenJung/claim-frequency-mlops/issues/20), [#25](https://github.com/RyanChenJung/claim-frequency-mlops/issues/25)

```bash
docker compose run --rm trainer claimfreq send-traffic --source test --batch-id clean-1
docker compose exec airflow-scheduler airflow dags trigger monitor --conf '{"batch_id": "clean-1"}'
```

The `monitor` run should succeed (green in Airflow), and the Evidently UI at <http://localhost:8001> shows the snapshot.

### 9. A drift scenario: the alert fires ⏳ [#30](https://github.com/RyanChenJung/claim-frequency-mlops/issues/30)

```bash
docker compose run --rm trainer claimfreq send-traffic --source column_swap --batch-id column_swap-1
docker compose exec airflow-scheduler airflow dags trigger monitor --conf '{"batch_id": "column_swap-1"}'
```

- The `monitor` run fails (red in Airflow).
- Its `retrain_on_drift` task triggers the `training` DAG, which registers a new challenger. Promoting it stays a human decision.
- The Evidently dashboard shows the drifted rating factors.
- If `ALERT_WEBHOOK_URL` is set, a Discord message names the failed checks.

The other scenarios are `schema_break`, `out_of_bounds`, `portfolio_shift`, and `concept_drift`. [`docs/contracts.md`](docs/contracts.md#what-the-monitor-must-catch) lists which checks each one must trip.

When you are done with the scenarios, and the last `monitor` run has finished, pause `monitor`:

```bash
docker compose exec airflow-scheduler airflow dags pause monitor
```

Otherwise the scheduled run at the next full hour checks these batches again, pooled with the rest of that hour's traffic ([step 4](#4-start-the-services)). If that run fails too, it triggers `training` a second time: one more challenger, trained on the same data, and, if `ALERT_WEBHOOK_URL` is set, one more Discord message. The pause stays after `docker compose down`, so unpause `monitor` again (step 4) before the next demo.

### 10. Stop

```bash
docker compose --profile airflow down       # keeps the data in Docker volumes
docker compose --profile airflow down -v    # also deletes the data, for a fresh start
```

### Troubleshooting

- **`no configuration file provided`**: run compose from the repo root, or add `--project-directory <repo>`.
- **`Set AIRFLOW_FERNET_KEY, AIRFLOW_JWT_SECRET, and HOST_REPO_DIR in .env`**: `airflow-init` checks these before Airflow starts. See step 2.
- **Port 5000 is taken on macOS**: that is the AirPlay Receiver, which is why MLflow is published on 5001.
- **uv `required-version` error**: run `uv self update`.
- **`password authentication failed` in `docker compose logs mlflow` after you change `POSTGRES_PASSWORD`**: Postgres sets the password only on the first start of an empty volume. Put the old value back, or start fresh with `docker compose --profile airflow down -v`. That deletes every MLflow run and model version, the Airflow history, and the prediction logs, but not `data/`.
- **Airflow tasks fail with `permission denied` on `/var/run/docker.sock` (Linux)**: the socket belongs to the `docker` group. Add that group id to `airflow-scheduler` with `group_add` in `docker-compose.yml`.
- **A task fails with exit code 3**: that command is still a stub, and the message names the role that owns it. Exit 10 means a check failed, and exit 1 an unexpected error ([contracts: exit codes](docs/contracts.md#cli--clipy)). Exit 124 means a `monitor` run hung until its 30-minute `timeout` ended it, for example on an Evidently UI that never answered. When it hangs in the Evidently push, the checks have already run: their lines are in the task log above the hang, but a failed check then sends no alert and triggers no retraining. Trigger `monitor` again once the UI answers.

## Development

Work on a branch per issue and open a PR with `Closes #N`; the [PR template](.github/pull_request_template.md) has the checklist. `main` is protected: CI (lint, format, tests, DAG validation, `requirements.txt` freshness, image builds) must be green to merge. The [dev setup](docs/contracts.md#dev-setup) and each role's [day-1 quickstart](docs/contracts.md#develop-against-fakes) are in the contracts.

## AI assistance

The course asks every submission to list its assisting sources, with the product name and version of any AI tool. Each team member lists their own:

- **RyanChenJung (C. Platform)**: Claude Code 2.1.325 to 2.1.330 (Anthropic), with the model Claude Opus 5.5 (`claude-opus-5-5`). Used to draft code, tests, docs, and PR reviews, and to run verification checks. Ryan read and approved each change before it was committed or posted. Commits it helped write carry a `Co-Authored-By: Claude Opus 5.5` trailer.
- **lx977 (A. Data)**: to be added.
- **MarSmallK (B. Modeling)**: to be added.
- **cyq01 (D. Serving and monitoring)**: to be added.

## License

[MIT](LICENSE)
