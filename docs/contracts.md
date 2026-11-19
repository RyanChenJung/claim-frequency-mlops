# Contracts

> Merge after PR #1 (scope docs).

The shared definitions that let the four roles build in parallel. The code in [`src/claimfreq/contracts/`](../src/claimfreq/contracts/) is the source of truth; this page is the map.

## Why contracts come first

Four people build four parts of one pipeline in three weeks. If the modeling code waits for the real data, and the API waits for a registered model, the team works in series and integration happens on the last day. Instead, M0 fixes every handoff up front: column names and types, file paths, the registered model name and its aliases, the API bodies, the log tables, the CLI commands, environment variables, and exit codes. Each contract comes with fakes or stubs, so every role can build and test against it on day 1 without asking another role a question. When the real parts arrive, they plug in without code changes on the other side.

## Decisions this page relies on

They are recorded in [scope.md](scope.md) and the ADRs, which PR #1 adds. Domain terms such as **policy**, **rating factor**, and **champion** are defined in the [glossary](../GLOSSARY.md).

- After `promote`, restart the `api` container, because the API loads `@champion` only at startup ([ADR-0007](adr/0007-semver-and-human-promotion.md)).
- The first champion is `1.0.0` ([ADR-0007](adr/0007-semver-and-human-promotion.md)).
- The MLflow server serves artifacts (`--serve-artifacts`), so clients need no storage credentials ([scope.md, Services](scope.md#services)).
- `/health` is a readiness check: HTTP 200 once the champion is loaded, HTTP 503 before ([scope.md, Serving](scope.md#serving)).

## Who produces and who consumes what

**P** = produces (writes the data or implements the interface). **C** = consumes (reads or calls it). Owners: A. Data (@lx977), B. Modeling (@MarSmallK), C. Platform (@RyanChenJung), D. Serving and monitoring (@cyq01).

| Contract | Module | A. Data | B. Modeling | C. Platform | D. Serving and monitoring |
|---|---|---|---|---|---|
| Data schema | [`schema.py`](../src/claimfreq/contracts/schema.py) | P | C | — | C |
| Data paths and drift scenarios | [`paths.py`](../src/claimfreq/contracts/paths.py) | P | C | C | C |
| Model (registry, tags, metric keys, I/O) | [`model.py`](../src/claimfreq/contracts/model.py) | — | P | — | C |
| Shared metrics | [`metrics.py`](../src/claimfreq/metrics.py) | — | C | — | C |
| API bodies and paths | [`api.py`](../src/claimfreq/contracts/api.py) | C (`send-traffic`) | — | C | P |
| Monitoring log tables | [`logs.py`](../src/claimfreq/contracts/logs.py) | P (`ground_truth`, via `send-traffic`) | — | C | P and C |
| Params (`params.yaml`) | [`params.py`](../src/claimfreq/contracts/params.py) | P and C | P and C | — | P and C |
| Environment variables | [`env.py`](../src/claimfreq/contracts/env.py) | C | C | P | C |
| CLI commands and exit codes | [`cli.py`](../src/claimfreq/cli.py) | P | P | C | P |
| Fakes | [`fakes.py`](../src/claimfreq/fakes.py) | C | C | C | C |

For `params.yaml`, each role owns its own section (see [Params](#params--paramspy)).

## The contracts

### Data schema — [`schema.py`](../src/claimfreq/contracts/schema.py)

One row is one **policy**. A policy frame has 12 columns, in this canonical order:

| Column | dtype | Valid values | Notes |
|---|---|---|---|
| `IDpol` | int64 | ≥ 1, unique | Policy id |
| `ClaimNb` | int64 | ≥ 0; clean stage ≤ `claim_nb_cap` (4) | Claim count |
| `Exposure` | float64 | > 0; clean stage ≤ `exposure_cap` (1.0) | Exposure, in years |
| `VehPower` | int64 | 4–15 | Rating factor |
| `VehAge` | int64 | 0–100 | Rating factor |
| `DrivAge` | int64 | 18–100 | Rating factor |
| `BonusMalus` | int64 | 50–350 | Rating factor |
| `VehBrand` | str | B1–B6, B10–B14 (11 codes) | Rating factor |
| `VehGas` | str | `Regular`, `Diesel` (no quotes) | Rating factor |
| `Area` | str | A–F | Rating factor |
| `Density` | int64 | 1–50,000 | Rating factor |
| `Region` | str | 22 INSEE codes, R11–R94 | Rating factor |

- The 9 **rating factors** are `FEATURE_COLUMNS`, the model inputs. `NUMERIC_FEATURES` and `CATEGORICAL_FEATURES` split them by type.
- The bounds are **hard validity bounds**: the API rejects values outside them with HTTP 422. Most equal the observed range. Only `BonusMalus` (50–230 in the data, 50–350 by law) and `Density` (1–27,000 in the data, 1–50,000 allowed) are wider than observed, so a plausible shift on them passes validation and is left for drift monitoring to catch.
- **Stages.** `raw` is the ingest output: no upper caps on `ClaimNb` and `Exposure`. `clean` applies to the clean, split, and valid drift outputs: `ClaimNb` and `Exposure` are capped with the values from `params.yaml`.
- `validate_frame(df, stage, params)` raises one `SchemaError` that lists every violation: missing or extra columns, dtype class, nulls, out-of-range counts, unknown categories, and duplicate `IDpol`. `find_violations` returns the same list without raising. Duplicate column names are reported alone (`duplicate_column`), because no other check can run on them. A string column of dtype `object` that holds `None` or NaN gets a `null` violation, not a spurious `dtype` one, and its categories are still checked.
- **Validate first, then coerce.** Validate the frame as it was read (`validate_frame` or `find_violations`, which report every problem at once). In the **raw** stage only, a float column in an integer slot passes the dtype check when all its non-null values are whole numbers inside int64's range, as OpenML ships `IDpol` and `ClaimNb`; one value such as 35.7 or 1e20 makes it a `dtype` violation. **Clean**-stage frames must already hold integer dtypes, because `send-traffic` posts their values as is and the strict API rejects `34.0`. Pyarrow-backed columns are handled too. Then convert the frame:
  - `coerce_frame(df)` gives all 12 columns in canonical dtypes and order. Use it before writing any parquet file.
  - `coerce_features(df)` gives `df[FEATURE_COLUMNS]` in canonical dtypes. It is the one way to build model input, from API requests and from `prediction_log` rows alike.
  - Neither ever truncates, wraps, or fills, whatever the source dtype (float, `object`, `category`, uint64). A null, a non-integral value in an integer column (for example `DrivAge = 35.7`), a value outside int64 (for example `1e20`), or a duplicate column name raises `ValueError`.
- `PolicyFeatures` is the Pydantic model of the 9 rating factors that the API validates. It forbids extra fields and uses strict integers (`"35"` and `35.0` are rejected).
- **Versioning.** `SCHEMA_VERSION = "1.0.0"`. `schema_hash()` fingerprints the feature spec (names, order, dtypes, bounds, categories). The register step compares it with the **champion**'s `schema_hash` tag: a different hash means a MAJOR model version.

### Model — [`model.py`](../src/claimfreq/contracts/model.py)

- Registered model `claim-frequency`, in MLflow experiment `claim-frequency`. Aliases: `champion` (served) and `challenger` (newest best model, not yet promoted).
- **Training convention:** the target is **claim frequency** `y = ClaimNb / Exposure`, with `sample_weight = Exposure`, on clean frames. `split_xy_w(df)` returns `(X, y, w)`.
- **Input:** a DataFrame with exactly `FEATURE_COLUMNS`, in canonical order and dtypes, without `IDpol` or targets, built with `schema.coerce_features`. **Output:** a 1-D float array, one value > 0 per row, in claims per policy-year. `FrequencyModel` is the protocol.
- **Signature.** Every logged model carries an MLflow signature inferred from the validation features, so `mlflow.pyfunc` enforces the input schema: the `FEATURE_COLUMNS` names, with `int64` numeric and string categorical columns. A frame built with `coerce_features` always passes it.
- `ConstantFrequencyModel` is the **baseline model**: it predicts `sum(ClaimNb) / sum(Exposure)` for every policy. Every candidate must beat it.
- **Artifact format** (what `train` logs, `register` registers, and the API loads):
  - The champion is a scikit-learn `Pipeline` with all preprocessing (such as categorical encoding) inside it.
  - `train` logs every candidate's fitted Pipeline with `mlflow.sklearn` using skops serialization, as an MLflow logged model named `MODEL_ARTIFACT_PATH` (`model`) linked to the candidate's child run; its `skops_trusted_types` are listed at log time. `train` and `register` run in separate containers, so `register` registers the best child's logged model (`models:/<model id>`, found in the run's outputs) rather than refitting or logging the model again. The registered version's `run_id` is that child run, which also holds the reference dataset.
  - The model's `requirements.txt` lists only its runtime libraries (scikit-learn, skops, numpy, pandas, pyarrow, plus the `mlflow` pin MLflow adds; plus `lightgbm` or `xgboost` when the final estimator comes from it; never `flaml`), not the whole project.
  - It is loaded with `mlflow.pyfunc.load_model("models:/claim-frequency@champion")`.
  - Its runtime libraries are a subset of {scikit-learn, lightgbm, xgboost}. That is why the `api` dependency group carries all three ([Images](#images)).
  - The pyfunc model receives a features-only frame built with `coerce_features`.
- **Run tags** (lineage): `git_commit`, `data_hash`, `schema_hash`.
  - `git_commit` is `env.git_commit()`: `GIT_COMMIT` if set, else `git rev-parse HEAD` in the working directory. If neither works, the command fails with exit 1.
  - `data_hash` is the md5 of the bytes of the train split file actually read: `model.file_md5(path)`. DVC 3 records the same plain content md5 in `dvc.lock` when `split` declares `splits/train.parquet`, `splits/validation.parquet`, and `splits/test.parquet` as **separate file outs**, so `split` must declare them that way. (A single `splits/` directory out would get a directory hash instead.)
- **Model version tags:** `semver`, `schema_hash`, `data_hash`, `git_commit`, `algorithm` (e.g. `constant`, `poisson_glm`, `lightgbm`, `flaml:lgbm`), `selection_metric`, `selection_metric_value`.
- **Semver rule** ([ADR-0007](adr/0007-semver-and-human-promotion.md)). Nothing registered yet: `1.0.0`. Versions exist but no champion: MINOR (the schema is not compared, since no client is served yet). With a champion, compared with it: `schema_hash` differs → MAJOR; else `data_hash` or `algorithm` differs → MINOR; else PATCH. The bump applies to the **highest semver registered so far**, not the champion's, so two challengers never share a version string.
  - Consequence: the first champion is `1.0.0` only if it is the first version ever registered. Before the real first training run, reset the registry with `MlflowClient().delete_registered_model("claim-frequency")` (runs and artifacts stay; the next `register` starts again at version 1, `1.0.0`), and run `register` only once before the first `promote`.
- **Aliases.** `register` sets `challenger` on the new version. `promote --version N` moves `champion` to N and removes `challenger` if it pointed at N, so `challenger` always means "newest best model, not yet promoted".
- **Champion comparison** (Role B's own keys, for the person deciding on `promote`). `register` logs it as **tags** on the challenger's run, so the run's metrics stay the model's own (`/model-info` reports them): `champion_version` and `champion_semver` (or `none`), `champion_val_<metric>` (the champion version re-scored on the current validation split), and `beats_champion` (`true`, `false`, `no_champion`, or `incomparable` when the schema changed).
- **Metric keys** are `<prefix><name>`. Prefixes: `val_` (validation; the only one used for selection), `test_` (final champion evaluation), `live_` (monitoring). Names: `poisson_deviance`, `gini`, `calibration_ratio`.
- **Selection direction.** `METRIC_GOALS` says which way each metric is better: `poisson_deviance` → minimize, `gini` → maximize, `calibration_ratio` → closest to 1.0. `is_better(metric, candidate, incumbent)` returns True only when the candidate is strictly better; a tie or a NaN is never better. It returns a Python `bool`, also for numpy inputs. Role B's `register` uses it.
- **Candidate scope** (for Roles B and C). `train` creates one parent MLflow run per invocation and one child run per candidate. The parent and every child are tagged `candidate_set=<parent run id>` (`TAG_CANDIDATE_SET`), so a search on that tag alone also returns children; a parent has no `mlflow.parentRunId` tag. The candidates are those in `training.candidates`: `constant`, `poisson_glm`, `lightgbm`, and `flaml` (algorithm `flaml:<best learner>`). The FLAML child holds one nested run per FLAML trial (grandchildren of the parent). A trial run has no `candidate_set` tag, carries `algorithm=flaml:<learner>`, `flaml_learner`, `config.<hyperparameter>` params, and the metric `flaml_holdout_poisson_deviance`, which is measured on a holdout FLAML takes from the train split (never the validation split), so it does not use the `val_` prefix. Trials are never candidates. `register` considers only the `FINISHED` children of the most recent `FINISHED` `train` parent run (by start time). If a candidate crashes, its parent ends `FAILED`; `register` then **falls back to the most recent `FINISHED` parent** and prints which candidate set it used. (In the DAG, `register` never runs after a failed `train`, so only manual runs see the fallback.) Everything else about `register` is Role B's.
- **Reference dataset:** `register` logs it on the model's run (the `train` child run of the registered version) at `reference/reference.parquet`, with `REFERENCE_COLUMNS` (`FEATURE_COLUMNS` + `ClaimNb` + `Exposure` + `predicted_frequency`) built from the validation split. The monitor loads it from the champion.

### Shared metrics — [`metrics.py`](../src/claimfreq/metrics.py)

Training and monitoring use the same code ([ADR-0002](adr/0002-poisson-deviance-selection-metric.md)). Each function takes `(y_freq, pred_freq, exposure)`:

- `poisson_deviance`: exposure-weighted mean Poisson deviance. Lower is better. Selects models by default.
- `gini`: exposure-weighted Lorenz-curve Gini. Higher is better. Poor ranking invites adverse selection. Policies with the same prediction form one step of the curve, so a constant model scores exactly 0 and the row order never changes the result.
- `calibration_ratio`: predicted claims / observed claims. 1.0 is perfect; below 1 means under-pricing. In monitoring it is the concept-drift signal.
- `compute_all` returns all three by name; `with_prefix(metrics, MetricPrefix.VAL)` makes the MLflow keys.

### API — [`api.py`](../src/claimfreq/contracts/api.py)

| Endpoint | Request | Response |
|---|---|---|
| `POST /predict` | `PredictRequest`: optional `batch_id` (≤ 64 chars), `policies` (1–1000 × `PolicyInput`) | `PredictResponse`: `request_id`, `model {name, version, semver}`, `predictions` in input order, each `{prediction_id, policy_id, predicted_frequency}` |
| `GET /health` | — | `HealthResponse`: HTTP 200 `{"status": "ok", "model_loaded": true}` once the model is loaded; HTTP 503 `{"status": "degraded", "model_loaded": false}` before that |
| `GET /model-info` | — | `ModelInfoResponse`: `name`, `version`, `semver`, `alias`, `run_id`, `trained_at`, `schema_hash`, `data_hash`, `git_commit`, `algorithm`, `metrics` |

- `PolicyInput` is `PolicyFeatures` plus an optional `policy_id` (the `IDpol`, for traceability), from 1 to 2^63 − 1 so it fits the `BigInteger` column of `prediction_log`. Ids (`request_id`, `prediction_id`) are uuid4 strings.
- `predicted_frequency` is > 0 and finite.
- `/model-info`: `trained_at` is a timezone-aware datetime, and `metrics` holds finite values only. The API omits any metric whose value is not finite (for example a NaN Gini), because JSON has no NaN.
- `/health` is a **readiness** check, so the compose healthcheck and `send-traffic` can wait for its HTTP 200.
- **422:** FastAPI's default validation body, `{"detail": [{"loc": [...], "msg": "...", "type": "..."}, ...]}`. Every 422 is also logged to `rejected_request_log`.
- **503:** until the model has loaded, `/predict` and `/model-info` return `{"detail": "model not loaded"}`, and `/health` returns its `degraded` body.

**The app** (Role D's contract; not implemented yet):

- **Entry point:** `uvicorn claimfreq.serving.app:app --host 0.0.0.0 --port 8000`, with container port 8000.
- **Factory:** `create_app(model: FrequencyModel | None = None, model_ref: ModelInfoResponse | None = None)`, so tests inject a model without MLflow.
- **Dev identity.** With an injected model and no `model_ref`, the app reports name `claim-frequency`, version `"0"`, semver `"0.0.0"`, alias `dev`, and `"dev"` for every lineage field (`run_id`, `schema_hash`, `data_hash`, `git_commit`, `algorithm`); `trained_at` and `metrics` are Role D's choice. `/predict` responses and `prediction_log` rows carry the same name, version, and semver.
- **Champion loading.** Without an injected model, the app loads `@champion` at startup. If no champion exists yet, it retries every 30 s and stays 503 until it loads one.
- **Switching models.** After `promote`, restart the `api` container ([ADR-0007](adr/0007-semver-and-human-promotion.md)). The first champion is `1.0.0`.

### Monitoring logs — [`logs.py`](../src/claimfreq/contracts/logs.py)

Three SQLAlchemy Core tables in the `monitoring` database (`MONITORING_DB_URL`). They work on PostgreSQL and SQLite.

| Table | One row per | Key columns |
|---|---|---|
| `prediction_log` | scored policy | `prediction_id` (PK), `request_id`, `batch_id`, `received_at`, `model_name`, `model_version`, `model_semver`, `policy_id`, the 9 rating factors under their schema names, `predicted_frequency` |
| `rejected_request_log` | rejected request (422) | `request_id` (PK), `received_at`, `batch_id`, `status_code`, `error_type`, `error_detail` (JSON), `n_policies` |
| `ground_truth` | scored policy whose outcome is known | `prediction_id` (PK, FK → `prediction_log`), `ClaimNb` (≥ 0), `Exposure` (> 0), `observed_at` |

- **Connect with `make_engine(url)`.** Every writer and reader uses it, and so do the tests. On SQLite it turns on foreign keys for every connection, as PostgreSQL always has them.
- **Who creates the tables:** the API calls `create_all(engine)` at startup, and `send-traffic` and `monitor` call it too. It is safe to call repeatedly; the API creates the tables before it reports ready. If two processes race to create a table on a first start, the one that loses retries once.
- **Ground truth** arrives after the prediction, as it would after a real policy period; `send-traffic` simulates it.
- Column names match the schema and the reference dataset: the rating factors in `prediction_log`, and `ClaimNb` and `Exposure` in `ground_truth`. Raw SQL on PostgreSQL must quote these mixed-case names: `SELECT "DrivAge" FROM prediction_log`.
- Timestamps are UTC. `prediction_log_rows(request, response, received_at)` builds the rows for one `/predict` call; it rejects a naive `received_at` and converts the value to UTC. SQLite does not store the offset, so values read back from SQLite are naive UTC.
- `rejected_request_log.batch_id` is set when the body is parseable JSON with a string `batch_id` of at most 64 characters; otherwise it is NULL.
- Rejection rate = rejected requests / (rejected requests + distinct `request_id` in `prediction_log`).

### Paths and drift scenarios — [`paths.py`](../src/claimfreq/contracts/paths.py)

All paths are relative to `DATA_DIR` (default `data`). Every file is parquet in `coerce_frame` column order and dtypes, except the `schema_break` and `out_of_bounds` drift files, which break the schema on purpose.

| Path | Written by | Stage |
|---|---|---|
| `raw/freMTPL2freq.parquet` | `ingest` | raw |
| `processed/policies.parquet` | `clean` | clean |
| `splits/train.parquet`, `splits/validation.parquet`, `splits/test.parquet` | `split`, as three separate DVC file outs (see `data_hash` under [Model](#model--modelpy)) | clean |
| `drift/<scenario>.parquet` | `make-drift`, one DVC stage that writes all five files | see below |
| `fake/policies.parquet` | `fake-data` (default output, git-ignored) | clean |

#### Drift files

Each **drift scenario** (`DriftScenario`) is a corruption of the test split. All drift files keep `IDpol`, `ClaimNb`, and `Exposure` under their canonical names and dtypes. Only rating factors are corrupted, except in `concept_drift`, which changes `ClaimNb` only.

| Scenario | Exact change | Passes clean schema? |
|---|---|---|
| `schema_break` | `DrivAge` renamed to `DriverAge` on all rows | No |
| `out_of_bounds` | `DrivAge` = 250 and `BonusMalus` = -1 on all rows | No |
| `column_swap` | `VehAge` and `VehPower` swapped on the rows where both values are valid in the other column (rule below) | Yes |
| `portfolio_shift` | Weighted sampling **without replacement** from the test split, toward young (low `DrivAge`) and dense (high `Density`) policies. `IDpol` is kept, so it stays unique. The exact weights are Role A's choice, above the minimum effect size below | Yes |
| `concept_drift` | Rating factors unchanged; `ClaimNb` increased so that the true claim rate is higher, still clipped to `claim_nb_cap`. The exact factor is Role A's choice, above the minimum effect size below | Yes |

- **The `column_swap` rule.** Swap `VehAge` and `VehPower` only on the rows where both values are valid in the other column: `VehAge` must be within `VehPower`'s bounds (4–15), and `VehPower` is always within `VehAge`'s bounds (0–100). All other rows stay unchanged, and the file keeps every test row. A test in `tests/test_paths.py` applies this rule to fake policies and checks that the result passes the clean schema.
- **Minimum effect sizes**, so that each scenario trips its check ([below](#what-the-monitor-must-catch)). Role A may go further.
  - `concept_drift` raises the observed claim rate of the file by at least 25% relative to the test split.
  - `portfolio_shift` lowers mean `DrivAge` by at least 5 years and raises median `Density` by at least 50%.
- **Integration ownership.** Role D owns the end-to-end check that each scenario fails exactly the check named in the [scenario table](#what-the-monitor-must-catch), at M2 integration, on Role A's real `make-drift` outputs.
- `schema_break` and `out_of_bounds` are never validated; the API rejects them with 422.
- **What `send-traffic` sends.** It posts every column except `IDpol`, `ClaimNb`, and `Exposure` as is, with no coercion, so the broken files reach the API broken. It uses `IDpol` as `policy_id`, and keeps `ClaimNb` and `Exposure` for `ground_truth`.

#### What the monitor must catch

`monitor` runs these checks on its window ([DAGs](#dags)); the thresholds are in the `monitor` section of `params.yaml`:

| Check | Runs when (otherwise it is reported as skipped) | Fails when |
|---|---|---|
| Rejection rate | the window has at least one request | rejected requests / all requests > `rejection_rate_max` |
| Data drift | the window has at least `min_predictions` predictions; otherwise "skipped (too few predictions)" | the share of drifted rating factors is ≥ `drift_share_max` |
| Prediction drift | as data drift | drift is detected on `predicted_frequency` and `fail_on_prediction_drift` is true |
| Calibration | at least one prediction in the window has ground truth; it runs on those predictions only; otherwise "skipped (no ground truth yet)" | the live calibration ratio is outside [`calibration_min`, `calibration_max`] |
| Live deviance | as calibration | live Poisson deviance ÷ reference Poisson deviance > `live_deviance_max_ratio` |

- **No data** means no row in either `prediction_log` or `rejected_request_log` for the window (and the `--batch-id` filter). `monitor` then logs a `no data in window` line and exits 0.
- **Exit code.** A skipped check never fails the run. `monitor` exits 10 if any check that ran failed, else 0.
- `claimfreq monitor --help` shows the same rules.

Each scenario must fail a specific check:

| Scenario | Check that must fail |
|---|---|
| clean test set | none (no false alarm) |
| `schema_break`, `out_of_bounds` | rejection rate > `rejection_rate_max` |
| `column_swap` | data drift share ≥ `drift_share_max` (2 of 9 rating factors drift) and/or prediction drift |
| `portfolio_shift` | data drift share and prediction drift |
| `concept_drift` | live calibration ratio outside [`calibration_min`, `calibration_max`]; data drift passes |

### Params — [`params.py`](../src/claimfreq/contracts/params.py)

[`params.yaml`](../params.yaml) holds every tunable value and is tracked by DVC as stage params. `load_params()` validates it.

- **Ownership.** Each section belongs to one role: `split` and `clean` to A, `training` to B, `traffic` and `monitor` to D. A role may add or change keys inside its own section in its own PR. That PR updates `Params` (`params.py`), `params.yaml`, and the table below together.
- **Contract keys.** Keys read by more than one role, and the top-level `seed`, are contract changes ([Changing a contract](#changing-a-contract)).
- Role A will add its own `drift:` section later, under the same rule.
- `params.yaml` sets every key explicitly. A test checks that every field of `Params` appears in the file; it does not pin the values.
- Validation is strict: unknown keys are rejected, and integer, float, and boolean fields do not convert types (`true` is not 1, `1` is not `true`, `4.0` is not 4, `"0.2"` is not 0.2). A float field still accepts a YAML integer such as `1`.
- Seeds are integers from 0 to 2^32 − 1. `fake-data --seed` accepts the same range.

| Section | Owner | Keys (shipped values) |
|---|---|---|
| `seed` (top level) | contract | `seed: 42`, the data seed: used only by `split` and `make-drift` |
| `split`, `clean` | A | `test_fraction: 0.2`, `validation_fraction: 0.1` (of the full dataset); `claim_nb_cap: 4`, `exposure_cap: 1.0` |
| `training` | B | `selection_metric: poisson_deviance` (one of the metric names), `flaml_time_budget_s: 300`, `flaml_max_iter: 500` (at most this many FLAML trials), `flaml_holdout_ratio: 0.2` (share of the train split FLAML holds out to compare trials; the validation split is never shown to FLAML), `candidates: [constant, poisson_glm, lightgbm, flaml]` (which candidates `train` fits; must include `constant`, no repeats), `seed: 42` (the model seed); `glm:` `alpha: 0.0001`, `max_iter: 1000`, `solver: newton-cholesky` (`lbfgs` or `newton-cholesky`), `drivage_edges: [21, 25, 30, 40, 50, 60, 70]` (left edges of the DrivAge bands after 18; strictly increasing), `vehage_bins: 10` (VehAge quantile bins; at least 2); `lightgbm:` `n_estimators: 200`, `learning_rate: 0.05`, `num_leaves: 15`, `min_child_samples: 500`, `subsample: 0.8`, `colsample_bytree: 0.8` |
| `traffic` | D | `batch_size: 100` |
| `monitor` | D | `min_predictions: 200` (fewest predictions in the window for the data and prediction drift checks to run; at least 1), `drift_share_max: 0.2` (fail when the share of drifted rating factors is ≥ this value, so 2 of 9 ≈ 0.22 fails), `fail_on_prediction_drift: true`, `calibration_min: 0.9`, `calibration_max: 1.1`, `live_deviance_max_ratio: 1.2`, `rejection_rate_max: 0.05` |

### Environment variables — [`env.py`](../src/claimfreq/contracts/env.py)

Defaults target local development on the host. The compose value is what every compose service and Airflow task uses, inside the compose network. DockerOperator tasks do not inherit the compose `environment:`, so every task sets its variables explicitly, to their compose values: the "Used by" column is the per-task list. `DATA_DIR` and `GIT_COMMIT` are the exceptions: no task sets them (`GIT_COMMIT` only as a manual override) ([Runtime](#runtime)).

| Variable | Default (host) | Compose value | Used by |
|---|---|---|---|
| `DATA_DIR` | `data` | unset in every Airflow task and compose service (meaning `data` under the repo root); set only in development and tests | `ingest`, `clean`, `split`, `make-drift`, `train`, `register`, `evaluate-test`, `send-traffic`, `fake-data` |
| `MLFLOW_TRACKING_URI` | `http://localhost:5001` | `http://mlflow:5000` | `train`, `register`, `promote`, `evaluate-test`, the API, `monitor` |
| `MONITORING_DB_URL` | `sqlite:///monitoring.db` | `postgresql+psycopg://${POSTGRES_USER}:${POSTGRES_PASSWORD}@postgres:5432/monitoring` | the API, `send-traffic`, `monitor` |
| `API_URL` | `http://localhost:8000` | `http://api:8000` | `send-traffic` |
| `EVIDENTLY_UI_URL` | `http://localhost:8001` | `http://evidently-ui:8000` | `monitor` |
| `ALERT_WEBHOOK_URL` | unset | from `.env`, optional | `monitor`; when unset, alerts only go to the log |
| `GIT_COMMIT` | unset (then `git rev-parse HEAD`) | unset: the trainer image runs `git rev-parse HEAD` on the mounted repo ([Images](#images)); set it only to override | `train` (the `git_commit` tag; `register` copies it from the run) |

- **Postgres.** `postgres:17`. Role C's init script ([`docker/postgres/init-databases.sql`](../docker/postgres/init-databases.sql)) creates the `airflow`, `mlflow`, and `monitoring` databases on the first start of an empty volume. Postgres is reachable only inside the compose network; no host port is published.
- **Secrets.** `POSTGRES_USER` and `POSTGRES_PASSWORD` come from `.env`, which git ignores. Role C provides a committed `.env.example`.
- **MLflow artifacts.** The MLflow server runs with artifact serving (`--serve-artifacts`, stored in the `mlflow-artifacts` volume), so clients need no storage credentials. Artifact URIs look like `mlflow-artifacts:/<experiment>/<run>/artifacts`.
- **MLflow allowed hosts.** MLflow 3 rejects requests whose `Host` header it does not know (HTTP 403, against DNS rebinding). The server allows `mlflow`, `localhost`, and `127.0.0.1` on any port. A client that needs another name must be added to `--allowed-hosts` in `docker-compose.yml`.

### CLI — [`cli.py`](../src/claimfreq/cli.py)

Every command exists with its final name and arguments. `fake-data`, `train`, `register`, and `promote` work today; the others print `not implemented yet — owner: Role X; see docs/contracts.md` and exit 3.

| Command | Owner | Reads → writes | Runs as |
|---|---|---|---|
| `ingest` | A | OpenML 41214 → `raw/` | DVC stage |
| `clean` | A | `raw/` → `processed/` | DVC stage |
| `split` | A | `processed/` → `splits/` | DVC stage |
| `make-drift` | A | `splits/test` → all 5 `drift/` files | DVC stage |
| `train` | B | `splits/train` + `splits/validation` → one parent MLflow run, with a child run per candidate | Airflow task |
| `register` | B | best child run of the latest `train` parent run by `selection_metric` ([candidate scope](#model--modelpy)) → model version with tags, alias `challenger`, reference dataset, comparison with `champion` | Airflow task |
| `promote --version N` | B | moves alias `champion` | a human |
| `evaluate-test` | B | champion on `splits/test` → `test_*` metrics | a human, after `promote` |
| `send-traffic --source {test,<scenario>} [--batch-id ID]` | A | posts to the API in batches of `traffic.batch_size`, then writes `ground_truth` for the scored rows | a developer (see [Runtime](#runtime)) |
| `monitor [--since ISO] [--until ISO] [--batch-id ID]` | D | `prediction_log` + `ground_truth` + `rejected_request_log` vs the reference dataset → Evidently UI; posts to `ALERT_WEBHOOK_URL` if set | Airflow task |
| `fake-data --rows N [--seed S] [--id-start ID] [--out PATH]` | contracts | synthetic clean policies, `IDpol` from `--id-start` (default 1) to `--id-start + --rows - 1`, which must not exceed `api.MAX_POLICY_ID` (exit 2) → `DATA_DIR/fake/policies.parquet` by default | works now |

**Exit codes.** Any nonzero code makes Airflow mark the task failed.

| Code | Meaning |
|---|---|
| 0 | OK |
| 1 | unexpected error: any unhandled exception. `main()` prints the traceback to stderr. Python's own default for a crash is also 1 |
| 2 | usage error (argparse) |
| 3 | not implemented yet |
| 10 | check failed (validation or a monitoring test). `main()` turns an uncaught `SchemaError` into 10 and prints its message to stderr, without a traceback |

- **Validation failures.** A command owner may simply let `validate_frame` raise: `main()` reports every violation and exits 10.
- **Registry checks.** `register` exits 10 when there is nothing valid to register: no `FINISHED` train run, or no candidate with a `val_<selection_metric>`. If the best run is already a registered version, it registers nothing, re-points `challenger` at it (unless it is the champion), and exits 0. **This check runs first**, so re-running `register` is always a no-op, even after the train split changed. Only for a run not registered yet does it compare the train split's md5 under `DATA_DIR` with the run's `data_hash`, and exit 10 if they differ. `promote --version N` exits 10 when version N does not exist.
- **`send-traffic`** exits 0 when every batch got an HTTP response of 200 or 422, and prints the counts. It can wait for `/health` to return 200 before the first batch. It exits 1 on connection errors, or on 5xx (including 503) after retries.
- **`batch_id`** identifies one `send-traffic` run and is shared by every HTTP request of that run. When `--batch-id` is omitted, the default is `<source>-<UTC timestamp %Y%m%dT%H%M%SZ>`, for example `portfolio_shift-20261201T153000Z`. `send-traffic` and `monitor` reject a `--batch-id` longer than `api.MAX_BATCH_ID_LENGTH` (64) with exit 2.
- **`monitor`** exits 10 if any check that ran failed, and 0 otherwise; a skipped check never fails it, and a window without data logs a `no data in window` line and exits 0 ([What the monitor must catch](#what-the-monitor-must-catch)). The window is described under [DAGs](#dags).

## Runtime

- **Working directory.** Every command runs with the repo root as the working directory, where `params.yaml`, `dvc.yaml`, `dvc.lock`, and `.git` are. In the trainer container, DockerOperator tasks bind-mount the repo root as the working directory. The image's virtual environment lives outside that mount ([Images](#images)).
- **How to run a command.** There is one way per place:
  - On the host: `uv run claimfreq <command>` and `uv run dvc repro`.
  - Inside an image (DAG tasks, `docker compose run`): bare `claimfreq <command>`, `python ...`, or `dvc ...`, never `uv run`. The image's virtual environment is on `PATH`.
  - DVC stage `cmd`s are bare `claimfreq <stage>`. They resolve on the host too, because `uv run dvc repro` puts the project's virtual environment on `PATH`.
- **`DATA_DIR`.** It stays unset in every Airflow task and every compose service, which means `data` under the repo root. Only development and tests set it.
- **Environment.** DockerOperator tasks do NOT inherit the compose `environment:`, so every task must set its variables explicitly, to their compose values. The "Used by" column of the [environment table](#environment-variables--envpy) is the per-task list. `DATA_DIR` and `GIT_COMMIT` are excluded from this rule: no task sets them (`GIT_COMMIT` only as a manual override).
- **`git_commit`.** The lineage tag uses `GIT_COMMIT` if set, else `git rev-parse HEAD` in the working directory. If neither works, the command fails with exit 1. `env.git_commit()` implements exactly this.
- **`data_hash`.** The md5 of the bytes of the train split file actually read (`model.file_md5`); see [Model](#model--modelpy).
- **`MONITORING_DB_URL`.** In compose: `postgresql+psycopg://${POSTGRES_USER}:${POSTGRES_PASSWORD}@postgres:5432/monitoring`, with the credentials from `.env`. The dev default stays `sqlite:///monitoring.db`. `psycopg[binary]` is a base dependency, because both images need it.
- **`MLFLOW_TRACKING_URI`.** The default is `http://localhost:5001`, because macOS AirPlay Receiver occupies port 5000. Compose publishes 5001→5000, and containers use `http://mlflow:5000`.
- **MLflow without a server.** For local development, set `MLFLOW_TRACKING_URI=sqlite:///mlflow.db`. Git ignores `mlflow.db`.
- **`send-traffic` against the compose stack** runs as `docker compose run --rm trainer claimfreq send-traffic ...`, so it is inside the compose network and gets the compose environment.
- **Version coupling.** The MLflow server image version must equal the `mlflow` client version in `uv.lock`, and the Evidently UI image version must equal the `evidently` version in `uv.lock`. **MLflow is pinned to `3.18.0`**: the server image `ghcr.io/mlflow/mlflow:v3.18.0`, and the clients `mlflow==3.18.0` (`trainer` group) and `mlflow-skinny==3.18.0` (`api` group). Changing it is a contract change: the server and both clients move together.
- **uv inside Docker.** Dockerfiles must set `UV_PYTHON_PREFERENCE=only-system`, so the image's own Python 3.12 is used. The environment variable overrides `python-preference = "only-managed"` in `pyproject.toml`, which exists for developer machines.

## DAGs

Airflow 3.4.0 (`apache/airflow:3.4.0-python3.12`, which ships the docker provider) runs in compose with `LocalExecutor`: `airflow-init` (checks the Airflow keys, migrates the `airflow` database, then exits), `airflow-apiserver`, `airflow-scheduler`, and `airflow-dag-processor`. They are in the `airflow` profile and run only on the demo machine: `docker compose --profile airflow up -d`. A plain `docker compose up -d` starts only `postgres` and `mlflow`, so teammates need no Airflow keys. The UI is at <http://localhost:8080>, published on 127.0.0.1 only, without login (`SimpleAuthManager`, all users admin). DAGs start paused.

- **Files.** [`dags/training.py`](../dags/training.py) and [`dags/monitor.py`](../dags/monitor.py) define the DAGs. [`dags/claimfreq_tasks.py`](../dags/claimfreq_tasks.py) is the one DockerOperator factory, and [`dags/monitor_window.py`](../dags/monitor_window.py) holds the window rules (standard library only, tested by pytest).
- **Every task** runs `claimfreq-trainer:latest` on the `claim-frequency-mlops_default` network, bind-mounts `HOST_REPO_DIR` at `/repo` as the working directory, and has `retries=0`.
  - Each task gets only its own variables: `train` and `register` get `MLFLOW_TRACKING_URI`; `monitor` gets `MLFLOW_TRACKING_URI` and `EVIDENTLY_UI_URL`, plus `MONITORING_DB_URL` and `ALERT_WEBHOOK_URL` through `private_environment`; `dvc_repro` gets none.
  - `private_environment` values are built at run time from the Airflow containers' environment (filled from `.env`). They never appear in the DAG code, the rendered-template view, or the task logs.
- **Airflow-only settings in `.env`** (only on the machine that runs the full stack): `HOST_REPO_DIR` (the repo's absolute host path; Docker resolves mount sources on the host), `AIRFLOW_FERNET_KEY`, `AIRFLOW_JWT_SECRET`, and on Linux `AIRFLOW_UID`. `.env.example` shows how to generate them.
- **Docker socket.** Only `airflow-scheduler` mounts `/var/run/docker.sock`, because with `LocalExecutor` the tasks run there. Docker Desktop's socket is readable by the container's root group. On a Linux host the socket belongs to the `docker` group, so add that group id with `group_add` before running Airflow there.
- **CI** parses `dags/` in the same Airflow image with [`scripts/validate_dags.py`](../scripts/validate_dags.py). It fails on an import error, on a task list that differs from the lists below, on a task with retries, on a `monitor` schedule without data intervals, or on wrong retraining wiring: the trigger rules, and the decision for exit codes 10, 3, and 1, which it exercises with Docker and triggering stubbed.
- **`training` DAG:** `dvc repro` → `train` → `register`, task ids `dvc_repro`, `train`, `register`. Manual trigger only (`schedule=None`).
  - `train` evaluates the candidates on validation (`val_*` metrics); there is no separate evaluate task.
  - `evaluate-test` is run by a human after `promote`.
  - The DAG has no data-version argument. It trains on whatever `dvc.lock` the mounted repo has checked out. To train on another data version, check out that git revision and trigger the DAG.
  - The exit code of `dvc repro` is DVC's own (nonzero on failure).
  - Exit code 3 is visible only for commands a DAG task runs directly (`train`, `register`, `monitor`). Under `dvc repro`, a stub stage appears as DVC's own failure.
  - `make-drift` is one DVC stage that writes all five drift files, so `dvc repro` keeps them up to date too.
  - **Dependency for Role C.** The `dvc repro` task needs Role A's `dvc init`, `dvc.yaml`, and `dvc[s3]` in the `trainer` group. Until then, the training DAG can only be wired, not run.
- **`monitor` DAG:** `monitor` → `retrain_on_drift` → `monitor_failed`.
  - `monitor` runs `claimfreq monitor` and records the container's exit code in XCom (`TrainerOperator`).
  - `retrain_on_drift` runs only when `monitor` failed. It triggers the `training` DAG (run id `drift__<monitor run id>`, conf `{"triggered_by": "monitor", ...}`) only for exit 10 (a failed check) and only when the `retrain_on_drift` param is true, which is the default. A crash (1) or a stub (3) never retrains. Retraining only produces a new challenger: promotion stays manual ([ADR-0007](adr/0007-semver-and-human-promotion.md)). The rule is [`dags/retrain_policy.py`](../dags/retrain_policy.py).
  - `monitor_failed` fails whenever `monitor` failed, so the DAG run stays red (the alert) even after a successful trigger. When `monitor` passes, it is skipped and the run succeeds.
  - Limitation: the triggered run trains on the data version the checked-out `dvc.lock` points to. Retraining fixes drift only once that data includes the new traffic (stretch goal "data v2", or a future step that folds `ground_truth` into the training data).
  - The schedule is hourly, with real data intervals (`CronDataIntervalTimetable`): the 02:00 run covers `[01:00, 02:00)`. A plain `@hourly` in Airflow 3 gives zero-length intervals, so every scheduled run would check an empty window and pass silently; `monitor_window.py` raises on such an interval. The DAG can be paused for the demo.
  - Optional run params: `batch_id`, `since`, `until`, and `retrain_on_drift` (boolean, default true).
  - A scheduled run uses `--since {{ data_interval_start }} --until {{ data_interval_end }}`.
  - A manual run uses the `since` and `until` params when given; without them, it covers the last 24 hours. With `batch_id`, it also passes `--batch-id`.
  - Every task has `retries=0`, so a failed check (exit 10) is not retried and triggers at most one training run.
- **The `monitor` window** applies to `received_at` in `prediction_log` and `rejected_request_log`. `--since` is inclusive and `--until` exclusive. A naive time is read as UTC. Ground truth is joined by `prediction_id`, whatever its `observed_at`.
- **`--batch-id ID`** restricts `monitor` to one `send-traffic` run. This is how the demo shows each drift scenario on its own.

## Images

- **Dependency groups** ([PEP 735](https://peps.python.org/pep-0735/)) in `pyproject.toml`. Each role adds its packages to `trainer` and `api` in its own PR (`trainer` already has `mlflow` and `skops` from #14):
  - `dev`: test and lint tools. API test dependencies (for example `httpx`, which FastAPI's `TestClient` needs) go here too.
  - `trainer`: A adds `dvc[s3]`; B adds `mlflow`, `flaml[automl]`, `lightgbm`, `xgboost`, `skops`; D adds `evidently` and an HTTP client.
  - `api`: D adds `fastapi`, `uvicorn`, `mlflow-skinny`, `lightgbm`, `xgboost`, `skops`.
  - `[tool.uv] default-groups = ["dev", "trainer", "api"]`, so a plain `uv sync` on a laptop installs everything. CI uses `uv sync --locked --all-groups` ([Dev setup](#dev-setup)).
  - After a rebase, re-run `uv lock` instead of hand-merging `uv.lock`.
- **Dockerfiles:**
  - `docker/mlflow.Dockerfile`, owned by C: the official MLflow image plus the psycopg 3 PostgreSQL driver, which it lacks.
  - `docker/trainer.Dockerfile`, owned by C, installs the base dependencies + `trainer` with `uv sync --locked --no-default-groups --group trainer`.
  - `docker/api.Dockerfile`, owned by D, installs the base dependencies + `api` with `uv sync --locked --no-default-groups --group api`.
  - Both set `UV_PYTHON_PREFERENCE=only-system` ([Runtime](#runtime)) and `UV_PROJECT_ENVIRONMENT=/opt/venv`, so the virtual environment lives outside the bind-mounted repo, and put `/opt/venv/bin` on `PATH`. Every command inside an image is bare `claimfreq <command>`, `python ...`, or `dvc ...`, never `uv run` ([Runtime](#runtime)).
  - The trainer image installs `git` and runs `git config --system --add safe.directory '*'`, so `git rev-parse HEAD` works on the bind-mounted repo, which another user owns. DAG tasks therefore do not pass `GIT_COMMIT`; it stays an override.
  - CI builds whichever Dockerfiles exist.
- **Which code runs in the trainer container.** The trainer image installs the project in editable mode (uv's default) from `WORKDIR /repo`. Every DockerOperator task and the compose `trainer` service bind-mount the host repo root at `/repo` as the working directory, so the code, `params.yaml`, `dvc.lock`, and `.git` all come from the checked-out revision. Rebuild the image only when dependencies change. The `api` image is self-contained: it copies the source at build time and uses no bind mount.
- **The `trainer` service and image.** The image is tagged `claimfreq-trainer:latest`, built with `docker compose --profile tools build trainer`. The compose `trainer` service is in the `tools` profile, so `docker compose up` does not start it; run commands with `docker compose run --rm trainer claimfreq <command>`. It waits for a healthy `postgres`, and sets the compose values of `MLFLOW_TRACKING_URI`, `MONITORING_DB_URL`, `API_URL`, `EVIDENTLY_UI_URL`, and `ALERT_WEBHOOK_URL`.
- **`.dockerignore`** keeps `.env`, `.dvc/config.local`, `.venv`, `data/`, `*.db`, and `.git` out of every build context, so no secret can be baked into an image.
- **Compose services.** Role C writes the compose services `postgres`, `mlflow` (published 5001→5000), the Airflow services, `trainer`, and `api` (built from `docker/api.Dockerfile`, published 8000→8000, healthcheck on `/health`). Role D writes the `evidently-ui` service (published 8001→8000, its image version equal to `evidently` in `uv.lock`), and Role C reviews it. Role D also owns `docker/api.Dockerfile` and the Evidently project that `monitor` pushes to.
- Airflow and its docker provider live only in the Airflow image, outside this `pyproject.toml` ([ADR-0003](adr/0003-airflow-with-docker-operator.md)).
- **API entry point** and the `create_app` factory: see "The app" under [API](#api--apipy).

## Develop against fakes

[`fakes.py`](../src/claimfreq/fakes.py) produces deterministic data that satisfies the contracts:

- `make_policies(n, seed, id_start)`: clean-stage policies with `IDpol` from `id_start` (default 1). Claims follow a known log-linear frequency (`true_frequency`: higher for young drivers and high bonus-malus), with a portfolio frequency of about 0.10, so a real model can beat the baseline. Give fake train, validation, and test files different `id_start` values so they do not share ids.
- `TrueFrequencyModel`: a `FrequencyModel` that predicts `true_frequency`, with no fitting. It is the non-constant stand-in for a champion.
- `make_reference_frame(n, seed, model)`: a reference dataset, with `TrueFrequencyModel` predictions by default.
- `make_predict_request(n, seed)`: a valid `PredictRequest`.

Every role starts the same way:

```bash
uv sync
uv run pytest
uv run claimfreq fake-data --rows 10000
```

When you implement a command, remove its entries from `STUBS` in [`tests/test_cli.py`](../tests/test_cli.py) in the same PR, and add tests for the real behavior.

Then, per role:

- **A. Data.** Implement `ingest` first. Download OpenML 41214, strip the quotes from `VehGas`, and select columns by name. Then run `validate_frame(df, "raw")` on the frame as read (its float `IDpol` and `ClaimNb` pass, being whole numbers) and `coerce_frame` before writing `raw/freMTPL2freq.parquet`. Continue with `clean`, `split`, and `make-drift`; the fake policies show what a valid clean frame looks like. `make_policies` produces clean-stage frames only, so A owns any raw-stage fake it needs for testing ingest. A also owns the future `drift:` section of `params.yaml`. Last, implement `send-traffic`, which replays your test and drift files through Role D's API; test it against a stub HTTP server until Role D's app lands:

  ```bash
  uv run claimfreq fake-data --rows 10000 --seed 3 --id-start 900000 --out data/fake/splits/test.parquet
  DATA_DIR=data/fake uv run claimfreq send-traffic --source test   # stub for now: exits 3
  ```

- **B. Modeling.** Implement `train` first, against fake splits under a git-ignored data directory:

  ```bash
  uv run claimfreq fake-data --rows 10000 --seed 1 --id-start 1 --out data/fake/splits/train.parquet
  uv run claimfreq fake-data --rows 2000 --seed 2 --id-start 500000 --out data/fake/splits/validation.parquet
  uv run claimfreq fake-data --rows 10000 --seed 3 --id-start 900000 --out data/fake/splits/test.parquet
  MLFLOW_TRACKING_URI=sqlite:///mlflow.db DATA_DIR=data/fake uv run claimfreq train
  ```

  Fit with `split_xy_w`, score with `metrics.compute_all`, and log `val_*` keys, with `env.git_commit()` and `model.file_md5` for the lineage tags. `ConstantFrequencyModel` is the baseline every candidate must beat, and `is_better` compares candidates.
- **C. Platform.** There is no CLI stub to fill; the CLI is your interface. Start with the CI workflow (the commands under [Dev setup](#dev-setup)), then the compose stack and a DAG task. This smoke step works today inside the trainer image, with the repo root bind-mounted as the working directory, `DATA_DIR` unset, and `MONITORING_DB_URL` set to its compose value:

  ```bash
  claimfreq fake-data --rows 1000
  python -c "from claimfreq.contracts import env, logs; e = logs.make_engine(env.get(env.MONITORING_DB_URL)); assert e.dialect.name == 'postgresql', e.url; logs.create_all(e)"
  ```

  It proves the mounts (the file appears under `data/fake/` in the repo on the host), that `MONITORING_DB_URL` reached the container (the assertion fails if it fell back to the SQLite default; SQLAlchemy masks the password in the message), and that Postgres is reachable. A stub's exit code 3 reaches a DAG task only when the task runs the command directly (`train`, `register`, `monitor`); under `dvc repro`, a stub stage shows as DVC's own failure. The `dvc repro` task also needs Role A's DVC setup before it can run ([DAGs](#dags)).
- **D. Serving and monitoring.** Start with the FastAPI app. Build it with `create_app(model=TrueFrequencyModel())`, validate with `PredictRequest`, build the model input with `coerce_features`, and log with `prediction_log_rows` to the default SQLite `MONITORING_DB_URL`. Then implement `monitor` (compare the logs with `make_reference_frame`). Use at least 10,000 fake policies per window: on fewer, the calibration check is dominated by noise and raises false alarms even on clean traffic. To feed the API before Role A's `send-traffic` lands, post `make_predict_request` bodies from a test.

## Changing a contract

- Changing any contract (a name, type, path, bound, table column, command, environment variable, or exit code) needs a PR labelled `contract-change`, reviewed by the owners of every affected role (see the matrix above).
- A change to the feature spec (rating factor names, order, dtypes, bounds, or categories) bumps `SCHEMA_VERSION`, changes `schema_hash()`, and implies a MAJOR model version ([ADR-0007](adr/0007-semver-and-human-promotion.md)). Update the pinned hash in `tests/test_schema.py` in the same PR.
- In `params.yaml`, adding or changing a key in your own section is not a contract change (see [Params](#params--paramspy)). Changing the top-level `seed`, or a key that another role reads, is.

## Dev setup

1. Install uv: <https://docs.astral.sh/uv/getting-started/installation/>. The project requires uv 0.13.9 or newer within 0.13 (`required-version` in `pyproject.toml`), so the whole team and CI write `uv.lock` the same way; an older uv stops with an error, and `uv self update` fixes it. uv installs Python 3.12 itself; the project uses only uv-managed interpreters.
2. From the repository root:

   ```bash
   uv sync
   uv run ruff check .
   uv run ruff format --check .
   uv run pytest
   ```

   `uv sync` installs every dependency group (`dev`, `trainer`, `api`). All three checks must pass before you open a PR.
3. CI runs the same checks after `uv sync --locked --all-groups`, which also fails if `uv.lock` is out of date.
