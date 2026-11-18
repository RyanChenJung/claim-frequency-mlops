# Challenges log

Engineering problems we hit, how we found them, and what we changed. This feeds the slide "Challenges and future work" (rubric item g). Add an entry whenever something costs you more than an hour or surprises the team.

Each entry has four parts:
- **Symptom:** what went wrong, or would have.
- **Found by:** how we noticed.
- **Fix:** what we changed, with a link to the PR or ADR.
- **Lesson:** what we would tell another team.

## Shortlist for the slides

1. **[Airflow 3's `@hourly` gives empty data intervals](#airflow-3-hourly-gives-empty-data-intervals)**: monitoring would have passed silently forever.
2. **[A contract bug caught before any code](#column_swap-could-not-pass-our-own-schema)**: four role-perspective reviews of the contracts found that a drift scenario contradicted the schema.
3. **[The two copies of the dataset are not the same](#the-kaggle-and-openml-copies-differ)**: a row-level comparison changed our data source.

---

## Data and contracts

### The Kaggle and OpenML copies differ

- **Symptom:** our proposal named the Kaggle copy of `freMTPL2freq`. We wanted to ingest from OpenML instead, because it needs no credentials, but only if the data was identical.
- **Found by:** a row-level comparison of all 678,013 policies against OpenML and six historical CASdatasets versions.
  - `ClaimNb` and 8 rating factors are identical.
  - The Kaggle copy merges Lorraine (`R41`, 12,990 rows) into "Nord-Pas-de-Calais" and rounds `Exposure`.
  - Both copies predate a 2022 revision that lowers `ClaimNb` on 9,095 rows.
- **Fix:** ingest from OpenML 41214. The README states the exact differences ([ADR-0001](adr/0001-openml-as-data-source.md)).
- **Lesson:** "same dataset" is a claim to test, not to assume. A provenance check took minutes and found a labelling bug.

### `column_swap` could not pass our own schema

- **Symptom:** the scope said the `column_swap` drift scenario (swap `VehAge` and `VehPower`) would "stay plausible and pass validation". But `VehPower` is bounded to 4–15 while `VehAge` spans 0–100, so about half of the swapped rows were invalid. The API would have rejected every batch, and the "silent failure" story would have collapsed into a second schema break.
- **Found by:** before any feature code, we reviewed the contracts PR from each role's point of view. Three of the six reviewers independently flagged it.
- **Fix:** swap only on rows where both values are valid in the other column, about 57% of rows ([#2](https://github.com/RyanChenJung/claim-frequency-mlops/pull/2)).
- **Lesson:** simulating each teammate's first week against the contracts is cheap. Finding this after Roles A and D had built on the schema would have forced a MAJOR schema bump.

### Silent value loss and ambiguous exit codes in the contracts code

- **Symptom:**
  - `astype(int64)` silently truncated `35.7` to `35`.
  - A constant model's Gini depended on row order when predictions tied (anywhere from −0.93 to +0.93 instead of 0).
  - A failed check and a crash both exited with code 1.
- **Found by:** an adversarial correctness review that recomputed every metric independently.
- **Fix:**
  - Coercion is lossless: it raises on non-integral values, nulls, and int64 overflow.
  - The Gini is tie-aware.
  - Exit code 10 means "check failed" and 1 means "crash" ([#2](https://github.com/RyanChenJung/claim-frequency-mlops/pull/2)).
- **Lesson:** test invariants, not examples: a constant predictor has Gini 0, and the result must not change under a shuffle.

## Platform

### Airflow 3 `@hourly` gives empty data intervals

- **Symptom:** with `schedule="@hourly"`, every scheduled `monitor` run had `data_interval_start == data_interval_end`. `monitor` would have checked an empty window, logged "no data", and exited 0, so it would pass silently forever.
- **Found by:** reading the rendered command of a real scheduled run: `--since 01:00 --until 01:00`.
- **Fix:** three layers ([#42](https://github.com/RyanChenJung/claim-frequency-mlops/pull/42)):
  - `CronDataIntervalTimetable`, so the 02:00 run covers `[01:00, 02:00)`;
  - `monitor_window.py` raises on an empty interval;
  - CI fails if `monitor` loses its data-interval timetable.
- **Lesson:** a monitoring system that can fail silently needs a test that its own inputs are non-empty. Airflow 3 changed this default from Airflow 2.

### The official MLflow image cannot talk to Postgres, or to other containers

- **Symptom:** the MLflow server crashed with `No module named 'psycopg'`. Once it ran, other containers got HTTP 403.
- **Found by:** running the compose stack and calling MLflow from a container as `http://mlflow:5000`.
- **Fix:** ([#40](https://github.com/RyanChenJung/claim-frequency-mlops/pull/40))
  - The image adds psycopg 3. Its SQLAlchemy maps `postgresql://` to psycopg 3, not psycopg2.
  - The URL names the driver explicitly.
  - `--allowed-hosts` includes `mlflow`, because MLflow 3 rejects unknown `Host` headers (DNS-rebinding protection).
- **Lesson:** verify a service from where its clients actually call it, not only from `localhost`.

### Port 5000 is taken on every Mac

- **Symptom:** MLflow's default port collides with the macOS AirPlay Receiver.
- **Found by:** a review of the env defaults, confirmed with `lsof` on the demo machine.
- **Fix:** MLflow is published on 5001 (compose maps 5001→5000).
- **Lesson:** check defaults against the team's actual machines.

### One uv version for the whole team

- **Symptom:** the lead's uv was 0.11.2, while a fresh install gives 0.13.9. Mixed versions can rewrite `uv.lock` back and forth.
- **Found by:** setting up CI, which installs the latest uv.
- **Fix:** `required-version = ">=0.13.9,<0.14"` in `pyproject.toml`; an older uv stops with a clear error ([#39](https://github.com/RyanChenJung/claim-frequency-mlops/pull/39)).
- **Lesson:** pin the tool that writes the lockfile, not just the dependencies.

### Compose interpolates services you never start

- **Symptom:** making the Airflow keys required (`${VAR:?}`) broke `docker compose up -d postgres mlflow` for teammates who never run Airflow.
- **Found by:** testing compose with a teammate-style `.env` that has no Airflow keys.
- **Fix:** ([#42](https://github.com/RyanChenJung/claim-frequency-mlops/pull/42))
  - Airflow moved to the `airflow` profile.
  - Its keys default to empty.
  - `airflow-init` fails with a message naming the missing keys.
- **Lesson:** test infrastructure changes with the least-privileged teammate's setup, not only the demo machine's.

### Containers launched by Airflow see the host, not Airflow

- **Symptom:** DockerOperator mount sources are resolved by the host's Docker daemon, so a path inside the Airflow container means nothing. The operator's default temp-dir mount fails for the same reason.
- **Found by:** reading the provider's behaviour while designing the DAGs.
- **Fix:**
  - `HOST_REPO_DIR` in `.env`.
  - `mount_tmp_dir=False`.
  - Secrets go through `private_environment`, so they never appear in the rendered-template view or the task logs. A check found 0 log files containing the database URL.
- **Lesson:** with Docker-in-Docker-style setups, write down whose filesystem each path refers to.

### DagsHub blocks anonymous reads, even on public repositories

- **Symptom:** we planned for graders to `dvc pull` the data anonymously from DagsHub.
- **Found by:** probing a public DagsHub repository: every unauthenticated request redirects to the login page.
- **Fix:** graders rebuild the data with `dvc repro` from the public OpenML source; teammates `dvc pull` with personal tokens stored only in `.dvc/config.local` ([ADR-0004](adr/0004-dvc-with-dagshub-remote.md)).
- **Lesson:** design reproducibility around what a stranger with no accounts can do.

### FLAML's MLflow integration breaks on MLflow 3.14+

- **Symptom:** FLAML 2.8's built-in MLflow logging crashes at the end of `fit()`. MLflow switched sklearn serialization to skops, which cannot serialize FLAML's wrapper.
- **Found by:** an end-to-end spike before choosing the AutoML tool.
- **Fix:** log trials as nested runs ourselves, and refit the best configuration as a plain scikit-learn Pipeline ([ADR-0006](adr/0006-flaml-for-automl.md)).
- **Lesson:** spike the integration you depend on, not just the library.

## Process

### Planning a three-week build for four people

- **Symptom:** the first ticket plan left two compose services unowned and put drift verification on the code-freeze day. It also gave one teammate the whole critical path (API → send-traffic → monitor).
- **Found by:** reviewing the ticket plan for coverage against the syllabus, for dependencies, and for whether each ticket was actionable.
- **Fix:**
  - Contracts first, with fakes and stubs, so M1 work runs in parallel.
  - Drift verification moved to M2.
  - `send-traffic` moved to Role A.
  - Integration split into three legs.
- **Lesson:** a dependency graph is easier to fix on paper than in week 3.
