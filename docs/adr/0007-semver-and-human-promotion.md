# Semantic versions on registry entries; promotion requires a human

Status: accepted (2026-11-17)

The MLflow registry numbers model versions 1, 2, 3, and so on, which tells a client nothing about what changed. Every registered version therefore also gets a `semver` tag. The register step computes the next version by comparing the new model's schema hash, DVC data hash, and algorithm with the current champion's:

| Bump | Meaning for API clients | Example |
|---|---|---|
| MAJOR | You must change your code: the input schema or API contract changed | A new required rating factor, or a renamed field |
| MINOR | Predictions change, but your code does not | Retrained on a new data version, or a different algorithm |
| PATCH | Practically the same model | Hyperparameter tweak or bug fix with the same data and algorithm |

The first champion is `1.0.0`.

Two refinements, decided when `register` was implemented (#47):

- **The bump applies to the highest semver registered so far**, not to the champion's. Two challengers registered before a promotion therefore never share a version string.
- **Without a champion**, the first registration is `1.0.0` and every later one is a MINOR bump; the schema is not compared, since no client is served yet.

So the first champion is `1.0.0` only if it is the first version registered. Before the real first training run, the team resets the registry (`MlflowClient().delete_registered_model("claim-frequency")`, which keeps runs and artifacts) and runs `register` only once before the first `promote`. A grader's first run on a clean machine gets `1.0.0` without any reset.

Every new best model is registered with the alias `challenger`. The register step compares it with the `champion` on the same holdout and logs the comparison to MLflow. **A human runs `promote` to move the `champion` alias.** The model sets premiums, so a person signs off before it goes live (the Week 10 topic "The human trust factor"). The API loads `@champion` at startup, so promotion means moving the alias and restarting the `api` container.

## Considered options

- **Automatic promotion when the challenger scores better:** rejected for the core scope. It could silently promote a model trained on drifted or corrupted data.
- **Baking the model into the API image at build time:** more immutable, but every model switch would require an image rebuild.

## Consequences

- Retraining is triggered manually. The training DAG trains on whatever data version the checked-out `dvc.lock` points to; to retrain on another data version, check out that git revision and trigger the DAG. An automatic trigger from the `monitor` DAG is a stretch goal, and even then promotion stays manual.
