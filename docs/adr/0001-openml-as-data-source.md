# Ingest freMTPL2freq from OpenML 41214, not the Kaggle copy

Status: accepted (2026-11-17)

Our proposal named the Kaggle copy `karansarpal/fremtpl2-french-motor-tpl-insurance-claims`. We ingest [OpenML dataset 41214](https://www.openml.org/d/41214) instead for two reasons. It downloads without credentials, so laptops and CI need no Kaggle tokens. It is also the more faithful copy.

A row-level comparison on 2026-11-17 established the following:

- Both sources contain the same 678,013 policies, with the same `IDpol` set and order.
- `ClaimNb` and the rating factors `VehPower`, `VehAge`, `DrivAge`, `BonusMalus`, `VehBrand`, `VehGas`, `Area`, and `Density` are identical.
- The Kaggle copy differs in two ways:
  - It replaces the `Region` codes with names and wrongly merges R41 (Lorraine, 12,990 rows) into "Nord-Pas-de-Calais".
  - It rounds `Exposure` to 9 decimals.
- Both copies derive from the pre-2022 [CASdatasets](https://github.com/dutangc/CASdatasets) `freMTPL2freq`.

## Considered options

- **Kaggle API:** needs a credential on every machine and a secret in CI, and carries the Lorraine labelling bug.
- **CASdatasets 2022 revision:** more correct, because it drops 22 policies and lowers `ClaimNb` on 9,095 rows. But it diverges from the proposal, from the actuarial literature, and from the scikit-learn examples, and it has to be parsed from R `.rda` files. We adopt it as stretch goal "data v2" instead.

## Consequences

- `Region` stays as INSEE codes (`R11`–`R94`).
- Ingestion must cast `IDpol` from float to int, strip the literal quotes from `VehGas` (`'Regular'`), and align columns by name.
- The README must not claim the two sources are "exactly the same data". It states that they share the same policies and lists the two differences.
