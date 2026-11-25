# Datasheet: freMTPL2freq (OpenML 41214)

This datasheet follows [Gebru et al., "Datasheets for Datasets"](https://arxiv.org/abs/1803.09010). It answers the questions that apply to this dataset and skips the rest. The [Known limitations](#known-limitations) section at the end collects everything a model user should know.

| | |
|---|---|
| Dataset | `freMTPL2freq`, French motor third-party liability (TPL) policies and claim counts |
| Copy we use | [OpenML dataset 41214](https://www.openml.org/d/41214), the pre-2022 CASdatasets version ([ADR-0001](adr/0001-openml-as-data-source.md)) |
| Size | 678,013 policies, 12 columns |
| Upstream | [CASdatasets](https://github.com/dutangc/CASdatasets) R package (C. Dutang, A. Charpentier) |
| Original source | An unnamed private French insurer |
| Numbers on this page | Most come from [`docs/eda/summary.json`](eda/summary.json), written by `uv run python scripts/eda.py` from train split md5 `6ccf5987bb4772e6412c4469f9384454`. The split sizes, the Kaggle comparison and the 2022-revision figures come from elsewhere, and each is cited where it appears. |
| Maintained by | Role A (Data), @lx977 |

## Motivation

**Why was the dataset created?** CASdatasets collects insurance datasets for teaching and research. It started as the companion to the book *Computational Actuarial Science with R* (A. Charpentier, ed.). The insurer's own reason for collecting the data is not documented. Policy and claim records like these are normally kept for underwriting and claims handling.

**Why do we use it?** Our pipeline predicts the expected number of claims per policy per year. `freMTPL2freq` is a standard benchmark for this task: it appears in the actuarial literature cited by CASdatasets and in scikit-learn's Poisson regression example. It is large enough for AutoML, and its known structure gives us realistic drift scenarios to build.

**Who created and funded it?** The data comes from an unnamed insurer. The CASdatasets maintainers prepared and published it. No funding is documented.

## Composition

**What does each row represent?** One motor TPL policy, observed over an exposure period of up to 2.01 years in the raw data (capped at 1 year by `clean`). Rows are policies, not people: one policyholder may appear on several rows (see [Known limitations](#known-limitations)).

**Columns.** The meanings come from the CASdatasets documentation. The last column gives the ranges observed in the data. The validity bounds in `src/claimfreq/contracts/schema.py` are wider for `BonusMalus` (50–350) and `Density` (1–50,000).

| Column | Type | Meaning | Observed values |
|---|---|---|---|
| `IDpol` | int | Policy ID, links to the severity table `freMTPL2sev` (not used here) | unique |
| `ClaimNb` | int | Number of claims during the exposure period (target) | 0–16 raw; capped at 4 |
| `Exposure` | float | Length of the exposure period, in years (Poisson offset) | up to 2.01 raw; capped at 1.0 |
| `VehPower` | int | Car power, ordered values | 4–15 |
| `VehAge` | int | Vehicle age, in years | 0–100 |
| `DrivAge` | int | Driver age, in years (the French minimum driving age was 18 when the data was collected; it has been 17 since 2024) | 18–100 |
| `BonusMalus` | int | Bonus-malus coefficient: below 100 is a bonus, above 100 a malus | 50–230 |
| `VehBrand` | str | Car brand, anonymised into codes | 11 codes: `B1`–`B6`, `B10`–`B14` |
| `VehGas` | str | Fuel: `Diesel` or `Regular` | 2 values |
| `Area` | str | Density class of the driver's municipality, from `A` (rural) to `F` (urban centre) | 6 values |
| `Density` | int | Inhabitants per km² of the driver's municipality | 1–27,000 |
| `Region` | str | Pre-2016 French region, as an INSEE code | 22 codes, `R11`–`R94` |

**Is there a label?** Yes. `ClaimNb` is the target, and `Exposure` is the offset. A policy's frequency is `ClaimNb / Exposure`, in claims per policy-year. Every frequency quoted on this page for a group of policies is `sum(ClaimNb) / sum(Exposure)` over the group, not the mean of the policies' frequencies.

**Is anything missing?** No column has missing values. Ingest validates every row against the schema, and the schema rejects nulls.

**Is it a sample?** It comes from one insurer's portfolio. How the insurer or CASdatasets selected these policies is not documented, so we can't say whether the data represents French motor TPL in general.

**What does the data look like?** These figures come from the train split (474,609 policies, after the caps). The plots are in [`docs/eda/`](eda/).

- **Claims are rare.** 94.98% of policies have no claim, and the portfolio averages 0.1006 claims per policy-year ([claim_count.png](eda/claim_count.png)).
- **Most policies are observed for less than a year.** Mean exposure is 0.53 years, and 24.96% of policies have a full year (after the cap). There is a second peak just under 0.1 years, about one month ([exposure.png](eda/exposure.png)).
- **Frequency by factor:**
  - Drivers aged 18–20 claim 0.249 times per year, 2.5 times the average ([freq_drivage.png](eda/freq_drivage.png)).
  - Frequency rises slowly with `BonusMalus` below 100 and steeply from 100 on. The six bands from 50 (the best level) to 90–99 go from 0.080 to 0.153, a gain of 0.073 in total. Then exactly 100, where a new driver starts, is at 0.270, 101–124 at 0.342, and 125 and above at 0.480 ([freq_bonusmalus.png](eda/freq_bonusmalus.png)). `BonusMalus` 100 holds 78% of the policies from 100 to 124, so it has a band of its own.
  - Frequency rises from 0.081 in `Area` A to 0.139 in `Area` F ([freq_area.png](eda/freq_area.png)).
  - Vehicles of age 0 claim 0.315 times per year, 3 times the average. This spike looks like an artefact of the pre-2022 claim counts. Across the full dataset, the 2022 revision removes 77% of the claims on `VehAge` 0 policies (4,024 of 5,205, against 18% for older vehicles), and in the revised data `VehAge` 0 is at 0.96 times the average. Short exposures explain only a small part of it: `VehAge` 0 policies have a mean exposure of 0.29 years, but within every exposure band below one year they still have 1.7 to 4.7 times the frequency of older vehicles ([freq_vehage.png](eda/freq_vehage.png)). The comparison with the 2022 file is Ryan's, from [his review of #61](https://github.com/RyanChenJung/claim-frequency-mlops/pull/61#pullrequestreview-6050146970); our pipeline does not read that file yet ([#33](https://github.com/RyanChenJung/claim-frequency-mlops/issues/33)).
  - Frequency falls steadily as exposure grows, from 0.457 for policies of at most 0.1 years to 0.215, 0.125, 0.102 and 0.087 across the bands in between, and 0.069 for full-year policies (`frequency_by_exposure` in `summary.json`; see limitation 8).

**Recommended splits.** None upstream. Our `split` stage draws a random split with seed 42, stratified on whether a policy has a claim, giving 474,609 train, 67,801 validation and 135,603 test rows. Each split has a claimant share of 5.02%.

**Errors, noise, redundancy.** See [Known limitations](#known-limitations): duplicate rating-factor tuples, long target tails, and the corrections in the 2022 revision.

**Confidential data or direct identifiers?** No names, addresses, dates or exact locations. Brands are anonymised codes, and location is coarse: region, density class, and municipal density. A rare combination of age, vehicle and municipal density could still narrow a row down to a small group, so the data should not be joined with other sources about the same people.

**Sensitive attributes.** `DrivAge` is a sensitive attribute in its own right (age). `Region`, `Density` and `Area` are not sensitive themselves, but they can act as proxies for socioeconomic status. The data has no gender column. For context, EU law has not allowed gender-based insurance pricing since 21 December 2012 (the Test-Achats ruling of the EU Court of Justice). The source doesn't say why the column is absent.

## Collection process

**How and when was it collected?** By the insurer, as part of its normal business. The CASdatasets documentation gives the period as about 2011–2013 and says the policies are "observed mostly on one year", but only 24.96% of train policies have a full year of exposure. We don't know how the policies were extracted or anonymised.

**Was there an ethical review?** None is documented, upstream or by us.

**Did the people concerned consent?** Not documented. The policyholders were customers of the insurer, and nothing indicates that they were asked about research use. No way for a policyholder to withdraw their data is documented either.

**How do we obtain it?** `claimfreq ingest` downloads OpenML 41214 with no credentials. On 2026-11-17 we compared this copy row by row with the Kaggle copy named in our proposal ([ADR-0001](adr/0001-openml-as-data-source.md)):

- Both contain the same 678,013 policies, with the same `IDpol` set and order.
- `ClaimNb` and the 8 rating factors `VehPower`, `VehAge`, `DrivAge`, `BonusMalus`, `VehBrand`, `VehGas`, `Area` and `Density` are identical.
- The Kaggle copy replaces the `Region` codes with names, and wrongly merges R41 (Lorraine, 12,990 rows) into "Nord-Pas-de-Calais".
- The Kaggle copy rounds `Exposure` to 9 decimals.

## Preprocessing

**Upstream.** Brands were anonymised into codes before publication. Beyond that, no upstream processing is documented.

**Ours.** Each step runs as a pipeline stage, and its output is versioned with DVC ([ADR-0004](adr/0004-dvc-with-dagshub-remote.md)):

1. **`ingest`** strips the literal quotes OpenML puts around `VehGas` (`'Regular'`) and validates the raw stage. It also checks the row count, the total `ClaimNb` and the number of regions against the documented totals, so a changed OpenML version stops the pipeline. Then it converts to the canonical column order and dtypes (`IDpol` arrives as a float). Nothing else changes, and the raw file is kept as `data/raw/freMTPL2freq.parquet`.
2. **`clean`** caps `ClaimNb` at 4 (9 rows changed) and `Exposure` at 1.0 year (1,224 rows changed). This is the treatment scikit-learn's Tweedie regression example applies to this dataset. It lowers total claims from 36,102 to 36,056. No feature is transformed, because every feature transform lives inside the model's pipeline.
3. **`split`** produces the three splits described under Composition.
4. **`make-drift`** writes five synthetic drift scenarios from the test split (`docs/scope.md`, Drift scenarios). They are corruptions we made on purpose, not real data.

The code is in `src/claimfreq/` (`ingest.py`, `clean.py`, `split.py`, `make_drift.py`).

## Uses

**What it is used for here.** Training and evaluating claim-frequency models (Poisson GLM and FLAML candidates), building the reference dataset for monitoring, and producing the drift scenarios.

**Other known uses.** Actuarial pricing research and teaching. The CASdatasets documentation lists several papers (Noll, Salzmann and Wüthrich 2020; Denuit, Charpentier and Trufin 2021), and scikit-learn's Poisson regression example uses it too.

**What a user should know.** Read [Known limitations](#known-limitations) before trusting a validation or test score from this data.

**What it should not be used for.**

- Setting real premiums or making decisions about individual people.
- Drawing conclusions about drivers outside France, about other insurers, or about years after 2013.
- Anything that needs gender, claim dates or claim costs (the costs are in `freMTPL2sev`, which we don't use).

## Distribution

**Where it is available.** [OpenML 41214](https://www.openml.org/d/41214), which lists the licence as CC0. The upstream CASdatasets R package is licensed GPL (≥ 2). We note that the two licences differ and don't draw any legal conclusion from it.

**Older versions.** OpenML 41214 is still version 1 and still serves the pre-2022 file. CASdatasets keeps every earlier version in its git history.

**How we share it.** We don't redistribute the data. The data files are git-ignored. Teammates `dvc pull` from DagsHub with personal tokens, and graders rebuild everything from OpenML with `dvc repro`, which needs no account ([ADR-0004](adr/0004-dvc-with-dagshub-remote.md)).

## Maintenance

**Upstream.** C. Dutang maintains CASdatasets, and issues go to its [GitHub repository](https://github.com/dutangc/CASdatasets/issues). In 2022 the maintainers revised `freMTPL2freq`. The revision drops 22 policies (leaving 677,991) and lowers `ClaimNb` on 9,095 rows, taking total claims from 36,102 to 26,444. The OpenML copy predates the revision and has not been updated. Its description says "677,991 motor third-part liability policies". That figure comes from the CASdatasets documentation, which has given it since at least January 2017, while the file had 678,013 rows; the 2022 revision brought the file in line with it. Trust the file, not the description; `ingest` checks the row count against 678,013.

**This datasheet.** Role A updates it when the data changes. Adopting the 2022 revision is stretch goal "data v2" ([#33](https://github.com/RyanChenJung/claim-frequency-mlops/issues/33)). After any data change, re-run `uv run python scripts/eda.py`, commit the new `docs/eda/` files, and update the figures quoted here. The `md5` in `summary.json` should equal the `train.parquet` md5 in `dvc.lock`.

## Known limitations

1. **Rating-factor tuples repeat across splits.** 29.47% of validation rows and 29.70% of test rows share their exact 9-factor tuple with at least one train row, and 29.65% of train rows share theirs with another train row. Part of this is common profiles. Part may be one policyholder's contract split across several rows, which an exact-match measure can't tell apart from a common profile. Flexible models can memorise these tuples, so `val_*` and `test_*` metrics may look slightly better than true out-of-sample performance. A split grouped by tuple would remove the effect, but the project keeps the seeded random split that the scope specifies.
2. **This is the pre-2022 version, and about a quarter of its claims are not in the corrected data.** The 2022 revision ([CASdatasets commit 531aa23](https://github.com/dutangc/CASdatasets/commit/531aa236fc344c1d16bfb26e6fd453299aa2cea5), released in 1.0-12) makes `ClaimNb` equal the number of claims recorded in `freMTPL2sev` for every policy. It drops 22 policies (each with one claim) and lowers `ClaimNb` on 9,095 rows, 9,094 of them to zero, taking total claims from 36,102 to 26,444 (−26.8%). The claimant share falls from 5.02% to 3.68%, and the capped frequency from 0.1006 to 0.0737 claims per policy-year. The removals are concentrated: the revision removes 63% of the claims on policies with `Exposure` ≤ 0.1 and 77% of the claims on `VehAge` 0 policies, against 26.8% overall. Upstream calls these counts inconsistencies and doesn't say whether they are recording errors or claims with no recorded cost, so about 9,700 of the claims in our target are uncertain. These figures are Ryan's, from joining our clean file to the 2022 file in [his review of #61](https://github.com/RyanChenJung/claim-frequency-mlops/pull/61#pullrequestreview-6050146970); [#33](https://github.com/RyanChenJung/claim-frequency-mlops/issues/33) will reproduce them in the pipeline. OpenML's description quotes 677,991 policies, an old documentation figure, but its file has the 678,013 policies of the pre-2022 version.
3. **The targets are capped.** `ClaimNb` goes up to 16 and `Exposure` up to 2.01 years in the raw data. `clean` caps them at 4 and 1.0, changing 9 and 1,224 rows, so models never see the tails.
4. **There is no time dimension.** The data has no policy or claim dates, so we can't make a time-based split or study trends. Our drift scenarios are synthetic for this reason.
5. **It covers one market and one insurer.** French motor TPL, one unnamed insurer, about 2011–2013. Results may not transfer to other markets, insurers or years.
6. **Sensitive and proxy attributes.** `DrivAge` is age, which is sensitive in its own right. `Region`, `Density` and `Area` can act as proxies for socioeconomic status. There is no gender column, so nothing can be checked for gender bias.
7. **Exposure and `VehAge` 0 have unexplained structure.** Many policies last about one month: 18.8% of train policies have at most 0.1 years of exposure, and the histogram peaks just under 0.1. `VehAge` 0 policies have short exposures (mean 0.29 years) and 3 times the average frequency, but the spike disappears in the 2022 revision (limitation 2). A model trained on this data learns a `VehAge` 0 effect that the corrected data doesn't support.
8. **Claims are not proportional to exposure.** In train, policies with at most 0.1 years of exposure have a frequency of 0.457, 6.6 times the 0.069 of full-year policies, and the frequency falls steadily across the bands in between (`frequency_by_exposure` in `summary.json`). In Ryan's comparison, the 2022 revision narrows the gap to 3.1 times but doesn't close it. Our models fit `ClaimNb / Exposure` with `Exposure` as the sample weight ([contracts.md](contracts.md), Model), which assumes that claims grow in proportion to exposure. So they will under-predict claims on short policies and over-predict them on full-year ones.
