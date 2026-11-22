"""Role A (Data): the ``claimfreq make-drift`` command.

``make-drift`` reads the test split, ``splits/test.parquet``, and writes one corrupted copy per
drift scenario to ``drift/<scenario>.parquet`` under ``DATA_DIR`` (docs/contracts.md, Drift
files). Each scenario simulates a production failure that the monitoring checks must catch:

- ``schema_break``: ``DrivAge`` renamed to ``DriverAge``. The API rejects it (422).
- ``out_of_bounds``: ``DrivAge = 250`` and ``BonusMalus = -1`` on every row. Rejected (422).
- ``column_swap``: ``VehAge`` and ``VehPower`` swapped where both values stay valid. Data drift.
- ``portfolio_shift``: weighted sampling toward young drivers in dense areas. Data drift and
  prediction drift.
- ``concept_drift``: the same policies with extra claims. Calibration and live deviance.

Every file keeps ``IDpol``, ``ClaimNb``, and ``Exposure``. The two random scenarios use the
top-level data ``seed``, each with its own generator, so changing one scenario's knobs in
``params.yaml`` (section ``drift``) never changes another scenario's file.

Checks run before anything is written. The three scenarios that must stay valid pass the clean
stage, or ``SchemaError`` is raised. The minimum effect sizes of the contract are measured, or
``EffectSizeError`` is raised. ``claimfreq`` turns both into exit code 10. The
``schema_break`` and ``out_of_bounds`` frames are broken on purpose and never validated.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from claimfreq.contracts import paths, schema
from claimfreq.contracts.params import Params, load_params
from claimfreq.contracts.paths import DriftScenario

# Exact changes of the two broken scenarios (docs/contracts.md, Drift files).
SCHEMA_BREAK_RENAME = {"DrivAge": "DriverAge"}
OUT_OF_BOUNDS_VALUES = {"DrivAge": 250, "BonusMalus": -1}

# Scenarios whose files must pass the clean stage.
VALID_SCENARIOS = (
    DriftScenario.COLUMN_SWAP,
    DriftScenario.PORTFOLIO_SHIFT,
    DriftScenario.CONCEPT_DRIFT,
)

# Minimum effect sizes, so each scenario trips its monitoring checks (docs/contracts.md).
MIN_EFFECTS: dict[str, float] = {
    "age_drop_years": 5.0,  # portfolio_shift: mean DrivAge at least 5 years lower
    "density_median_increase": 0.50,  # portfolio_shift: median Density at least 50% higher
    "claim_rate_increase": 0.40,  # concept_drift: claim rate at least 40% higher
}


class EffectSizeError(ValueError):
    """Raised when a drift scenario is weaker than the contract's minimum effect size."""


def schema_break(test: pd.DataFrame) -> pd.DataFrame:
    """Rename ``DrivAge`` to ``DriverAge``; the column keeps its position and values."""
    return test.rename(columns=SCHEMA_BREAK_RENAME)


def out_of_bounds(test: pd.DataFrame) -> pd.DataFrame:
    """Set ``DrivAge`` to 250 and ``BonusMalus`` to -1 on every row."""
    return test.assign(**OUT_OF_BOUNDS_VALUES)


def column_swap(test: pd.DataFrame) -> pd.DataFrame:
    """Swap ``VehAge`` and ``VehPower`` on the rows where both values stay valid.

    That is where ``VehAge`` lies within ``VehPower``'s bounds (4-15). ``VehPower`` always fits
    ``VehAge``'s bounds (0-100). Every other row is unchanged, and every row is kept.
    """
    low, high = schema.NUMERIC_BOUNDS["VehPower"]
    swap = test["VehAge"].between(low, high)
    swapped = test.copy()
    swapped.loc[swap, ["VehAge", "VehPower"]] = test.loc[swap, ["VehPower", "VehAge"]].to_numpy()
    return swapped


def _standardize(values: np.ndarray) -> np.ndarray:
    """Return ``values`` in standard deviations from their mean (all zeros if constant)."""
    std = values.std()
    if std == 0:
        return np.zeros_like(values)
    return (values - values.mean()) / std


def portfolio_shift(test: pd.DataFrame, params: Params, rng: np.random.Generator) -> pd.DataFrame:
    """Sample policies without replacement, favouring young drivers in dense areas.

    Each policy's weight is ``exp(-age_weight * z(DrivAge) + density_weight * z(log Density))``
    (``params.drift.portfolio_shift``). Rows keep their order from the test split, which
    ``split`` shuffles with the seed, and every ``IDpol`` appears at most once.
    """
    knobs = params.drift.portfolio_shift
    age = test["DrivAge"].to_numpy(dtype=float)
    log_density = np.log(test["Density"].to_numpy(dtype=float))
    log_weight = -knobs.age_weight * _standardize(age) + knobs.density_weight * _standardize(
        log_density
    )
    # Subtracting the largest log-weight avoids overflow and leaves the probabilities unchanged.
    weights = np.exp(log_weight - log_weight.max())
    n_sample = round(knobs.sample_fraction * len(test))
    rows = rng.choice(len(test), size=n_sample, replace=False, p=weights / weights.sum())
    return test.iloc[np.sort(rows)].reset_index(drop=True)


def concept_drift(test: pd.DataFrame, params: Params, rng: np.random.Generator) -> pd.DataFrame:
    """Add random claims so the claim rate rises by about ``claim_multiplier``.

    Each policy gets ``Poisson((claim_multiplier - 1) * portfolio rate * Exposure)`` extra
    claims, so a policy insured longer is more likely to gain one, as with a real rise in
    accident rates. The result is clipped to ``clean.claim_nb_cap``. The rating factors and
    ``Exposure`` are unchanged.
    """
    knobs = params.drift.concept_drift
    exposure = test[schema.EXPOSURE].to_numpy(dtype=float)
    claims = test[schema.CLAIM_NB].to_numpy()
    rate = claims.sum() / exposure.sum()
    extra = rng.poisson((knobs.claim_multiplier - 1) * rate * exposure)
    capped = np.minimum(claims + extra, params.clean.claim_nb_cap)
    return test.assign(**{schema.CLAIM_NB: capped.astype("int64")})


def build_scenarios(test: pd.DataFrame, params: Params) -> dict[DriftScenario, pd.DataFrame]:
    """Return every drift scenario's frame, built from the same test split."""
    return {
        DriftScenario.SCHEMA_BREAK: schema_break(test),
        DriftScenario.OUT_OF_BOUNDS: out_of_bounds(test),
        DriftScenario.COLUMN_SWAP: column_swap(test),
        # Each random scenario gets its own generator from the data seed.
        DriftScenario.PORTFOLIO_SHIFT: portfolio_shift(
            test, params, np.random.default_rng(params.seed)
        ),
        DriftScenario.CONCEPT_DRIFT: concept_drift(
            test, params, np.random.default_rng(params.seed)
        ),
    }


def _claim_rate(df: pd.DataFrame) -> float:
    """Return claims per exposure-year."""
    return df[schema.CLAIM_NB].sum() / df[schema.EXPOSURE].sum()


def check_effects(
    test: pd.DataFrame, scenarios: dict[DriftScenario, pd.DataFrame]
) -> dict[str, float]:
    """Measure each random scenario's effect and raise ``EffectSizeError`` below the minimums.

    Returns:
        The measured effects, keyed like ``MIN_EFFECTS``. The error lists every effect that falls
        short, with its value and the minimum.
    """
    shifted = scenarios[DriftScenario.PORTFOLIO_SHIFT]
    concept = scenarios[DriftScenario.CONCEPT_DRIFT]
    effects = {
        "age_drop_years": float(test["DrivAge"].mean() - shifted["DrivAge"].mean()),
        "density_median_increase": float(
            shifted["Density"].median() / test["Density"].median() - 1
        ),
        "claim_rate_increase": float(_claim_rate(concept) / _claim_rate(test) - 1),
    }
    problems = [
        f"{name}: {effects[name]:.3f}, minimum {minimum}"
        for name, minimum in MIN_EFFECTS.items()
        # Written as "not >=" so a NaN effect (e.g. an empty portfolio_shift sample) also fails.
        if not effects[name] >= minimum
    ]
    if problems:
        lines = "\n".join(f"- {problem}" for problem in problems)
        raise EffectSizeError(
            "drift scenarios are weaker than the contract's minimum effect sizes; "
            f"strengthen the drift: section of params.yaml:\n{lines}"
        )
    return effects


def make_drift(params: Params | None = None) -> tuple[dict[Path, int], dict[str, float]]:
    """Read the test split, build and check every scenario, then write all five files.

    Args:
        params: Loaded ``params.yaml``; read from the repo root when omitted.

    Returns:
        The row count of each written file, keyed by its path, and the measured effects.

    Raises:
        SchemaError: if the test split, or a scenario that must stay valid, breaks the clean
            stage (exit 10).
        EffectSizeError: if a scenario is weaker than the contract's minimum (exit 10).
    """
    if params is None:
        params = load_params()

    test = pd.read_parquet(paths.data_path(paths.TEST))
    schema.validate_frame(test, "clean", params)
    test = schema.coerce_frame(test)

    scenarios = build_scenarios(test, params)
    for scenario in VALID_SCENARIOS:
        schema.validate_frame(scenarios[scenario], "clean", params)
    effects = check_effects(test, scenarios)

    written: dict[Path, int] = {}
    for scenario, frame in scenarios.items():
        if scenario in VALID_SCENARIOS:
            frame = schema.coerce_frame(frame)
        out = paths.drift_path(scenario)
        out.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(out, index=False)
        written[out] = len(frame)
    return written, effects
