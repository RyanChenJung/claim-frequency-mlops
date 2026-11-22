"""Tests for ``claimfreq make-drift``: each scenario's shape, schema status, and effect size.

The test split is a ``make_policies`` fake; the CLI tests use a temporary ``DATA_DIR``, so
nothing is read from or written into the repository's ``data/``.
"""

import pandas as pd
import pytest

from claimfreq import make_drift as make_drift_module
from claimfreq.cli import EXIT_CHECK_FAILED, EXIT_OK, main
from claimfreq.contracts import paths, schema
from claimfreq.contracts.params import (
    ConceptDriftParams,
    DriftParams,
    Params,
    PortfolioShiftParams,
)
from claimfreq.contracts.paths import DriftScenario
from claimfreq.fakes import make_policies
from claimfreq.make_drift import MIN_EFFECTS, EffectSizeError, build_scenarios, check_effects

N_ROWS = 10_000
ID = schema.ID_COLUMN


def checks(df: pd.DataFrame) -> set[tuple[str, str]]:
    """Return the (check, column) pairs of every clean-stage violation of ``df``."""
    return {(v.check, v.column) for v in schema.find_violations(df, "clean")}


@pytest.fixture
def holdout() -> pd.DataFrame:
    """Return a fake clean-stage test split (the input of make-drift), with test-range ids.

    It is shuffled, like the real test split, so the tests cannot depend on IDpol order.
    """
    policies = make_policies(N_ROWS, seed=3, id_start=900_000)
    return policies.sample(frac=1, random_state=0).reset_index(drop=True)


@pytest.fixture
def scenarios(holdout) -> dict[DriftScenario, pd.DataFrame]:
    """Return every scenario built from the fake test split with the default knobs."""
    return build_scenarios(holdout, Params())


@pytest.fixture
def data_dir(tmp_path, monkeypatch, holdout):
    """Write the fake test split under a temporary ``DATA_DIR`` and return the folder."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    test_path = tmp_path / paths.TEST
    test_path.parent.mkdir(parents=True)
    holdout.to_parquet(test_path, index=False)
    return tmp_path


def test_minimum_effects_are_the_contract_numbers():
    # Pinned here, so lowering a minimum in make_drift.py fails a test (docs/contracts.md).
    assert MIN_EFFECTS == {
        "age_drop_years": 5.0,
        "density_median_increase": 0.50,
        "claim_rate_increase": 0.40,
    }


def test_schema_break_renames_only_drivage(holdout, scenarios):
    broken = scenarios[DriftScenario.SCHEMA_BREAK]
    expected = ["DriverAge" if c == "DrivAge" else c for c in schema.FRAME_COLUMNS]
    assert list(broken.columns) == expected
    pd.testing.assert_frame_equal(broken.rename(columns={"DriverAge": "DrivAge"}), holdout)
    assert checks(broken) == {("missing_column", "DrivAge"), ("extra_column", "DriverAge")}


def test_out_of_bounds_sets_both_columns_on_every_row(holdout, scenarios):
    broken = scenarios[DriftScenario.OUT_OF_BOUNDS]
    assert (broken["DrivAge"] == 250).all()
    assert (broken["BonusMalus"] == -1).all()
    others = [c for c in schema.FRAME_COLUMNS if c not in ("DrivAge", "BonusMalus")]
    pd.testing.assert_frame_equal(broken[others], holdout[others])
    assert checks(broken) == {("out_of_range", "DrivAge"), ("out_of_range", "BonusMalus")}


def test_column_swap_follows_the_contract_rule(holdout, scenarios):
    swapped = scenarios[DriftScenario.COLUMN_SWAP]
    low, high = schema.NUMERIC_BOUNDS["VehPower"]
    swap = holdout["VehAge"].between(low, high)
    assert swap.any() and (~swap).any()  # the fake has rows of both kinds
    assert (swapped.loc[swap, "VehAge"] == holdout.loc[swap, "VehPower"]).all()
    assert (swapped.loc[swap, "VehPower"] == holdout.loc[swap, "VehAge"]).all()
    pd.testing.assert_frame_equal(swapped[~swap], holdout[~swap])
    assert len(swapped) == len(holdout)
    assert checks(swapped) == set()


def test_portfolio_shift_samples_real_rows_and_meets_its_effect_sizes(holdout, scenarios):
    shifted = scenarios[DriftScenario.PORTFOLIO_SHIFT]
    assert len(shifted) == round(Params().drift.portfolio_shift.sample_fraction * N_ROWS)
    assert shifted[ID].is_unique
    # Sampled rows keep their relative order from the test split.
    position = pd.Series(holdout.index, index=holdout[ID])
    assert position.loc[shifted[ID]].is_monotonic_increasing
    # Every sampled row is an unchanged row of the test split.
    expected = holdout.set_index(ID).loc[shifted[ID]].reset_index()[schema.FRAME_COLUMNS]
    pd.testing.assert_frame_equal(shifted, expected)
    assert checks(shifted) == set()

    effects = check_effects(holdout, scenarios)
    assert effects["age_drop_years"] >= MIN_EFFECTS["age_drop_years"]
    assert effects["density_median_increase"] >= MIN_EFFECTS["density_median_increase"]


def test_concept_drift_only_adds_claims_and_meets_its_effect_size(holdout, scenarios):
    concept = scenarios[DriftScenario.CONCEPT_DRIFT]
    claims = schema.CLAIM_NB
    pd.testing.assert_frame_equal(concept.drop(columns=claims), holdout.drop(columns=claims))
    assert (concept[claims] >= holdout[claims]).all()
    assert concept[claims].max() <= Params().clean.claim_nb_cap
    assert checks(concept) == set()

    effects = check_effects(holdout, scenarios)
    assert effects["claim_rate_increase"] >= MIN_EFFECTS["claim_rate_increase"]


def test_effects_are_measured_as_the_contract_states(holdout, scenarios):
    # An independent measurement, so check_effects is not its own oracle.
    def rate(df):
        return df["ClaimNb"].sum() / df["Exposure"].sum()

    shifted = scenarios[DriftScenario.PORTFOLIO_SHIFT]
    concept = scenarios[DriftScenario.CONCEPT_DRIFT]
    expected = {
        "age_drop_years": holdout["DrivAge"].mean() - shifted["DrivAge"].mean(),
        "density_median_increase": shifted["Density"].median() / holdout["Density"].median() - 1,
        "claim_rate_increase": rate(concept) / rate(holdout) - 1,
    }
    assert check_effects(holdout, scenarios) == pytest.approx(expected)


def test_weak_portfolio_knobs_raise(holdout):
    weak = Params(
        drift=DriftParams(portfolio_shift=PortfolioShiftParams(age_weight=0.0, density_weight=0.0))
    )
    with pytest.raises(EffectSizeError) as error:
        check_effects(holdout, build_scenarios(holdout, weak))
    assert "age_drop_years" in str(error.value)
    assert "density_median_increase" in str(error.value)


def test_sample_fraction_sets_the_row_count(holdout):
    params = Params(drift=DriftParams(portfolio_shift=PortfolioShiftParams(sample_fraction=0.3)))
    assert len(build_scenarios(holdout, params)[DriftScenario.PORTFOLIO_SHIFT]) == 3_000


def test_the_same_seed_gives_the_same_files(holdout, scenarios):
    again = build_scenarios(holdout, Params())
    for scenario, frame in scenarios.items():
        pd.testing.assert_frame_equal(again[scenario], frame)
    other = build_scenarios(holdout, Params(seed=7))
    shift = DriftScenario.PORTFOLIO_SHIFT
    assert set(other[shift][ID]) != set(scenarios[shift][ID])


def test_cli_make_drift_writes_all_five_files(data_dir, capsys):
    assert main(["make-drift"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "make-drift: effects age_drop_years=" in out
    for scenario in DriftScenario:
        path = data_dir / paths.DRIFT_DIR / f"{scenario}.parquet"
        written = pd.read_parquet(path)
        assert f"wrote {len(written)} policies to {path}" in out
        if scenario in make_drift_module.VALID_SCENARIOS:
            assert {c: str(written[c].dtype) for c in written} == schema.DTYPES
            schema.validate_frame(written, "clean")


def test_cli_make_drift_exits_10_on_a_weak_knob_and_writes_nothing(data_dir, monkeypatch, capsys):
    weak = Params(drift=DriftParams(concept_drift=ConceptDriftParams(claim_multiplier=1.05)))
    monkeypatch.setattr(make_drift_module, "load_params", lambda: weak)

    assert main(["make-drift"]) == EXIT_CHECK_FAILED
    err = capsys.readouterr().err
    assert "claim_rate_increase" in err
    assert "age_drop_years" not in err  # the other knobs are still strong enough
    assert "Traceback" not in err
    assert not (data_dir / paths.DRIFT_DIR).exists()
