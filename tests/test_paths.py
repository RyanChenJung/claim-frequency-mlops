"""Tests for the paths contract: the drift-file rules that keep valid scenarios valid."""

from claimfreq.contracts.schema import NUMERIC_BOUNDS, validate_frame
from claimfreq.fakes import make_policies


def test_column_swap_rule_keeps_the_clean_schema():
    df = make_policies(5000)
    # The rule's premise: every VehPower value is a valid VehAge, but not the other way round.
    age_low, age_high = NUMERIC_BOUNDS["VehAge"]
    power_low, power_high = NUMERIC_BOUNDS["VehPower"]
    assert age_low <= power_low and power_high <= age_high
    # Swap only where VehAge is also a valid VehPower; every other row stays as it is.
    swap = df["VehAge"].between(power_low, power_high)
    swapped = df.copy()
    swapped.loc[swap, ["VehAge", "VehPower"]] = df.loc[swap, ["VehPower", "VehAge"]].to_numpy()
    validate_frame(swapped, "clean")
    assert len(swapped) == len(df)
    assert (swapped["VehAge"] != df["VehAge"]).any()
