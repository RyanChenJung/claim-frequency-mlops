"""Tests for the EDA script (scripts/eda.py): hand-computed numbers, then a run on fakes.

The run tests write fake raw, clean, and split files under a temporary ``DATA_DIR``, so nothing
is read from or written into the repository's ``data/`` or ``docs/eda/``.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from claimfreq.contracts import paths, schema
from claimfreq.fakes import make_policies

# scripts/ is not a package; put it on sys.path, as the DAG tests do for dags/.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import eda  # noqa: E402

OUTPUT_FILES = {
    "claim_count.png",
    "exposure.png",
    "freq_drivage.png",
    "freq_bonusmalus.png",
    "freq_vehage.png",
    "freq_area.png",
    "summary.json",
}


def frame(**columns) -> pd.DataFrame:
    """Return a tiny frame with only the given columns, for the hand-computed tests."""
    return pd.DataFrame(columns)


# --- Bands ---------------------------------------------------------------------------------


def test_band_labels_give_single_values_ranges_and_an_open_last_band():
    assert eda.band_labels([18, 21, 25]) == ["18-20", "21-24", "25+"]
    assert eda.band_labels([0, 1, 2, 3, 6]) == ["0", "1", "2", "3-5", "6+"]
    assert eda.band_labels([50, 51, 60]) == ["50", "51-59", "60+"]


def test_assign_bands_puts_each_edge_in_the_band_it_starts():
    values = pd.Series([18, 20, 21, 24, 25, 99], name="DrivAge")
    bands = eda.assign_bands(values, [18, 21, 25])
    assert bands.tolist() == ["18-20", "18-20", "21-24", "21-24", "25+", "25+"]
    assert bands.cat.categories.tolist() == ["18-20", "21-24", "25+"]


def test_assign_bands_rejects_a_value_below_the_first_edge():
    with pytest.raises(ValueError, match="below the first band edge"):
        eda.assign_bands(pd.Series([17, 30], name="DrivAge"), [18, 25])


def test_every_plotted_factor_has_bands_or_a_category_order():
    for column in eda.FREQUENCY_FACTORS:
        assert column in eda.BAND_EDGES or column in eda.CATEGORY_ORDER
    assert eda.CATEGORY_ORDER["Area"] == ["A", "B", "C", "D", "E", "F"]


# --- Frequency -----------------------------------------------------------------------------


def test_frequency_is_a_ratio_of_sums_not_a_mean_of_ratios():
    df = frame(
        DrivAge=[18, 20, 30, 30],
        ClaimNb=[1, 0, 2, 0],
        Exposure=[0.25, 0.75, 1.0, 0.25],
    )
    table = eda.frequency_table(df, eda.assign_bands(df["DrivAge"], [18, 25]))
    # 18-24: 1 claim over 1.0 year = 1.0. The mean of ratios would be (4 + 0) / 2 = 2.0.
    # 25+: 2 claims over 1.25 years = 1.6.
    assert table["policies"].tolist() == [2, 2]
    assert table["claims"].tolist() == [1, 2]
    assert table["exposure"].tolist() == pytest.approx([1.0, 1.25])
    assert table["frequency"].tolist() == pytest.approx([1.0, 1.6])
    assert eda.portfolio_frequency(df) == pytest.approx(3 / 2.25)


def test_an_empty_band_keeps_its_row_with_no_frequency():
    df = frame(DrivAge=[18, 30], ClaimNb=[1, 0], Exposure=[0.5, 1.0])
    table = eda.frequency_table(df, eda.assign_bands(df["DrivAge"], [18, 21, 25]))
    assert table.index.tolist() == ["18-20", "21-24", "25+"]
    assert table.loc["21-24", "policies"] == 0
    assert np.isnan(table.loc["21-24", "frequency"])
    assert eda._table_records(table)[1]["frequency"] is None


def test_area_bands_follow_the_category_order():
    df = frame(Area=["F", "A", "C"], ClaimNb=[0, 1, 0], Exposure=[1.0, 0.5, 1.0])
    table = eda.frequency_table(df, eda.factor_bands(df, "Area"))
    assert table.index.tolist() == ["A", "B", "C", "D", "E", "F"]
    assert table.loc["A", "frequency"] == pytest.approx(2.0)


# --- Claim counts --------------------------------------------------------------------------


def test_claim_count_shares_leave_out_counts_that_never_occur():
    df = frame(ClaimNb=[0, 2, 2, 0, 2])
    shares = eda.claim_count_shares(df)
    # In value order, not by frequency: 2 is the more common value here.
    assert shares.index.tolist() == [0, 2]
    assert shares.tolist() == pytest.approx([0.4, 0.6])


# --- Duplicate tuples ----------------------------------------------------------------------


def test_cross_split_share_compares_the_nine_factors_only():
    train = make_policies(3, seed=0)
    same = train.iloc[[0]].copy()
    other_targets = same.assign(IDpol=999, ClaimNb=3, Exposure=0.1)  # same 9 factors
    one_factor_off = same.assign(Density=same["Density"] + 1)
    other = pd.concat([same, other_targets, one_factor_off, train.iloc[[1]]])
    # Rows 1, 2, and 4 match a train tuple; row 3 differs in Density only.
    assert eda.cross_split_duplicate_share(train, other) == pytest.approx(0.75)
    assert eda.cross_split_duplicate_share(train, make_policies(5, seed=1)) == 0.0


def test_within_share_counts_every_copy_of_a_repeated_tuple():
    policies = make_policies(3, seed=0)
    df = pd.concat([policies.iloc[[0, 0]], policies.iloc[[1, 2]]])
    assert eda.within_duplicate_share(df) == pytest.approx(0.5)


# --- Capped rows ---------------------------------------------------------------------------


def test_changed_rows_matches_on_idpol_and_counts_only_the_given_rows():
    raw = frame(IDpol=[1, 2, 3], Exposure=[1.5, 0.5, 2.0])
    clean = frame(IDpol=[3, 1, 2], Exposure=[1.0, 1.0, 0.5])  # another order
    assert eda.changed_rows(raw, clean, "Exposure") == 2
    assert eda.changed_rows(raw, clean[clean["IDpol"] == 1], "Exposure") == 1
    assert eda.changed_rows(raw, clean[clean["IDpol"] == 2], "Exposure") == 0


# --- Runs on fake pipeline files -----------------------------------------------------------


@pytest.fixture
def data_dir(tmp_path, monkeypatch) -> Path:
    """Write fake raw, clean, and split files under a temporary ``DATA_DIR``.

    Rows 0 and 900 have raw exposures above 1 year, and rows 1 and 901 more than 4 raw claims.
    Rows 0 and 1 are in train and rows 900 and 901 in test, so train holds one capped row of each
    kind and the whole dataset two. The splits are disjoint slices with distinct ids.
    """
    clean = make_policies(1_000, seed=0)
    raw = clean.copy()
    raw.loc[0, schema.EXPOSURE] = 1.7
    raw.loc[900, schema.EXPOSURE] = 1.5
    raw.loc[1, schema.CLAIM_NB] = 6
    raw.loc[901, schema.CLAIM_NB] = 5
    clean.loc[[0, 900], schema.EXPOSURE] = 1.0
    clean.loc[[1, 901], schema.CLAIM_NB] = schema.DEFAULT_CLAIM_NB_CAP
    # Validation repeats 50 train tuples under new ids, so its duplicate share is at least 0.5.
    validation = pd.concat(
        [clean.iloc[700:750], clean.iloc[:50].assign(IDpol=np.arange(2_001, 2_051))]
    )
    files = {
        paths.RAW: raw,
        paths.PROCESSED: clean,
        paths.TRAIN: clean.iloc[:700],
        paths.VALIDATION: validation,
        paths.TEST: clean.iloc[800:],
    }
    root = tmp_path / "data"
    for relative, df in files.items():
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(root / relative, index=False)
    monkeypatch.setenv("DATA_DIR", str(root))
    return root


def test_run_writes_six_plots_and_a_summary(data_dir, tmp_path):
    out_dir = tmp_path / "eda"
    summary = eda.run(out_dir)

    assert {p.name for p in out_dir.iterdir()} == OUTPUT_FILES
    assert json.loads((out_dir / "summary.json").read_text()) == summary
    assert set(summary) == {"dataset", "train", "duplicates"}

    dataset, train, duplicates = summary["dataset"], summary["train"], summary["duplicates"]
    assert dataset["rows"] == 1_000
    assert dataset["claim_nb_max_before_cap"] == 6
    assert dataset["exposure_max_before_cap"] == pytest.approx(1.7)
    assert dataset["claims_before_cap"] - dataset["claims_after_cap"] == (6 - 4) + (5 - 4)
    assert dataset["claim_nb_capped_rows"] == dataset["exposure_capped_rows"] == 2
    assert train["rows"] == 700
    assert train["claim_nb_capped_rows"] == train["exposure_capped_rows"] == 1
    assert sum(train["claim_counts"].values()) == train["rows"]
    assert set(train["frequency_by"]) == set(eda.FREQUENCY_FACTORS)
    assert train["md5"] == eda.file_md5(data_dir / paths.TRAIN)
    assert duplicates["validation_rows_matching_train"] >= 0.5
    assert 0.0 <= duplicates["test_rows_matching_train"] < 0.5


def test_rerun_writes_byte_identical_files(data_dir, tmp_path):
    eda.run(tmp_path / "first")
    eda.run(tmp_path / "second")
    for name in OUTPUT_FILES:
        first = (tmp_path / "first" / name).read_bytes()
        assert first == (tmp_path / "second" / name).read_bytes()
        # No matplotlib version in the PNG metadata, so an upgrade alone changes nothing.
        assert b"Software" not in first


def test_main_prints_the_duplicate_shares(data_dir, tmp_path, capsys):
    assert eda.main(["--out-dir", str(tmp_path / "eda")]) == 0
    assert "validation_rows_matching_train" in capsys.readouterr().out
