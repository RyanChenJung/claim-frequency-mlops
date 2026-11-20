"""Tests for ``claimfreq clean``: target capping, caps from params, and the clean-stage file.

Every test builds its raw file from ``make_policies`` under a temporary ``DATA_DIR``, so
nothing is read from or written into the repository's ``data/``.
"""

import pandas as pd
import pytest

from claimfreq.clean import clip_targets
from claimfreq.cli import EXIT_CHECK_FAILED, EXIT_OK, main
from claimfreq.contracts import paths, schema
from claimfreq.contracts.params import CleanParams, Params
from claimfreq.fakes import make_policies

N_ROWS = 200
# Rows given values above the default caps (4 claims, 1.0 year), as the real raw data has.
TAIL_ROWS = [0, 1, 2, 3]


def make_raw_with_tails() -> pd.DataFrame:
    """Return raw-stage policies with two ``ClaimNb`` and two ``Exposure`` values above the caps."""
    df = make_policies(N_ROWS)
    df.loc[0, schema.CLAIM_NB] = 7
    df.loc[1, schema.CLAIM_NB] = 16  # the largest ClaimNb in the real raw data
    df.loc[2, schema.EXPOSURE] = 1.5
    df.loc[3, schema.EXPOSURE] = 2.01  # the largest Exposure in the real raw data
    return df


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    """Write a raw file with long tails under a temporary ``DATA_DIR`` and return the folder."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    raw_path = tmp_path / paths.RAW
    raw_path.parent.mkdir(parents=True)
    make_raw_with_tails().to_parquet(raw_path, index=False)
    return tmp_path


def test_clip_targets_caps_only_the_values_above_the_caps():
    raw = make_raw_with_tails()
    cleaned, n_clipped = clip_targets(raw, Params())

    assert n_clipped == {schema.CLAIM_NB: 2, schema.EXPOSURE: 2}
    assert cleaned.loc[[0, 1], schema.CLAIM_NB].tolist() == [4, 4]
    assert cleaned.loc[[2, 3], schema.EXPOSURE].tolist() == [1.0, 1.0]
    # Every other row is identical, and the dtypes are kept.
    others = ~raw.index.isin(TAIL_ROWS)
    pd.testing.assert_frame_equal(cleaned[others], raw[others])
    assert cleaned[schema.CLAIM_NB].dtype == "int64"
    # The input frame itself is not modified.
    assert raw.loc[1, schema.CLAIM_NB] == 16


def test_clip_targets_uses_the_caps_from_params():
    raw = make_raw_with_tails()
    params = Params(clean=CleanParams(claim_nb_cap=2, exposure_cap=0.5))
    cleaned, n_clipped = clip_targets(raw, params)

    assert cleaned[schema.CLAIM_NB].max() <= 2
    assert cleaned[schema.EXPOSURE].max() <= 0.5
    assert n_clipped == {
        schema.CLAIM_NB: int((raw[schema.CLAIM_NB] > 2).sum()),
        schema.EXPOSURE: int((raw[schema.EXPOSURE] > 0.5).sum()),
    }
    # The clean-stage check passes with the same params.
    schema.validate_frame(cleaned, "clean", params)


def test_cli_clean_writes_a_valid_clean_file(data_dir, capsys):
    assert main(["clean"]) == EXIT_OK
    assert "clipped ClaimNb on 2 rows, Exposure on 2 rows" in capsys.readouterr().out

    written = pd.read_parquet(data_dir / paths.PROCESSED)
    assert len(written) == N_ROWS
    assert list(written.columns) == schema.FRAME_COLUMNS
    assert {column: str(written[column].dtype) for column in written} == schema.DTYPES
    schema.validate_frame(written, "clean")


def test_cli_clean_exits_10_on_a_bad_raw_file_and_writes_nothing(data_dir, capsys):
    raw_path = data_dir / paths.RAW
    raw = pd.read_parquet(raw_path)
    raw.loc[5, "DrivAge"] = 250  # outside the hard bounds, even in the raw stage
    raw.to_parquet(raw_path, index=False)

    assert main(["clean"]) == EXIT_CHECK_FAILED
    err = capsys.readouterr().err
    assert "raw frame" in err
    assert "DrivAge" in err
    assert not (data_dir / paths.PROCESSED).exists()


def test_cli_clean_accepts_whole_number_floats_in_the_raw_file(data_dir):
    raw_path = data_dir / paths.RAW
    raw = pd.read_parquet(raw_path)
    # Valid in the raw stage, as OpenML ships IDpol and ClaimNb.
    raw.astype({schema.ID_COLUMN: "float64", schema.CLAIM_NB: "float64"}).to_parquet(
        raw_path, index=False
    )

    assert main(["clean"]) == EXIT_OK
    written = pd.read_parquet(data_dir / paths.PROCESSED)
    assert written[schema.CLAIM_NB].dtype == "int64"
    schema.validate_frame(written, "clean")
