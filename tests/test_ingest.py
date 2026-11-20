"""Tests for ``claimfreq ingest``: column selection, quote stripping, checks, and the raw file.

No test touches the network: ``download`` is replaced with a small OpenML-like fake, and
``DATA_DIR`` points to ``tmp_path``, so nothing is written into the repository.
"""

from types import SimpleNamespace

import pandas as pd
import pytest

from claimfreq import ingest as ingest_module
from claimfreq.cli import EXIT_CHECK_FAILED, EXIT_OK, main
from claimfreq.contracts import paths, schema
from claimfreq.fakes import make_policies
from claimfreq.ingest import CountCheckError, check_counts, prepare

N_ROWS = 50


def make_raw(n: int = N_ROWS) -> pd.DataFrame:
    """Return ``n`` policies that look like the OpenML 41214 download.

    It has the quirks the real download has: float ``IDpol`` and ``ClaimNb``, ``VehGas`` in
    literal quotes, ``category`` dtypes, a shuffled column order, and one extra column.
    """
    df = make_policies(n)
    df = df.astype({schema.ID_COLUMN: "float64", schema.CLAIM_NB: "float64"})
    df["VehGas"] = "'" + df["VehGas"] + "'"
    df = df.astype({column: "category" for column in schema.CATEGORICAL_FEATURES})
    df["extra_column"] = 0.0
    return df[list(reversed(df.columns))]


def counts_of(df: pd.DataFrame) -> dict[str, int]:
    """Return the frame's own totals, in the shape of ``ingest.EXPECTED_COUNTS``."""
    return {
        "rows": len(df),
        "claims": int(df[schema.CLAIM_NB].sum()),
        "regions": int(df["Region"].nunique()),
    }


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    """Point ``DATA_DIR`` at a temporary folder and return the raw file's path in it."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    return tmp_path / paths.RAW


def test_prepare_strips_quotes_selects_columns_and_passes_raw_schema():
    df = prepare(make_raw())
    assert list(df.columns) == schema.FRAME_COLUMNS
    assert set(df["VehGas"]) <= {"Regular", "Diesel"}
    # The frame as read (float ids, category columns) is valid in the raw stage.
    schema.validate_frame(df, "raw")


def test_cli_ingest_writes_the_canonical_raw_parquet(data_dir, monkeypatch, capsys):
    raw = make_raw()
    monkeypatch.setattr(ingest_module, "download", lambda: raw)
    monkeypatch.setattr(ingest_module, "EXPECTED_COUNTS", counts_of(raw))

    assert main(["ingest"]) == EXIT_OK
    assert f"wrote {N_ROWS} policies" in capsys.readouterr().out
    written = pd.read_parquet(data_dir)
    assert list(written.columns) == schema.FRAME_COLUMNS
    assert {column: str(written[column].dtype) for column in written} == schema.DTYPES
    # Once the quirks are undone, the file holds exactly the policies the fake started from.
    pd.testing.assert_frame_equal(written, make_policies(N_ROWS), check_dtype=False)


def test_cli_ingest_exits_10_on_wrong_totals_and_writes_nothing(data_dir, monkeypatch, capsys):
    # EXPECTED_COUNTS keeps the real totals, so the 50-row fake fails the count check.
    monkeypatch.setattr(ingest_module, "download", make_raw)

    assert main(["ingest"]) == EXIT_CHECK_FAILED
    err = capsys.readouterr().err
    assert f"rows: expected 678,013, got {N_ROWS}" in err
    assert "Traceback" not in err
    assert not data_dir.exists()


def test_cli_ingest_exits_10_on_a_schema_violation_and_writes_nothing(
    data_dir, monkeypatch, capsys
):
    raw = make_raw()
    raw["Region"] = raw["Region"].cat.add_categories("R99")
    raw.loc[raw.index[0], "Region"] = "R99"
    monkeypatch.setattr(ingest_module, "download", lambda: raw)
    # The totals match, so the schema check is the only one that can fail.
    monkeypatch.setattr(ingest_module, "EXPECTED_COUNTS", counts_of(raw))

    assert main(["ingest"]) == EXIT_CHECK_FAILED
    err = capsys.readouterr().err
    assert "Region: 1 unknown categor" in err
    assert "Traceback" not in err
    assert not data_dir.exists()


def test_check_counts_passes_on_matching_totals_and_names_each_mismatch():
    df = prepare(make_raw())
    expected = counts_of(df)
    check_counts(df, expected)  # matching totals: no error

    with pytest.raises(CountCheckError) as excinfo:
        check_counts(df, {**expected, "claims": expected["claims"] + 1})
    message = str(excinfo.value)
    assert f"- claims: expected {expected['claims'] + 1:,}, got {expected['claims']:,}" in message
    assert "- rows" not in message
    assert "- regions" not in message


def test_download_asks_openml_for_dataset_41214(monkeypatch):
    raw = make_raw()
    calls = []

    def fake_fetch_openml(**kwargs):
        """Stand-in for sklearn's fetch_openml: record the arguments, return a fake bunch."""
        calls.append(kwargs)
        return SimpleNamespace(frame=raw)

    monkeypatch.setattr(ingest_module, "fetch_openml", fake_fetch_openml)
    assert ingest_module.download() is raw
    # The dataset id and parsing options must not drift (ADR-0001).
    assert calls == [{"data_id": 41214, "as_frame": True, "parser": "pandas"}]
