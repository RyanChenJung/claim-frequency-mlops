"""Tests for ``claimfreq split``: exact sizes, disjoint ids, seeding, stratification, and files.

Every test splits ``make_policies`` data; the CLI tests use a temporary ``DATA_DIR``, so
nothing is read from or written into the repository's ``data/``.
"""

import pandas as pd
import pytest

from claimfreq.cli import EXIT_CHECK_FAILED, EXIT_OK, main
from claimfreq.contracts import paths, schema
from claimfreq.contracts.params import Params, SplitParams
from claimfreq.fakes import make_policies
from claimfreq.split import SPLIT_PATHS, split_policies

N_ROWS = 1_000
NAMES = ["train", "validation", "test"]


def ids(df: pd.DataFrame) -> set[int]:
    """Return the set of ``IDpol`` values of a frame."""
    return set(df[schema.ID_COLUMN])


@pytest.fixture
def policies() -> pd.DataFrame:
    """Return clean-stage fake policies, the input of ``split``."""
    return make_policies(N_ROWS)


@pytest.fixture
def data_dir(tmp_path, monkeypatch, policies):
    """Write the clean file under a temporary ``DATA_DIR`` and return the folder."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    processed = tmp_path / paths.PROCESSED
    processed.parent.mkdir(parents=True)
    policies.to_parquet(processed, index=False)
    return tmp_path


@pytest.mark.parametrize(
    ("test_fraction", "validation_fraction", "expected"),
    [
        (0.2, 0.1, {"train": 700, "validation": 100, "test": 200}),  # params.yaml values
        (0.3, 0.2, {"train": 500, "validation": 200, "test": 300}),
    ],
)
def test_split_sizes_are_exact_fractions_of_the_full_dataset(
    policies, test_fraction, validation_fraction, expected
):
    params = Params(
        split=SplitParams(test_fraction=test_fraction, validation_fraction=validation_fraction)
    )
    parts = split_policies(policies, params)
    assert {name: len(part) for name, part in parts.items()} == expected


def test_split_ids_are_disjoint_and_cover_every_policy(policies):
    parts = split_policies(policies, Params())
    train, validation, test = (ids(parts[name]) for name in NAMES)
    assert not train & validation
    assert not train & test
    assert not validation & test
    assert train | validation | test == ids(policies)


def test_same_seed_gives_the_same_split_and_another_seed_a_different_one(policies):
    first = split_policies(policies, Params(seed=42))
    again = split_policies(policies, Params(seed=42))
    for name in NAMES:
        pd.testing.assert_frame_equal(first[name], again[name])
    other = split_policies(policies, Params(seed=7))
    assert ids(other["test"]) != ids(first["test"])


def test_every_split_has_the_same_share_of_claimants(policies):
    overall = (policies[schema.CLAIM_NB] > 0).mean()
    for name, part in split_policies(policies, Params()).items():
        share = (part[schema.CLAIM_NB] > 0).mean()
        # Stratification is exact up to rounding: at most a row or two per split.
        assert abs(share - overall) <= 2 / len(part), name


def test_every_split_is_sorted_by_id_with_a_fresh_index(policies):
    for part in split_policies(policies, Params()).values():
        assert part[schema.ID_COLUMN].is_monotonic_increasing
        assert part.index.equals(pd.RangeIndex(len(part)))


def test_cli_split_writes_three_valid_files(data_dir, capsys):
    assert main(["split"]) == EXIT_OK
    out = capsys.readouterr().out
    sizes = {"train": 700, "validation": 100, "test": 200}
    for name, relative in SPLIT_PATHS.items():
        written = pd.read_parquet(data_dir / relative)
        assert len(written) == sizes[name]
        assert {column: str(written[column].dtype) for column in written} == schema.DTYPES
        schema.validate_frame(written, "clean")
        assert f"wrote {sizes[name]} policies to {data_dir / relative}" in out


def test_cli_split_exits_10_on_a_bad_clean_file_and_writes_nothing(data_dir, capsys):
    processed = data_dir / paths.PROCESSED
    df = pd.read_parquet(processed)
    df.loc[0, schema.CLAIM_NB] = 7  # above claim_nb_cap: valid raw data, but not clean
    df.to_parquet(processed, index=False)

    assert main(["split"]) == EXIT_CHECK_FAILED
    err = capsys.readouterr().err
    assert "clean frame" in err
    assert "ClaimNb" in err
    assert not (data_dir / "splits").exists()
