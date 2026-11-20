"""Role A (Data): the ``claimfreq ingest`` command.

``ingest`` downloads the French motor claims dataset (OpenML 41214, ADR-0001), fixes the one
quirk OpenML ships (``VehGas`` values wrapped in literal quotes), and writes the raw-stage file
``raw/freMTPL2freq.parquet`` under ``DATA_DIR`` (docs/contracts.md, Paths).

Every check runs before anything is written, so a bad download never reaches disk:

1. ``schema.validate_frame(df, "raw")`` on the frame as read. It reports every schema problem
   at once and raises ``SchemaError``; ``claimfreq`` turns that into exit code 10.
2. ``check_counts`` compares the row count, the total ``ClaimNb``, and the number of ``Region``
   codes with the documented totals. A mismatch raises ``CountCheckError`` (also exit 10).

Only then is the frame converted with ``schema.coerce_frame`` and written.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_openml

from claimfreq.contracts import paths, schema

# OpenML dataset id of freMTPL2freq (ADR-0001).
OPENML_DATA_ID = 41214

# Documented totals of the OpenML 41214 download (docs/scope.md). If OpenML ever ships a
# different version, ingest stops instead of passing changed data down the pipeline.
EXPECTED_COUNTS: dict[str, int] = {
    "rows": 678_013,
    "claims": 36_102,
    "regions": 22,
}


class CountCheckError(ValueError):
    """Raised by ``check_counts`` when the data's totals differ from the documented ones."""


def download() -> pd.DataFrame:
    """Download OpenML 41214 and return it as a DataFrame, exactly as OpenML ships it.

    This is the only function that touches the network, so tests replace it with a fake.
    scikit-learn caches the download in ``~/scikit_learn_data``, so later runs are fast.
    """
    bunch = fetch_openml(data_id=OPENML_DATA_ID, as_frame=True, parser="pandas")
    return bunch.frame


def prepare(raw: pd.DataFrame) -> pd.DataFrame:
    """Select the policy columns by name and strip the literal quotes from ``VehGas``.

    Columns are picked by name, in canonical order, so extra or reordered columns in the
    download do not matter. A missing column is left out here and reported by
    ``validate_frame`` with every other problem, instead of failing early with a ``KeyError``.
    Only ``VehGas`` is unquoted: a quoted value in any other column is a change in the source,
    and ``validate_frame`` reports it as an unknown category.
    """
    present = [column for column in schema.FRAME_COLUMNS if column in raw.columns]
    df = raw[present].copy()
    if "VehGas" in df.columns:
        # OpenML ships 'Regular' and 'Diesel' with the quotes; the schema expects them bare.
        df["VehGas"] = df["VehGas"].astype("str").str.strip("'")
    return df


def check_counts(df: pd.DataFrame, expected: dict[str, int] | None = None) -> None:
    """Raise ``CountCheckError`` unless the frame's totals match ``expected``.

    Args:
        df: The prepared frame, already validated with ``validate_frame(df, "raw")``.
        expected: ``{"rows": ..., "claims": ..., "regions": ...}``. Defaults to
            ``EXPECTED_COUNTS``, read at call time so tests can swap in small numbers.

    The error lists every total that differs, with the expected and the actual value.
    """
    expected = EXPECTED_COUNTS if expected is None else expected
    actual = {
        "rows": len(df),
        # Validation has already confirmed that ClaimNb holds whole numbers.
        "claims": int(df[schema.CLAIM_NB].sum()),
        "regions": int(df["Region"].nunique()),
    }
    problems = [
        f"{name}: expected {expected[name]:,}, got {actual[name]:,}"
        for name in expected
        if actual[name] != expected[name]
    ]
    if problems:
        lines = "\n".join(f"- {problem}" for problem in problems)
        raise CountCheckError(
            f"OpenML {OPENML_DATA_ID} totals differ from the documented ones:\n{lines}"
        )


def ingest() -> tuple[Path, int]:
    """Download, check, and write the raw-stage file; return its path and its row count.

    Raises:
        SchemaError: if the download breaks the raw-stage schema (exit 10).
        CountCheckError: if its totals differ from ``EXPECTED_COUNTS`` (exit 10).
    """
    df = prepare(download())
    # Both checks run on the frame before coercion, and before anything is written.
    schema.validate_frame(df, "raw")
    check_counts(df)

    out = paths.data_path(paths.RAW)
    out.parent.mkdir(parents=True, exist_ok=True)
    schema.coerce_frame(df).to_parquet(out, index=False)
    return out, len(df)
