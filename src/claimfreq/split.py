"""Role A (Data): the ``claimfreq split`` command.

``split`` reads the clean-stage file ``processed/policies.parquet`` and deals its policies into
three files under ``DATA_DIR`` (docs/contracts.md, Paths):

- ``splits/test.parquet``: ``split.test_fraction`` of all policies (default 0.2);
- ``splits/validation.parquet``: ``split.validation_fraction`` of all policies (default 0.1);
- ``splits/train.parquet``: the rest (default 0.7).

Both fractions are of the full dataset, and the sizes are exact: they are given to scikit-learn
as row counts, not fractions, which it would round up at each of the two steps. The split is
random with the top-level data ``seed``, and stratified on whether a policy has a claim, so all
three files have the same share of claimants (about 5%) and their scores are comparable. Each
policy lands in exactly one file, so the ``IDpol`` values are disjoint by construction.

Rows are shuffled with the seed, not sorted. freMTPL2's lowest ``IDpol`` values are all
claimants, so a file sorted by ``IDpol`` would start with an unrepresentative block, and a
partial replay or an early monitoring window would look like drift on clean traffic. The input
is sorted by ``IDpol`` first, so which file a policy lands in, and where, depends only on the set
of policies and the seed, never on the input's row order.

The input and all three outputs must pass the clean stage before any file is written; a failure
raises ``SchemaError``, which ``claimfreq`` turns into exit code 10.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from claimfreq.contracts import paths, schema
from claimfreq.contracts.params import Params, load_params

# Output file of each split, in the order they are reported.
SPLIT_PATHS: dict[str, Path] = {
    "train": paths.TRAIN,
    "validation": paths.VALIDATION,
    "test": paths.TEST,
}


def split_policies(df: pd.DataFrame, params: Params) -> dict[str, pd.DataFrame]:
    """Deal ``df`` into train, validation, and test frames, stratified on having a claim.

    Args:
        df: A clean-stage policy frame, in any row order and with any index (it is sorted by
            ``IDpol`` and re-indexed first).
        params: Only ``params.split`` (the fractions) and ``params.seed`` are used.

    Returns:
        ``{"train": ..., "validation": ..., "test": ...}``, each in seeded random order with
        a fresh 0..n-1 index. The same policies and seed always give the same three frames,
        whatever the input's row order.
    """
    # A canonical input order, so the result depends only on the policies and the seed.
    df = df.sort_values(schema.ID_COLUMN).reset_index(drop=True)
    n_rows = len(df)
    # Row counts rather than fractions, so both sizes are exact fractions of the full dataset.
    n_test = round(params.split.test_fraction * n_rows)
    n_validation = round(params.split.validation_fraction * n_rows)
    has_claim = df[schema.CLAIM_NB] > 0

    rest, test = train_test_split(
        df, test_size=n_test, stratify=has_claim, random_state=params.seed
    )
    train, validation = train_test_split(
        rest,
        test_size=n_validation,
        stratify=has_claim.loc[rest.index],
        random_state=params.seed,
    )
    parts = {"train": train, "validation": validation, "test": test}
    # train_test_split returns its rows in seeded random order. Keep that order: sorting by
    # IDpol would put the real data's block of claimants first.
    return {name: part.reset_index(drop=True) for name, part in parts.items()}


def split(params: Params | None = None) -> dict[Path, int]:
    """Read the clean file, split it, check all three parts, and write them.

    Args:
        params: Loaded ``params.yaml``; read from the repo root when omitted.

    Returns:
        The row count of each written file, keyed by its path.

    Raises:
        SchemaError: if the input or any split breaks the clean stage (exit 10).
    """
    if params is None:
        params = load_params()

    df = pd.read_parquet(paths.data_path(paths.PROCESSED))
    schema.validate_frame(df, "clean", params)
    parts = split_policies(df, params)
    # Check every part before writing any, so a failure never leaves a partial set of splits.
    for part in parts.values():
        schema.validate_frame(part, "clean", params)

    written: dict[Path, int] = {}
    for name, part in parts.items():
        out = paths.data_path(SPLIT_PATHS[name])
        out.parent.mkdir(parents=True, exist_ok=True)
        # The file's bytes, and so the md5 that DVC records and data_hash carries, also depend
        # on the pandas and pyarrow versions written into the parquet metadata. uv.lock pins both.
        schema.coerce_frame(part).to_parquet(out, index=False)
        written[out] = len(part)
    return written
