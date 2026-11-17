"""Paths contract: where each pipeline stage reads and writes its data files.

All paths are relative to ``DATA_DIR`` (environment variable, default ``data``). Role A writes
them as DVC stage outputs (``dvc.yaml`` refers to them as ``data/...``); Roles B and D read them.
Every file is parquet with the column order and dtypes of ``schema.coerce_frame``, except
the ``schema_break`` and ``out_of_bounds`` drift files, which break the schema on purpose.

Stage outputs:

- ``raw/freMTPL2freq.parquet``: ``ingest`` output, raw stage.
- ``processed/policies.parquet``: ``clean`` output, clean stage.
- ``splits/{train,validation,test}.parquet``: ``split`` output, clean stage. ``dvc.yaml``
  declares the three files as separate outs, so ``dvc.lock`` records the plain content md5 of
  ``train.parquet`` that the ``data_hash`` tag carries (``model.file_md5``).
- ``drift/<scenario>.parquet``: ``make-drift`` output, one file per ``DriftScenario``, all five
  written by one DVC stage.
- ``fake/policies.parquet``: default ``fake-data`` output, clean stage, git-ignored.
"""

from enum import StrEnum
from pathlib import Path

from claimfreq.contracts import env

RAW = Path("raw/freMTPL2freq.parquet")
PROCESSED = Path("processed/policies.parquet")
TRAIN = Path("splits/train.parquet")
VALIDATION = Path("splits/validation.parquet")
TEST = Path("splits/test.parquet")
DRIFT_DIR = Path("drift")
FAKE = Path("fake/policies.parquet")


class DriftScenario(StrEnum):
    """Named, reproducible corruptions of the test split (scope.md, Drift scenarios).

    Every drift file keeps ``IDpol``, ``ClaimNb``, and ``Exposure`` under their canonical names
    and dtypes. Only rating factors are corrupted, except in ``concept_drift``, which changes
    ``ClaimNb`` only.

    - ``schema_break``: ``DrivAge`` renamed to ``DriverAge`` on all rows. Fails the clean
      schema, so it is never validated; the API rejects it with 422.
    - ``out_of_bounds``: ``DrivAge = 250`` and ``BonusMalus = -1`` on all rows. Fails the clean
      schema, so it is never validated; the API rejects it with 422.
    - ``column_swap``: swaps ``VehAge`` and ``VehPower`` only on the rows where both values are
      valid in the other column, that is where ``VehAge`` is within ``VehPower``'s bounds
      (4-15); ``VehPower`` always fits ``VehAge``'s bounds (0-100). All other rows stay
      unchanged and the file keeps every test row. Passes the clean schema; only distribution
      checks catch it.
    - ``portfolio_shift``: weighted sampling without replacement from the test split, toward
      young (low ``DrivAge``) and dense (high ``Density``) policies. ``IDpol`` is kept, so it
      stays unique. Passes the clean schema. The exact weights are Role A's choice, as long as
      mean ``DrivAge`` drops by at least 5 years and median ``Density`` rises by at least 50%.
    - ``concept_drift``: rating factors unchanged; ``ClaimNb`` increased so that the true claim
      rate is higher, still clipped to ``clean.claim_nb_cap``. Passes the clean schema; only the
      calibration ratio on ground truth catches it. The exact factor is Role A's choice, as long
      as the file's observed claim rate is at least 25% above the test split's.

    ``send-traffic`` posts every column except ``IDpol``, ``ClaimNb``, and ``Exposure`` as is,
    with no coercion, uses ``IDpol`` as ``policy_id``, and keeps ``ClaimNb`` and ``Exposure``
    for ``ground_truth``.
    """

    SCHEMA_BREAK = "schema_break"
    OUT_OF_BOUNDS = "out_of_bounds"
    COLUMN_SWAP = "column_swap"
    PORTFOLIO_SHIFT = "portfolio_shift"
    CONCEPT_DRIFT = "concept_drift"


def data_dir() -> Path:
    """Return ``DATA_DIR`` as a path, read at call time so tests can override it."""
    # An empty DATA_DIR falls back to the default rather than to the current directory.
    return Path(env.get(env.DATA_DIR) or env.DEFAULTS[env.DATA_DIR])


def data_path(relative: Path) -> Path:
    """Resolve one of this module's relative paths under ``DATA_DIR``.

    Example: ``data_path(paths.TRAIN)`` is ``data/splits/train.parquet`` by default.
    """
    return data_dir() / relative


def drift_path(scenario: DriftScenario) -> Path:
    """Return the data file of one drift scenario under ``DATA_DIR``."""
    return data_dir() / DRIFT_DIR / f"{DriftScenario(scenario)}.parquet"
