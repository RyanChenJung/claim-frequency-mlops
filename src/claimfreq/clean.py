"""Role A (Data): the ``claimfreq clean`` command.

``clean`` reads the raw-stage file ``raw/freMTPL2freq.parquet``, caps the two target columns,
and writes the clean-stage file ``processed/policies.parquet`` under ``DATA_DIR``
(docs/contracts.md, Paths and Data schema):

- ``ClaimNb`` above ``clean.claim_nb_cap`` (default 4) is clipped to the cap.
- ``Exposure`` above ``clean.exposure_cap`` (default 1.0 year) is clipped to the cap.

Capping at 4 claims and 1 year is the standard actuarial treatment of this dataset. Exposure
above one year cannot occur in a one-year policy period, and the few policies with more than 4
claims would dominate a count model. Nothing else changes: the rating factors stay as they are,
because every feature transform lives inside the model's Pipeline (docs/contracts.md, Model),
so training and serving always see the same inputs.

Checks run before anything is written: the input must pass the raw stage, and the capped frame
the clean stage. A failure raises ``SchemaError``, which ``claimfreq`` turns into exit code 10.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from claimfreq.contracts import paths, schema
from claimfreq.contracts.params import Params, load_params


def clip_targets(df: pd.DataFrame, params: Params) -> tuple[pd.DataFrame, dict[str, int]]:
    """Clip ``ClaimNb`` and ``Exposure`` at their caps from ``params.clean``.

    Returns:
        A copy of ``df`` with both columns clipped, and the number of clipped rows per column
        (keys ``ClaimNb`` and ``Exposure``). Values at or below a cap are unchanged.
    """
    caps = {
        schema.CLAIM_NB: params.clean.claim_nb_cap,
        schema.EXPOSURE: params.clean.exposure_cap,
    }
    clipped_df = df.copy()
    n_clipped: dict[str, int] = {}
    for column, cap in caps.items():
        n_clipped[column] = int((clipped_df[column] > cap).sum())
        clipped_df[column] = clipped_df[column].clip(upper=cap)
    return clipped_df, n_clipped


def clean(params: Params | None = None) -> tuple[Path, int, dict[str, int]]:
    """Read the raw file, cap the targets, check, and write the clean-stage file.

    Args:
        params: Loaded ``params.yaml``; read from the repo root when omitted.

    Returns:
        The written path, its row count, and the number of clipped rows per column.

    Raises:
        SchemaError: if the input breaks the raw stage, or the result the clean stage (exit 10).
    """
    if params is None:
        params = load_params()

    raw = pd.read_parquet(paths.data_path(paths.RAW))
    # Check the input first, so a stale or hand-edited raw file stops here.
    schema.validate_frame(raw, "raw")
    cleaned, n_clipped = clip_targets(raw, params)
    # The caps come from params, so the clean-stage check must use the same params.
    schema.validate_frame(cleaned, "clean", params)

    out = paths.data_path(paths.PROCESSED)
    out.parent.mkdir(parents=True, exist_ok=True)
    schema.coerce_frame(cleaned).to_parquet(out, index=False)
    return out, len(cleaned), n_clipped
