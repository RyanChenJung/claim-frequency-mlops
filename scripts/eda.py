"""Exploratory data analysis of freMTPL2freq: the plots for slide a and the datasheet's numbers.

Run it from the repo root once the data pipeline has written its files under ``DATA_DIR``:

    uv run python scripts/eda.py

It reads the pipeline's files and writes six PNGs and ``summary.json`` to ``docs/eda/``
(``--out-dir`` changes the folder):

- The plots and the ``train`` figures use the train split only, so nothing that could shape a
  modelling choice looks at validation or test data. Train is post-cap: ``ClaimNb`` and
  ``Exposure`` are clipped at ``clean.claim_nb_cap`` and ``clean.exposure_cap``, which the
  script reads from params.yaml (Role A's own keys).
- The ``dataset`` figures describe all 678,013 policies, read from the raw and clean files,
  so the datasheet can quote the tails that the caps remove.
- The ``duplicates`` figures read all three splits: the share of validation and test rows
  whose exact 9-factor tuple also appears in train (#29, from the review of #53).

Observed frequency is claims per policy-year, ``sum(ClaimNb) / sum(Exposure)`` per band, the
quantity a Poisson model with an exposure offset estimates. It is not the mean of each
policy's ``ClaimNb / Exposure``, which short policies would dominate.

Re-runs on the same data write byte-identical files: fixed figure sizes, no timestamps, and
no matplotlib version in the PNG metadata. The script uses matplotlib's ``Figure`` API rather
than ``pyplot``, so it needs no GUI backend and keeps no global figure state.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from claimfreq.contracts import paths, schema
from claimfreq.contracts.model import file_md5
from claimfreq.contracts.params import Params, load_params

DEFAULT_OUT_DIR = Path("docs/eda")

# Band edges: each band runs from one edge up to the next edge minus 1, and the last one is
# open-ended. They are defined here, not read from params.yaml, because a key that two roles
# read becomes a contract key (docs/contracts.md, Params). The DrivAge edges equal the GLM's
# `training.glm.drivage_edges` today, so the plot shows the bands the GLM uses.
BAND_EDGES: dict[str, list[int]] = {
    "DrivAge": [18, 21, 25, 30, 40, 50, 60, 70],  # 18-20, 21-24, 25-29, ..., 70+
    "BonusMalus": [50, 51, 60, 70, 80, 90, 100, 125],  # 50 is the best level
    "VehAge": [0, 1, 2, 3, 6, 11, 16, 21],  # 0, 1, 2, 3-5, 6-10, 11-15, 16-20, 21+
}
# Factors plotted as they are, one bar per category, in this order (rural A to urban F).
CATEGORY_ORDER: dict[str, list[str]] = {"Area": [a.value for a in schema.Area]}
FREQUENCY_FACTORS = ["DrivAge", "BonusMalus", "VehAge", "Area"]

# 16:9, so each plot fills a slide. With no "Software" key, a matplotlib upgrade alone does
# not change the PNG bytes.
FIGSIZE = (8, 4.5)
DPI = 150
PNG_METADATA = {"Software": None}
DECIMALS = 4  # rounding in summary.json, so diffs stay readable


# --- Computations (pure functions, tested on hand-built frames) ----------------------------


def band_labels(edges: Sequence[int]) -> list[str]:
    """Return one label per band: ``[18, 21, 25]`` gives ``["18-20", "21-24", "25+"]``.

    A band one value wide gets that value alone (``"50"``); the last band is open-ended.
    """
    labels = []
    for low, high in zip(edges[:-1], edges[1:], strict=True):
        labels.append(str(low) if high - low == 1 else f"{low}-{high - 1}")
    labels.append(f"{edges[-1]}+")
    return labels


def assign_bands(values: pd.Series, edges: Sequence[int]) -> pd.Series:
    """Return the band label of each value, as an ordered categorical with every band.

    Raises:
        ValueError: if a value lies below the first edge (the schema bounds rule this out).
    """
    codes = np.searchsorted(np.asarray(edges), values.to_numpy(), side="right") - 1
    if (codes < 0).any():
        raise ValueError(f"{values.name} has values below the first band edge {edges[0]}")
    bands = pd.Categorical.from_codes(codes, categories=band_labels(edges), ordered=True)
    return pd.Series(bands, index=values.index, name="band")


def factor_bands(df: pd.DataFrame, column: str) -> pd.Series:
    """Return the plotting band of each row: a numeric band, or the category itself."""
    if column in BAND_EDGES:
        return assign_bands(df[column], BAND_EDGES[column])
    categories = CATEGORY_ORDER[column]
    return pd.Series(
        pd.Categorical(df[column], categories=categories, ordered=True),
        index=df.index,
        name="band",
    )


def frequency_table(df: pd.DataFrame, bands: pd.Series) -> pd.DataFrame:
    """Return policies, claims, exposure, and observed frequency per band.

    ``frequency`` is ``claims / exposure``, a ratio of sums. Every band appears, in order; an
    empty band has 0 policies and a NaN frequency.
    """
    table = df.groupby(bands, observed=False).agg(
        policies=(schema.CLAIM_NB, "size"),
        claims=(schema.CLAIM_NB, "sum"),
        exposure=(schema.EXPOSURE, "sum"),
    )
    table["frequency"] = table["claims"] / table["exposure"].where(table["exposure"] > 0)
    return table


def portfolio_frequency(df: pd.DataFrame) -> float:
    """Return the frame's observed frequency, ``sum(ClaimNb) / sum(Exposure)``."""
    return float(df[schema.CLAIM_NB].sum() / df[schema.EXPOSURE].sum())


def claim_count_shares(df: pd.DataFrame) -> pd.Series:
    """Return the share of policies with each ``ClaimNb`` value that occurs, in value order.

    Values that never occur are left out, so every share is positive (a log axis needs that).
    """
    return df[schema.CLAIM_NB].value_counts(normalize=True).sort_index()


def cross_split_duplicate_share(train: pd.DataFrame, other: pd.DataFrame) -> float:
    """Return the share of ``other``'s rows whose 9 rating factors exactly match a train row.

    Only ``schema.FEATURE_COLUMNS`` are compared; ``IDpol`` and the targets are ignored.
    """
    in_train = pd.MultiIndex.from_frame(other[schema.FEATURE_COLUMNS]).isin(
        pd.MultiIndex.from_frame(train[schema.FEATURE_COLUMNS])
    )
    return float(in_train.mean())


def within_duplicate_share(df: pd.DataFrame) -> float:
    """Return the share of rows whose 9 rating factors occur on at least one other row.

    Every copy counts: in ``[a, a, b, c]`` the share is 0.5.
    """
    return float(df.duplicated(schema.FEATURE_COLUMNS, keep=False).mean())


def changed_rows(before: pd.DataFrame, after: pd.DataFrame, column: str) -> int:
    """Return how many of ``after``'s rows hold a different ``column`` value than ``before``.

    Rows are matched on ``IDpol``. With the raw file as ``before``, this counts the rows that
    ``clean`` capped among the policies of ``after``.
    """
    before_values = before.set_index(schema.ID_COLUMN)[column]
    after_values = after.set_index(schema.ID_COLUMN)[column]
    return int((before_values.loc[after_values.index] != after_values).sum())


def _round(value: float) -> float:
    return round(float(value), DECIMALS)


def _table_records(table: pd.DataFrame) -> list[dict[str, Any]]:
    """Return a frequency table as JSON-ready rows; an empty band's frequency is null."""
    return [
        {
            "band": str(band),
            "policies": int(row["policies"]),
            "claims": int(row["claims"]),
            "exposure": _round(row["exposure"]),
            "frequency": None if pd.isna(row["frequency"]) else _round(row["frequency"]),
        }
        for band, row in table.iterrows()
    ]


def summarize(
    raw: pd.DataFrame,
    clean: pd.DataFrame,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    params: Params,
) -> dict[str, Any]:
    """Return every number the datasheet and slide a quote, grouped as summary.json is.

    - ``dataset``: all policies, before (raw) and after (clean) the caps.
    - ``train``: the train split, after the caps; the plots show the same data.
    - ``duplicates``: shares of rows sharing their exact 9-factor tuple (see the module doc).
    """
    dataset = {
        "rows": len(raw),
        "claims_before_cap": int(raw[schema.CLAIM_NB].sum()),
        "claims_after_cap": int(clean[schema.CLAIM_NB].sum()),
        "claimant_share": _round((clean[schema.CLAIM_NB] > 0).mean()),
        "claim_nb_max_before_cap": int(raw[schema.CLAIM_NB].max()),
        "exposure_max_before_cap": _round(raw[schema.EXPOSURE].max()),
        "claim_nb_capped_rows": changed_rows(raw, clean, schema.CLAIM_NB),
        "exposure_capped_rows": changed_rows(raw, clean, schema.EXPOSURE),
        "regions": int(clean["Region"].nunique()),
    }
    shares = claim_count_shares(train)
    train_summary = {
        "rows": len(train),
        "claims": int(train[schema.CLAIM_NB].sum()),
        "exposure_years": _round(train[schema.EXPOSURE].sum()),
        "claims_per_exposure_year": _round(portfolio_frequency(train)),
        "zero_claim_share": _round(shares.get(0, 0.0)),
        # Counts, not shares: the share of 4-claim policies rounds to 0 at 4 decimals.
        "claim_counts": {
            str(k): int(v) for k, v in train[schema.CLAIM_NB].value_counts().sort_index().items()
        },
        "mean_exposure_after_cap": _round(train[schema.EXPOSURE].mean()),
        "full_year_share_after_cap": _round(
            (train[schema.EXPOSURE] == params.clean.exposure_cap).mean()
        ),
        "claim_nb_capped_rows": changed_rows(raw, train, schema.CLAIM_NB),
        "exposure_capped_rows": changed_rows(raw, train, schema.EXPOSURE),
        "frequency_by": {
            column: _table_records(frequency_table(train, factor_bands(train, column)))
            for column in FREQUENCY_FACTORS
        },
    }
    duplicates = {
        "validation_rows_matching_train": _round(cross_split_duplicate_share(train, validation)),
        "test_rows_matching_train": _round(cross_split_duplicate_share(train, test)),
        "train_rows_duplicated_within_train": _round(within_duplicate_share(train)),
    }
    return {"dataset": dataset, "train": train_summary, "duplicates": duplicates}


# --- Plots ---------------------------------------------------------------------------------


def _figure() -> tuple[Figure, Any]:
    figure = Figure(figsize=FIGSIZE, dpi=DPI, layout="constrained")
    return figure, figure.subplots()


def _save(figure: Figure, path: Path) -> Path:
    figure.savefig(path, dpi=DPI, metadata=PNG_METADATA)
    return path


def plot_claim_counts(train: pd.DataFrame, path: Path, claim_nb_cap: int) -> Path:
    """Bar chart of the share of policies per claim count, on a log axis with % labels."""
    shares = claim_count_shares(train)
    figure, ax = _figure()
    bars = ax.bar(shares.index.astype(str), shares.to_numpy(), color="C0")
    # Three significant digits, so the rare counts do not print as 0.00%.
    ax.bar_label(bars, labels=[f"{100 * s:.3g}%" for s in shares], padding=2)
    ax.set_yscale("log")
    ax.set_ylim(top=3)  # headroom above the tallest bar's label
    ax.set_xlabel(f"Claims in the policy period (ClaimNb, capped at {claim_nb_cap})")
    ax.set_ylabel("Share of policies (log scale)")
    ax.set_title(f"Claim counts (train): {shares.get(0, 0.0):.1%} of policies have no claim")
    return _save(figure, path)


def plot_exposure(train: pd.DataFrame, path: Path, capped_rows: int, exposure_cap: float) -> Path:
    """Histogram of exposure up to the cap, with the full-year spike explained in a note."""
    exposure = train[schema.EXPOSURE]
    figure, ax = _figure()
    ax.hist(exposure, bins=50, range=(0.0, exposure_cap), color="C0")
    full_year = (exposure == exposure_cap).mean()
    ax.text(
        0.02,
        0.95,
        f"Full-year policies: {full_year:.1%} of train\n(includes {capped_rows:,} capped rows)",
        transform=ax.transAxes,
        va="top",
    )
    ax.set_xlabel(f"Exposure (years, capped at {exposure_cap:g})")
    ax.set_ylabel("Policies")
    ax.set_title("Exposure per policy (train)")
    return _save(figure, path)


def plot_frequency(train: pd.DataFrame, column: str, path: Path) -> Path:
    """Observed frequency per band (line, left axis) over exposure per band (bars, right)."""
    table = frequency_table(train, factor_bands(train, column))
    x = np.arange(len(table))
    figure, ax = _figure()
    bars_ax = ax.twinx()
    bars_ax.bar(x, table["exposure"], color="lightgrey", label="Exposure")
    # Draw the frequency axis above the bars, with a see-through background.
    ax.set_zorder(bars_ax.get_zorder() + 1)
    ax.patch.set_visible(False)
    ax.plot(x, table["frequency"], marker="o", color="C0", label="Observed frequency")
    average = portfolio_frequency(train)
    ax.axhline(average, color="grey", linestyle="--", label=f"Train average ({average:.3f})")
    ax.set_xticks(x, [str(band) for band in table.index])
    ax.set_ylim(bottom=0)
    ax.set_xlabel(column)
    ax.set_ylabel("Claims per policy-year")
    bars_ax.set_ylabel("Exposure (policy-years)")
    handles, labels = ax.get_legend_handles_labels()
    bar_handles, bar_labels = bars_ax.get_legend_handles_labels()
    # Below the axes, so it never covers a point or a bar.
    figure.legend(handles + bar_handles, labels + bar_labels, loc="outside lower center", ncols=3)
    ax.set_title(f"Observed claim frequency by {column} (train)")
    return _save(figure, path)


# --- Entry point ---------------------------------------------------------------------------


def run(out_dir: Path = DEFAULT_OUT_DIR, params: Params | None = None) -> dict[str, Any]:
    """Read the pipeline's files under ``DATA_DIR``, write the plots and summary.json.

    Args:
        out_dir: Folder for the six PNGs and summary.json.
        params: Loaded ``params.yaml``, for the caps; read from the repo root when omitted.

    Returns:
        The summary that was written to ``out_dir / "summary.json"``.
    """
    if params is None:
        params = load_params()
    raw = schema.coerce_frame(pd.read_parquet(paths.data_path(paths.RAW)))
    clean = pd.read_parquet(paths.data_path(paths.PROCESSED))
    train_path = paths.data_path(paths.TRAIN)
    train = pd.read_parquet(train_path)
    validation = pd.read_parquet(paths.data_path(paths.VALIDATION))
    test = pd.read_parquet(paths.data_path(paths.TEST))

    summary = summarize(raw, clean, train, validation, test, params)
    # The data_hash lineage tag (model.file_md5), also recorded for train.parquet in dvc.lock,
    # so the committed plots name the exact data version that produced them.
    summary["train"]["md5"] = file_md5(train_path)

    out_dir.mkdir(parents=True, exist_ok=True)
    plot_claim_counts(train, out_dir / "claim_count.png", params.clean.claim_nb_cap)
    plot_exposure(
        train,
        out_dir / "exposure.png",
        summary["train"]["exposure_capped_rows"],
        params.clean.exposure_cap,
    )
    for column in FREQUENCY_FACTORS:
        plot_frequency(train, column, out_dir / f"freq_{column.lower()}.png")
    text = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    (out_dir / "summary.json").write_text(text, encoding="utf-8")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    """Parse ``--out-dir``, run the EDA, and print the headline numbers."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)
    summary = run(args.out_dir)
    print(f"Wrote 6 plots and summary.json to {args.out_dir}")
    print(json.dumps(summary["duplicates"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
