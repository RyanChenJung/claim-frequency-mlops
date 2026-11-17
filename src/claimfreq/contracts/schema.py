"""Data schema contract: the single source of truth for policy columns.

Every role reads column names, dtypes, valid ranges, and categories from this module:

- Role A (Data) validates its ingest, clean, split, and drift outputs with ``validate_frame``.
- Role B (Modeling) trains on ``FEATURE_COLUMNS`` and tags model versions with ``schema_hash()``.
- Role D (Serving and monitoring) validates API requests with ``PolicyFeatures`` and logs the
  same column names, so monitoring compares live traffic with the reference dataset without
  renaming anything.

Two layers use the same constants:

- ``NUMERIC_BOUNDS`` and the category enums are the hard validity bounds. The API rejects values
  outside them with HTTP 422. They are domain-plausible: most equal the observed range, and only
  ``BonusMalus`` and ``Density`` are wider, so a realistic portfolio shift still passes
  validation and is left for drift monitoring to catch.
- Frame stages (``raw`` and ``clean``) add target-column rules on top of the feature bounds.

Order of operations: validate a frame as it was read (``find_violations`` or ``validate_frame``,
which report every problem at once), then convert it with ``coerce_frame`` or
``coerce_features``. Raw-stage validation accepts a float column of whole numbers in an integer
slot, as OpenML ships ``IDpol`` and ``ClaimNb``; clean-stage frames must already hold integers.
Coercion never truncates, wraps, or fills, whatever the source dtype: it raises on a null, a
non-integral value, or a value outside int64.

Any change to the feature spec (names, order, dtypes, bounds, categories) changes
``schema_hash()``, must bump ``SCHEMA_VERSION``, and implies a MAJOR model version (ADR-0007).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    # Imported for type hints only: params.py imports this module for the default caps.
    from claimfreq.contracts.params import Params

SCHEMA_VERSION = "1.0.0"

# --- Column names -------------------------------------------------------------------------------

ID_COLUMN = "IDpol"
CLAIM_NB = "ClaimNb"
EXPOSURE = "Exposure"
TARGET_COLUMNS = [CLAIM_NB, EXPOSURE]

# Rating factors (model inputs) in canonical order. Model input frames, API payloads, and the
# monitoring log all use exactly these names.
FEATURE_COLUMNS = [
    "VehPower",
    "VehAge",
    "DrivAge",
    "BonusMalus",
    "VehBrand",
    "VehGas",
    "Area",
    "Density",
    "Region",
]
NUMERIC_FEATURES = ["VehPower", "VehAge", "DrivAge", "BonusMalus", "Density"]
CATEGORICAL_FEATURES = ["VehBrand", "VehGas", "Area", "Region"]

# All columns of a policy frame (raw and clean stages), in canonical order.
FRAME_COLUMNS = [ID_COLUMN, CLAIM_NB, EXPOSURE, *FEATURE_COLUMNS]

# Canonical pandas dtypes. "str" is pandas' default string dtype from pandas 3 onward.
DTYPES: dict[str, str] = {
    ID_COLUMN: "int64",
    CLAIM_NB: "int64",
    EXPOSURE: "float64",
    **{column: "int64" for column in NUMERIC_FEATURES},
    **{column: "str" for column in CATEGORICAL_FEATURES},
}

# --- Hard validity bounds -----------------------------------------------------------------------

# Inclusive (min, max) per numeric rating factor. Observed OpenML ranges are in the comments.
# Only BonusMalus and Density are wider than observed, so plausible-but-shifted traffic on them
# still reaches monitoring; the others equal the observed range.
NUMERIC_BOUNDS: dict[str, tuple[int, int]] = {
    "VehPower": (4, 15),  # observed 4-15: the dataset's class range
    "VehAge": (0, 100),  # observed 0-100
    "DrivAge": (18, 100),  # observed 18-100; 18 is the French minimum driving age
    "BonusMalus": (50, 350),  # observed 50-230; legal coefficient range is 0.50-3.50
    "Density": (1, 50_000),  # observed 1-27,000 inhabitants per km2
}

# Default clean-stage caps. params.yaml may override them (see params.py). Capping ClaimNb at 4
# and Exposure at 1 year is the standard actuarial treatment, also used in scikit-learn's
# Poisson regression example.
DEFAULT_CLAIM_NB_CAP = 4
DEFAULT_EXPOSURE_CAP = 1.0


# --- Categories ---------------------------------------------------------------------------------
# StrEnum members are plain strings, so JSON payloads and parquet files carry the bare values
# while Python code still gets static types.


class Area(StrEnum):
    """Area code of the policyholder's municipality, from A (rural) to F (urban)."""

    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"
    F = "F"


class VehBrand(StrEnum):
    """Anonymized vehicle brand code (11 values)."""

    B1 = "B1"
    B2 = "B2"
    B3 = "B3"
    B4 = "B4"
    B5 = "B5"
    B6 = "B6"
    B10 = "B10"
    B11 = "B11"
    B12 = "B12"
    B13 = "B13"
    B14 = "B14"


class VehGas(StrEnum):
    """Fuel type. OpenML ships quoted values (``'Regular'``); ingest strips the quotes."""

    REGULAR = "Regular"
    DIESEL = "Diesel"


class Region(StrEnum):
    """Pre-2016 French region as an INSEE code (22 values), as shipped by OpenML 41214."""

    R11 = "R11"
    R21 = "R21"
    R22 = "R22"
    R23 = "R23"
    R24 = "R24"
    R25 = "R25"
    R26 = "R26"
    R31 = "R31"
    R41 = "R41"
    R42 = "R42"
    R43 = "R43"
    R52 = "R52"
    R53 = "R53"
    R54 = "R54"
    R72 = "R72"
    R73 = "R73"
    R74 = "R74"
    R82 = "R82"
    R83 = "R83"
    R91 = "R91"
    R93 = "R93"
    R94 = "R94"


# Allowed values per categorical column, in enum definition order.
CATEGORIES: dict[str, list[str]] = {
    "VehBrand": [member.value for member in VehBrand],
    "VehGas": [member.value for member in VehGas],
    "Area": [member.value for member in Area],
    "Region": [member.value for member in Region],
}


# --- Pydantic model (API validation) ------------------------------------------------------------


def _bounded_int(column: str) -> Any:
    """Return a strict integer Field whose bounds come from ``NUMERIC_BOUNDS``.

    Building the Field from the constant keeps the API bounds and the frame bounds identical.
    """
    low, high = NUMERIC_BOUNDS[column]
    return Field(strict=True, ge=low, le=high)


class PolicyFeatures(BaseModel):
    """The 9 rating factors of one policy, as the API receives them.

    - ``extra="forbid"`` makes renamed or unknown fields fail. The ``schema_break`` drift
      scenario relies on this to produce HTTP 422.
    - Integer fields are strict: ``"35"``, ``35.0``, ``True``, and ``numpy.int64(35)`` are all
      rejected, because silently coercing them would hide a broken client. Build payloads from
      JSON or from native Python ints (``DataFrame.to_dict("records")`` returns native ints).
    - Categorical fields accept the plain string values (for example ``"Regular"``).
    """

    model_config = ConfigDict(extra="forbid")

    VehPower: int = _bounded_int("VehPower")
    VehAge: int = _bounded_int("VehAge")
    DrivAge: int = _bounded_int("DrivAge")
    BonusMalus: int = _bounded_int("BonusMalus")
    VehBrand: VehBrand
    VehGas: VehGas
    Area: Area
    Density: int = _bounded_int("Density")
    Region: Region


# --- Frame validation ---------------------------------------------------------------------------

Stage = Literal["raw", "clean"]


@dataclass(frozen=True)
class Violation:
    """One failed check on a policy frame.

    ``check`` is one of ``duplicate_column``, ``missing_column``, ``extra_column``, ``dtype``,
    ``null``, ``out_of_range``, ``unknown_category``, or ``duplicate_id``. ``count`` is the
    number of offending rows (1 for column-level checks).
    """

    check: str
    column: str
    count: int
    message: str


class SchemaError(ValueError):
    """Raised by ``validate_frame`` with every violation found, not just the first one."""

    def __init__(self, stage: Stage, violations: list[Violation]) -> None:
        self.stage = stage
        self.violations = violations
        lines = "\n".join(f"- {v.message}" for v in violations)
        super().__init__(f"{len(violations)} schema violation(s) in {stage} frame:\n{lines}")


def _fits_int64(values: np.ndarray) -> bool:
    """Return True when every value lies in int64's range [-2**63, 2**63).

    Both bounds are exact in float64; int64's maximum 2**63 - 1 is not, because float64 rounds
    it up to 2**63, so the upper bound must be exclusive.
    """
    return bool(((values >= -(2.0**63)) & (values < 2.0**63)).all())


def _has_dtype_class(series: pd.Series, dtype: str, allow_integral_float: bool) -> bool:
    """Check the dtype class rather than the exact dtype.

    Comparing against ``object`` breaks under pandas 3 (strings default to ``str``), and a
    nullable ``Int64`` or a ``category`` of strings is as good as the canonical dtype here.

    With ``allow_integral_float`` (raw stage only), a float column passes as int64 when every
    non-null value is a whole number inside int64's range, because OpenML ships ``IDpol`` and
    ``ClaimNb`` as floats and ``coerce_frame`` converts them without loss. One value such as 35.7
    or 1e20 makes it a dtype violation. Clean-stage files are written after ``coerce_frame``, so
    they must already hold integers: ``send-traffic`` posts their values as is, and the strict
    API rejects ``34.0``. An unsigned column passes only if it fits int64.

    The integrality test runs on numpy values, so pyarrow-backed columns work too. Nulls are left
    out of every judgment, since the null check reports those rows: pandas does not count an
    ``object`` column of strings with a ``None`` in it as a string column.
    """
    if dtype == "int64":
        non_null = series.dropna()
        if pd.api.types.is_float_dtype(series):
            if not allow_integral_float:
                return False
            values = non_null.to_numpy(dtype="float64")
            return bool((np.mod(values, 1) == 0).all()) and _fits_int64(values)
        if pd.api.types.is_unsigned_integer_dtype(series):
            return bool((non_null.to_numpy(dtype="uint64") < np.uint64(2**63)).all())
        return pd.api.types.is_integer_dtype(series)
    if dtype == "float64":
        return pd.api.types.is_float_dtype(series)
    return pd.api.types.is_string_dtype(series.dropna())


def _range_limits(stage: Stage, params: Params | None) -> dict[str, tuple[float, float]]:
    """Return inclusive (min, max) limits per numeric column for a stage."""
    if params is None:
        claim_nb_cap, exposure_cap = DEFAULT_CLAIM_NB_CAP, DEFAULT_EXPOSURE_CAP
    else:
        claim_nb_cap, exposure_cap = params.clean.claim_nb_cap, params.clean.exposure_cap
    inf = float("inf")
    # Raw data keeps the long tails (ClaimNb up to 16, Exposure up to 2.01); clean caps them.
    claim_nb_max = inf if stage == "raw" else claim_nb_cap
    exposure_max = inf if stage == "raw" else exposure_cap
    return {
        ID_COLUMN: (1, inf),
        CLAIM_NB: (0, claim_nb_max),
        # Exposure must be strictly positive; the strict lower bound is applied separately.
        EXPOSURE: (0, exposure_max),
        **NUMERIC_BOUNDS,
    }


def find_violations(
    df: pd.DataFrame, stage: Stage, params: Params | None = None
) -> list[Violation]:
    """Return every schema violation of a policy frame, without raising.

    Args:
        df: A raw-stage or clean-stage policy frame (``FRAME_COLUMNS``, any column order).
        stage: ``"raw"`` (ingest output) or ``"clean"`` (clean, split, and drift outputs that
            are meant to be valid).
        params: Loaded ``params.yaml``. Only ``params.clean`` is used, for the clean-stage caps.
            When omitted, the defaults (4 claims, 1.0 year) apply, so pass ``load_params()``
            if the caps were tuned.

    Returns:
        A list of violations; empty when the frame is valid. All checks are vectorized, and a
        column with the wrong dtype class skips its value checks, which would be meaningless.
        Duplicate column names are reported alone, because ``df[name]`` is then ambiguous and
        no other check can run.
    """
    duplicated = df.columns[df.columns.duplicated()].unique()
    if len(duplicated):
        return [
            Violation("duplicate_column", str(column), 1, f"duplicate column {column!r}")
            for column in duplicated
        ]

    violations: list[Violation] = []

    for column in FRAME_COLUMNS:
        if column not in df.columns:
            violations.append(Violation("missing_column", column, 1, f"missing column {column!r}"))
    for column in df.columns:
        if column not in FRAME_COLUMNS:
            violations.append(
                Violation("extra_column", str(column), 1, f"unexpected column {column!r}")
            )

    limits = _range_limits(stage, params)
    for column in FRAME_COLUMNS:
        if column not in df.columns:
            continue
        series = df[column]

        n_null = int(series.isna().sum())
        if n_null:
            violations.append(Violation("null", column, n_null, f"{column}: {n_null} null(s)"))

        expected = DTYPES[column]
        if not _has_dtype_class(series, expected, allow_integral_float=stage == "raw"):
            violations.append(
                Violation(
                    "dtype", column, 1, f"{column}: dtype {series.dtype}, expected {expected}"
                )
            )
            continue

        if column in limits:
            low, high = limits[column]
            strict_low = column == EXPOSURE
            # NaN compares False on both sides, so nulls are not double-counted here.
            too_low = series <= low if strict_low else series < low
            bad = too_low | (series > high)
            n_bad = int(bad.sum())
            if n_bad:
                interval = f"{'(' if strict_low else '['}{low}, {high}]"
                shown = sorted(series[bad].unique().tolist())[:5]
                violations.append(
                    Violation(
                        "out_of_range",
                        column,
                        n_bad,
                        f"{column}: {n_bad} value(s) outside {interval}, e.g. {shown}",
                    )
                )

        if column in CATEGORIES:
            unknown = series.notna() & ~series.isin(CATEGORIES[column])
            n_unknown = int(unknown.sum())
            if n_unknown:
                shown = sorted(map(str, series[unknown].unique().tolist()))[:5]
                violations.append(
                    Violation(
                        "unknown_category",
                        column,
                        n_unknown,
                        f"{column}: {n_unknown} unknown categor(ies), e.g. {shown}",
                    )
                )

    if ID_COLUMN in df.columns:
        n_dup = int(df[ID_COLUMN].dropna().duplicated().sum())
        if n_dup:
            violations.append(
                Violation("duplicate_id", ID_COLUMN, n_dup, f"{ID_COLUMN}: {n_dup} duplicate(s)")
            )

    return violations


def validate_frame(df: pd.DataFrame, stage: Stage, params: Params | None = None) -> None:
    """Raise ``SchemaError`` listing all violations if ``df`` breaks the stage's schema.

    See ``find_violations`` for the arguments and the checks. The ``schema_break`` and
    ``out_of_bounds`` drift files violate the schema on purpose and are never validated.
    """
    violations = find_violations(df, stage, params)
    if violations:
        raise SchemaError(stage, violations)


def _to_int64(column: str, series: pd.Series) -> pd.Series:
    """Convert one null-free column to int64, raising ``ValueError`` rather than losing a value.

    ``pd.to_numeric`` first turns ``object`` and ``category`` columns of numbers into a numeric
    dtype, and a pyarrow-backed result is moved to its numpy dtype (pyarrow has no ``%``). A
    signed integer (or boolean) result fits int64 as is. Any other result is checked first,
    because ``astype`` would truncate 35.7 to 35, saturate 1e20 at int64's maximum, and wrap
    2**63 around to -2**63.
    """
    try:
        numeric = pd.to_numeric(series)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{column}: not numeric ({error})") from None
    if isinstance(numeric.dtype, pd.ArrowDtype):
        numeric = numeric.astype(numeric.dtype.numpy_dtype)
    if pd.api.types.is_signed_integer_dtype(numeric) or pd.api.types.is_bool_dtype(numeric):
        return numeric.astype("int64")
    if pd.api.types.is_unsigned_integer_dtype(numeric):
        outside = numeric[numeric.to_numpy(dtype="uint64") >= np.uint64(2**63)]
    else:
        values = numeric.to_numpy(dtype="float64")
        non_integral = numeric[np.mod(values, 1) != 0]
        if len(non_integral):
            shown = non_integral.unique().tolist()[:5]
            raise ValueError(f"{column}: non-integral values for int64, e.g. {shown}")
        outside = numeric[(values < -(2.0**63)) | (values >= 2.0**63)]
    if len(outside):
        shown = outside.unique().tolist()[:5]
        raise ValueError(f"{column}: values outside the int64 range, e.g. {shown}")
    return numeric.astype("int64")


def _coerce(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Select ``columns`` in order and cast them to canonical dtypes without losing information.

    A plain ``astype`` silently truncates 35.7 to 35 and keeps a null in a string column, so
    every such case raises ``ValueError`` here instead, as do duplicate column names (they make
    ``df[column]`` a frame instead of a series).
    """
    selected = df[columns]
    duplicated = selected.columns[selected.columns.duplicated()].unique().tolist()
    if duplicated:
        raise ValueError(f"cannot coerce duplicate columns: {duplicated}")
    nulls = selected.isna().sum()
    if nulls.any():
        raise ValueError(f"cannot coerce nulls: {nulls[nulls > 0].to_dict()}")
    converted = {
        column: _to_int64(column, selected[column])
        if DTYPES[column] == "int64"
        else selected[column].astype(DTYPES[column])
        for column in columns
    }
    return pd.DataFrame(converted, index=selected.index)


def coerce_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return ``df`` with exactly ``FRAME_COLUMNS``, in canonical order and canonical dtypes.

    Integers become int64, ``Exposure`` becomes float64, and categoricals become pandas strings.
    Extra columns are dropped, and a missing column raises ``KeyError``. Consumers that want
    pandas ``category`` dtypes (for example a LightGBM pipeline) convert after this call.

    Call ``validate_frame`` on the uncoerced frame first. Coercion checks no bounds or
    categories, but it never loses a value, whatever the source dtype (float, ``object``,
    ``category``, uint64): a null, a non-integral value in an integer column (for example
    ``DrivAge = 35.7``), a value outside int64, or a duplicate column name raises ``ValueError``.
    """
    return _coerce(df, FRAME_COLUMNS)


def coerce_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return ``df[FEATURE_COLUMNS]`` in canonical dtypes: the one way to build model input.

    The API builds the model input from validated request policies with it, and the monitor
    builds it from ``prediction_log`` rows, so every model sees the same columns and dtypes
    (``model.FrequencyModel``). It raises exactly like ``coerce_frame``.
    """
    return _coerce(df, FEATURE_COLUMNS)


# --- Versioning ---------------------------------------------------------------------------------


def _feature_spec() -> list[dict[str, Any]]:
    """Describe each rating factor (name, dtype, bounds or categories) in canonical order."""
    spec: list[dict[str, Any]] = []
    for column in FEATURE_COLUMNS:
        entry: dict[str, Any] = {"name": column, "dtype": DTYPES[column]}
        if column in NUMERIC_BOUNDS:
            entry["bounds"] = list(NUMERIC_BOUNDS[column])
        if column in CATEGORIES:
            entry["categories"] = list(CATEGORIES[column])
        spec.append(entry)
    return spec


def schema_hash() -> str:
    """Return a stable 12-character fingerprint of the feature spec.

    The hash covers feature names, order, dtypes, bounds, and categories, but not the target
    columns or the clean-stage caps, because only the features form the API input. Canonical
    JSON (sorted keys, no whitespace) makes the value identical across runs and platforms. The
    register step compares it with the champion's ``schema_hash`` tag to decide a MAJOR bump.
    """
    payload = json.dumps(_feature_spec(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
