"""Tests for the data schema contract: frame validation, PolicyFeatures, and schema_hash."""

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from claimfreq.contracts import schema
from claimfreq.contracts.params import Params
from claimfreq.contracts.schema import (
    CATEGORIES,
    DTYPES,
    FEATURE_COLUMNS,
    FRAME_COLUMNS,
    NUMERIC_BOUNDS,
    NUMERIC_FEATURES,
    PolicyFeatures,
    SchemaError,
    coerce_features,
    coerce_frame,
    find_violations,
    schema_hash,
    validate_frame,
)
from claimfreq.fakes import make_policies

# Update deliberately, together with SCHEMA_VERSION, when the feature spec changes.
EXPECTED_SCHEMA_HASH = "aab3e74fe93a"

VALID_POLICY = {
    "VehPower": 6,
    "VehAge": 3,
    "DrivAge": 35,
    "BonusMalus": 50,
    "VehBrand": "B12",
    "VehGas": "Regular",
    "Area": "C",
    "Density": 1200,
    "Region": "R11",
}


@pytest.fixture
def policies() -> pd.DataFrame:
    """200 valid clean-stage fake policies."""
    return make_policies(200, seed=1)


def checks(df: pd.DataFrame, stage: schema.Stage = "clean") -> set[tuple[str, str]]:
    """Return the (check, column) pairs of every violation in ``df``."""
    return {(v.check, v.column) for v in find_violations(df, stage)}


# --- Frame validation ---------------------------------------------------------------------------


def test_fake_policies_are_valid_in_both_stages(policies):
    validate_frame(policies, "clean")
    validate_frame(policies, "raw")


def test_column_constants_are_consistent():
    assert sorted(NUMERIC_FEATURES + list(CATEGORIES)) == sorted(FEATURE_COLUMNS)
    assert len(FRAME_COLUMNS) == 12


def test_missing_and_extra_columns(policies):
    df = policies.drop(columns="DrivAge").assign(Foo=1)
    assert checks(df) == {("missing_column", "DrivAge"), ("extra_column", "Foo")}


def test_wrong_dtype_class(policies):
    df = policies.assign(DrivAge=policies["DrivAge"].astype(str))
    assert checks(df) == {("dtype", "DrivAge")}


def test_nulls(policies):
    df = policies.copy()
    df.loc[0, "Exposure"] = np.nan
    df.loc[1, "Region"] = None
    assert checks(df) == {("null", "Exposure"), ("null", "Region")}


@pytest.mark.parametrize("missing", [None, np.nan])
def test_object_string_column_with_nulls_has_no_dtype_violation(policies, missing):
    df = policies.astype({"Region": object})
    df.loc[0, "Region"] = missing
    df.loc[1, "Region"] = "R99"  # the unknown-category check must still run
    assert df["Region"].dtype == object
    assert checks(df) == {("null", "Region"), ("unknown_category", "Region")}


def test_duplicate_column_is_reported_alone(policies):
    df = pd.concat([policies, policies[["DrivAge"]]], axis=1)
    assert checks(df) == {("duplicate_column", "DrivAge")}


def test_out_of_range_counts_every_row(policies):
    df = policies.copy()
    df.loc[[0, 1, 2], "DrivAge"] = 250
    df.loc[3, "BonusMalus"] = -1
    df.loc[4, "Exposure"] = 0.0  # exposure must be strictly positive
    violations = {v.column: v.count for v in find_violations(df, "clean")}
    assert violations == {"DrivAge": 3, "BonusMalus": 1, "Exposure": 1}


def test_unknown_category(policies):
    df = policies.copy()
    df.loc[0, "VehGas"] = "'Regular'"  # quoted raw value that ingest must strip
    df.loc[1, "Area"] = "G"
    assert checks(df) == {("unknown_category", "VehGas"), ("unknown_category", "Area")}


def test_duplicate_id(policies):
    df = policies.copy()
    df.loc[1, "IDpol"] = df.loc[0, "IDpol"]
    assert checks(df) == {("duplicate_id", "IDpol")}


def test_clean_caps_apply_only_to_clean_stage(policies):
    df = policies.copy()
    df.loc[0, "ClaimNb"] = 16
    df.loc[1, "Exposure"] = 2.01
    assert find_violations(df, "raw") == []
    assert checks(df, "clean") == {("out_of_range", "ClaimNb"), ("out_of_range", "Exposure")}


def test_clean_caps_come_from_params(policies):
    df = policies.copy()
    df.loc[0, "ClaimNb"] = 5
    params = Params.model_validate({"clean": {"claim_nb_cap": 5}})
    assert find_violations(df, "clean", params) == []


def test_validate_frame_collects_all_violations(policies):
    df = policies.drop(columns="Region").assign(DrivAge=10)
    with pytest.raises(SchemaError) as excinfo:
        validate_frame(df, "clean")
    assert {(v.check, v.column) for v in excinfo.value.violations} == {
        ("missing_column", "Region"),
        ("out_of_range", "DrivAge"),
    }
    assert "2 schema violation(s)" in str(excinfo.value)


def test_openml_like_frame_validates_as_read_then_coerces(policies):
    # Mimic the OpenML download: float ids and counts, category columns, shuffled order.
    raw = policies.astype({"IDpol": "float64", "ClaimNb": "float64", "Area": "category"})
    raw = raw[list(reversed(raw.columns))]
    # Validate first: float columns of whole numbers pass the int64 dtype check.
    assert find_violations(raw, "raw") == []
    coerced = coerce_frame(raw)
    assert list(coerced.columns) == FRAME_COLUMNS
    assert coerced["IDpol"].dtype == "int64"
    assert pd.api.types.is_string_dtype(coerced["Area"])
    pd.testing.assert_frame_equal(coerced, policies)


def test_coerce_features_returns_model_input(policies):
    features = coerce_features(policies.astype({"DrivAge": "float64", "Region": "category"}))
    assert list(features.columns) == FEATURE_COLUMNS
    assert {c: str(features[c].dtype) for c in FEATURE_COLUMNS} == {
        c: DTYPES[c] for c in FEATURE_COLUMNS
    }
    pd.testing.assert_frame_equal(features, policies[FEATURE_COLUMNS])


@pytest.mark.parametrize("coerce", [coerce_frame, coerce_features])
def test_coercion_never_truncates_a_float(policies, coerce):
    df = policies.astype({"DrivAge": "float64"})
    df.loc[0, "DrivAge"] = 35.7
    # Validating first reports it in both stages; coercing would otherwise give 35.
    assert checks(df, "raw") == {("dtype", "DrivAge")}
    assert checks(df) == {("dtype", "DrivAge")}
    with pytest.raises(ValueError, match="DrivAge"):
        coerce(df)


def test_null_in_a_float_integer_column_is_only_a_null(policies):
    df = policies.astype({"ClaimNb": "float64"})
    df.loc[0, "ClaimNb"] = np.nan
    assert checks(df, "raw") == {("null", "ClaimNb")}


def test_clean_stage_requires_integer_dtypes(policies):
    # Clean files are posted as is by send-traffic, and the strict API rejects 34.0.
    df = policies.astype({"DrivAge": "float64"})
    assert checks(df, "raw") == set()
    assert checks(df) == {("dtype", "DrivAge")}


@pytest.mark.parametrize(
    ("dtype", "value"),
    [("float64", 1e20), ("float64", 2.0**63), ("uint64", 2**63)],
)
def test_raw_ids_outside_int64_are_a_dtype_violation(policies, dtype, value):
    # Without this, validation would pass a frame that coerce_frame then refuses.
    ids = policies["IDpol"].astype(object)
    ids[0] = value
    df = policies.assign(IDpol=ids.astype(dtype))
    assert checks(df, "raw") == {("dtype", "IDpol")}


def test_pyarrow_backed_numbers_validate_and_coerce(policies):
    raw = policies.astype({"IDpol": "double[pyarrow]", "DrivAge": "int64[pyarrow]"})
    assert find_violations(raw, "raw") == []
    pd.testing.assert_frame_equal(coerce_frame(raw), policies)


@pytest.mark.parametrize("coerce", [coerce_frame, coerce_features])
@pytest.mark.parametrize(
    ("column", "dtype", "value"),
    [
        ("DrivAge", "object", 35.7),  # numbers held as Python objects
        ("VehAge", "category", 1.5),  # a category of numbers
        ("Density", "float64", 1e20),  # whole, but astype would saturate it at int64's maximum
        ("BonusMalus", "uint64", 2**63),  # astype would wrap it around to -2**63
    ],
)
def test_coercion_never_loses_a_value_whatever_the_dtype(policies, coerce, column, dtype, value):
    values = policies[column].astype(object)
    values[0] = value
    df = policies.assign(**{column: values.astype(dtype)})
    with pytest.raises(ValueError, match=column):
        coerce(df)


@pytest.mark.parametrize("coerce", [coerce_frame, coerce_features])
def test_coercion_rejects_nulls(policies, coerce):
    df = policies.copy()
    df.loc[0, "Region"] = None
    with pytest.raises(ValueError, match="Region"):
        coerce(df)


@pytest.mark.parametrize("coerce", [coerce_frame, coerce_features])
def test_coercion_rejects_duplicate_columns(policies, coerce):
    df = pd.concat([policies, policies[["DrivAge"]]], axis=1)
    with pytest.raises(ValueError, match="duplicate columns: \\['DrivAge'\\]"):
        coerce(df)


# --- PolicyFeatures (API validation) ------------------------------------------------------------


def test_valid_policy_is_accepted():
    policy = PolicyFeatures(**VALID_POLICY)
    assert policy.model_dump(mode="json") == VALID_POLICY


def test_policy_fields_match_feature_columns():
    assert list(PolicyFeatures.model_fields) == FEATURE_COLUMNS


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("DrivAge", 17),  # out of range
        ("BonusMalus", -50),  # out of range
        ("Area", "G"),  # unknown category
        ("DrivAge", "35"),  # string-typed int
        ("DrivAge", 35.0),  # float-typed int
    ],
)
def test_invalid_value_is_rejected(field, value):
    with pytest.raises(ValidationError):
        PolicyFeatures(**{**VALID_POLICY, field: value})


def test_extra_and_missing_fields_are_rejected():
    renamed = {k: v for k, v in VALID_POLICY.items() if k != "DrivAge"} | {"DriverAge": 35}
    with pytest.raises(ValidationError) as excinfo:
        PolicyFeatures(**renamed)
    assert {e["type"] for e in excinfo.value.errors()} == {"missing", "extra_forbidden"}


def test_json_string_int_is_rejected():
    payload = '{"VehPower": 6, "VehAge": 3, "DrivAge": "35", "BonusMalus": 50, "VehBrand": "B12",'
    payload += ' "VehGas": "Regular", "Area": "C", "Density": 1200, "Region": "R11"}'
    with pytest.raises(ValidationError):
        PolicyFeatures.model_validate_json(payload)


def test_pydantic_bounds_equal_schema_constants():
    """Single source of truth: the API bounds are exactly NUMERIC_BOUNDS and CATEGORIES."""
    for column, (low, high) in NUMERIC_BOUNDS.items():
        metadata = PolicyFeatures.model_fields[column].metadata
        assert [m.ge for m in metadata if hasattr(m, "ge")] == [low]
        assert [m.le for m in metadata if hasattr(m, "le")] == [high]
        assert any(getattr(m, "strict", False) for m in metadata)
    for column, allowed in CATEGORIES.items():
        enum = PolicyFeatures.model_fields[column].annotation
        assert [member.value for member in enum] == allowed


# --- Versioning ---------------------------------------------------------------------------------


def test_schema_hash_is_pinned():
    assert schema_hash() == EXPECTED_SCHEMA_HASH
    assert schema.SCHEMA_VERSION == "1.0.0"


def test_schema_hash_changes_with_the_spec(monkeypatch):
    monkeypatch.setitem(schema.NUMERIC_BOUNDS, "DrivAge", (17, 100))
    assert schema_hash() != EXPECTED_SCHEMA_HASH


def test_schema_hash_changes_with_feature_order(monkeypatch):
    monkeypatch.setattr(schema, "FEATURE_COLUMNS", list(reversed(FEATURE_COLUMNS)))
    assert schema_hash() != EXPECTED_SCHEMA_HASH
