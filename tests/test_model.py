"""Tests for the model contract: split_xy_w, file_md5, is_better, and the constant baseline."""

import hashlib
import math

import numpy as np
import pytest
from sklearn.base import clone, is_regressor

from claimfreq.contracts.model import (
    METRIC_GOALS,
    REFERENCE_COLUMNS,
    ConstantFrequencyModel,
    MetricName,
    file_md5,
    is_better,
    split_xy_w,
)
from claimfreq.contracts.schema import FEATURE_COLUMNS
from claimfreq.fakes import make_policies


def test_split_xy_w():
    df = make_policies(100)
    X, y, w = split_xy_w(df)
    assert list(X.columns) == FEATURE_COLUMNS
    np.testing.assert_allclose(y * w, df["ClaimNb"])
    np.testing.assert_allclose(w, df["Exposure"])


def test_constant_model_predicts_portfolio_frequency():
    df = make_policies(1000)
    X, y, w = split_xy_w(df)
    model = ConstantFrequencyModel().fit(X, y, sample_weight=w)
    predictions = model.predict(X)
    assert predictions.shape == (len(X),)
    assert predictions.dtype == float
    np.testing.assert_allclose(predictions, df["ClaimNb"].sum() / df["Exposure"].sum())


def test_constant_model_is_a_sklearn_regressor():
    model = ConstantFrequencyModel()
    assert is_regressor(model)
    assert clone(model).get_params() == {}


def test_constant_model_rejects_data_without_claims():
    df = make_policies(10).assign(ClaimNb=0)
    with pytest.raises(ValueError, match="at least one claim"):
        ConstantFrequencyModel().fit(*split_xy_w(df))


def test_reference_columns():
    assert REFERENCE_COLUMNS[: len(FEATURE_COLUMNS)] == FEATURE_COLUMNS
    assert REFERENCE_COLUMNS[len(FEATURE_COLUMNS) :] == [
        "ClaimNb",
        "Exposure",
        "predicted_frequency",
    ]


def test_file_md5_is_the_plain_content_md5(tmp_path):
    small = tmp_path / "abc.txt"
    small.write_bytes(b"abc")
    assert file_md5(small) == "900150983cd24fb0d6963f7d28e17f72"
    train = tmp_path / "train.parquet"
    make_policies(1000).to_parquet(train, index=False)
    assert file_md5(train) == hashlib.md5(train.read_bytes()).hexdigest()


def test_every_metric_has_a_goal():
    assert set(METRIC_GOALS) == set(MetricName)


@pytest.mark.parametrize(
    ("metric", "candidate", "incumbent", "expected"),
    [
        ("poisson_deviance", 0.30, 0.31, True),
        ("poisson_deviance", 0.31, 0.30, False),
        ("gini", 0.31, 0.30, True),
        ("gini", 0.30, 0.31, False),
        ("calibration_ratio", 0.98, 1.05, True),  # 0.02 from 1 beats 0.05 from 1
        ("calibration_ratio", 1.05, 0.98, False),
        (MetricName.CALIBRATION_RATIO, 1.03, 0.95, True),
        ("poisson_deviance", 0.30, 0.30, False),  # a tie is not better
        ("gini", math.nan, 0.30, False),  # NaN is never better
        ("poisson_deviance", np.float64(0.30), np.float64(0.31), True),  # numpy in, bool out
    ],
)
def test_is_better(metric, candidate, incumbent, expected):
    assert is_better(metric, candidate, incumbent) is expected


def test_is_better_rejects_unknown_metrics():
    with pytest.raises(ValueError):
        is_better("rmse", 1.0, 2.0)
