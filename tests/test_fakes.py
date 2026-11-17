"""Tests for the fakes: determinism, validity, and a learnable signal."""

import pandas as pd
import pytest

from claimfreq.contracts.model import REFERENCE_COLUMNS, ConstantFrequencyModel, split_xy_w
from claimfreq.contracts.schema import FEATURE_COLUMNS, validate_frame
from claimfreq.fakes import (
    TrueFrequencyModel,
    make_policies,
    make_predict_request,
    make_reference_frame,
    true_frequency,
)
from claimfreq.metrics import poisson_deviance


def test_make_policies_is_deterministic():
    pd.testing.assert_frame_equal(make_policies(500, seed=3), make_policies(500, seed=3))
    assert not make_policies(500, seed=3).equals(make_policies(500, seed=4))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_make_policies_is_valid(seed):
    validate_frame(make_policies(2000, seed=seed), "clean")


def test_make_policies_id_start_offsets_ids_only():
    shifted = make_policies(100, seed=3, id_start=900_000)
    assert shifted["IDpol"].tolist() == list(range(900_000, 900_100))
    validate_frame(shifted, "clean")
    default = make_policies(100, seed=3)
    pd.testing.assert_frame_equal(shifted.drop(columns="IDpol"), default.drop(columns="IDpol"))


@pytest.mark.parametrize("id_start", [0, 2**63 - 2])
def test_make_policies_rejects_ids_outside_the_policy_id_range(id_start):
    # 2**63 - 2 + 10 - 1 would wrap around past int64's maximum instead of failing.
    with pytest.raises(ValueError, match="IDpol range"):
        make_policies(10, id_start=id_start)


def test_portfolio_frequency_is_realistic():
    df = make_policies(50_000)
    assert df["ClaimNb"].sum() / df["Exposure"].sum() == pytest.approx(0.10, abs=0.01)


def test_true_frequency_beats_the_constant_baseline():
    df = make_policies(20_000)
    X, y, w = split_xy_w(df)
    baseline = ConstantFrequencyModel().fit(X, y, sample_weight=w).predict(X)
    assert poisson_deviance(y, true_frequency(df), w) < poisson_deviance(y, baseline, w)


def test_true_frequency_model_predicts_true_frequency():
    df = make_policies(300)
    predictions = TrueFrequencyModel().predict(df[FEATURE_COLUMNS])
    assert predictions.shape == (300,)
    assert (predictions > 0).all()
    assert predictions == pytest.approx(true_frequency(df))


def test_make_reference_frame():
    reference = make_reference_frame(300)
    assert list(reference.columns) == REFERENCE_COLUMNS
    assert (reference["predicted_frequency"] > 0).all()
    pd.testing.assert_frame_equal(reference, make_reference_frame(300))


def test_make_reference_frame_predictions_vary_by_default():
    reference = make_reference_frame(300)
    assert reference["predicted_frequency"].nunique() > 1
    X, y, w = split_xy_w(make_policies(300))
    constant = ConstantFrequencyModel().fit(X, y, sample_weight=w)
    assert make_reference_frame(300, model=constant)["predicted_frequency"].nunique() == 1


def test_make_predict_request():
    request = make_predict_request(25, seed=2)
    assert len(request.policies) == 25
    assert request.policies[0].policy_id == 1
    assert request == make_predict_request(25, seed=2)
