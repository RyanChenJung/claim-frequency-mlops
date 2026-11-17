"""Tests for the shared metrics, with hand-computable cases."""

import math

import numpy as np
import pytest

from claimfreq.contracts.model import ConstantFrequencyModel, MetricPrefix, split_xy_w
from claimfreq.fakes import make_policies, true_frequency
from claimfreq.metrics import (
    calibration_ratio,
    compute_all,
    gini,
    poisson_deviance,
    with_prefix,
)


def test_constant_model_is_perfectly_calibrated():
    X, y, w = split_xy_w(make_policies(2000))
    predictions = ConstantFrequencyModel().fit(X, y, sample_weight=w).predict(X)
    assert calibration_ratio(y, predictions, w) == pytest.approx(1.0)


def test_calibration_ratio_by_hand():
    # Predicted claims 0.2 * 1 + 0.2 * 0.5 = 0.3; observed claims 1 * 1 + 0 = 1.
    assert calibration_ratio([1.0, 0.0], [0.2, 0.2], [1.0, 0.5]) == pytest.approx(0.3)


def test_gini_by_hand():
    # Two policies with equal exposure; only the second has a claim.
    # Correct ranking: Lorenz points (0, 0), (0.5, 0), (1, 1) -> AUC 0.25 -> Gini 0.5.
    assert gini([0.0, 1.0], [0.1, 0.9], [1.0, 1.0]) == pytest.approx(0.5)
    # Reversed ranking: (0, 0), (0.5, 1), (1, 1) -> AUC 0.75 -> Gini -0.5.
    assert gini([0.0, 1.0], [0.9, 0.1], [1.0, 1.0]) == pytest.approx(-0.5)


def loop_gini(y_freq, pred_freq, exposure) -> float:
    """Independent Gini: a plain trapezoid sum over policies sorted by prediction."""
    order = sorted(range(len(pred_freq)), key=lambda i: pred_freq[i])
    total_exposure = sum(exposure)
    total_claims = sum(y * e for y, e in zip(y_freq, exposure, strict=True))
    area = height = 0.0
    for i in order:
        step = y_freq[i] * exposure[i] / total_claims
        area += exposure[i] / total_exposure * (2 * height + step) / 2
        height += step
    return 1 - 2 * area


def test_constant_model_gini_is_exactly_zero():
    X, y, w = split_xy_w(make_policies(2000))
    predictions = ConstantFrequencyModel().fit(X, y, sample_weight=w).predict(X)
    assert gini(y, predictions, w) == 0.0


def test_gini_ignores_row_order_with_ties():
    df = make_policies(5000)
    _, y, w = split_xy_w(df)
    predictions = np.round(true_frequency(df), 2)  # coarse predictions, so many ties
    assert len(np.unique(predictions)) < len(predictions) / 10
    shuffled = np.random.default_rng(0).permutation(len(df))
    assert gini(y[shuffled], predictions[shuffled], w[shuffled]) == pytest.approx(
        gini(y, predictions, w), abs=1e-12
    )


def test_gini_matches_an_independent_computation_without_ties():
    rng = np.random.default_rng(1)
    y, predictions, w = rng.poisson(0.5, 300).astype(float), rng.random(300), rng.random(300)
    assert len(np.unique(predictions)) == len(predictions)
    assert gini(y, predictions, w) == pytest.approx(loop_gini(y, predictions, w), abs=1e-12)


def test_perfect_ranking_beats_reversed_ranking():
    df = make_policies(2000)
    _, y, w = split_xy_w(df)
    assert gini(y, y + 0.01, w) > gini(y, 1 / (y + 0.01), w)


def test_deviance_is_zero_for_exact_predictions():
    y = np.array([0.5, 1.0, 2.0])
    assert poisson_deviance(y, y, np.array([1.0, 0.5, 0.25])) == pytest.approx(0.0)


def test_deviance_is_positive_for_wrong_predictions():
    assert poisson_deviance([0.0, 2.0], [1.0, 1.0], [1.0, 1.0]) > 0


def test_undefined_without_claims():
    assert math.isnan(gini([0.0, 0.0], [0.1, 0.2], [1.0, 1.0]))
    assert math.isnan(calibration_ratio([0.0, 0.0], [0.1, 0.2], [1.0, 1.0]))


def test_compute_all_and_prefix():
    metrics = compute_all([0.0, 1.0], [0.1, 0.9], [1.0, 1.0])
    assert set(metrics) == {"poisson_deviance", "gini", "calibration_ratio"}
    prefixed = with_prefix(metrics, MetricPrefix.VAL)
    assert set(prefixed) == {"val_poisson_deviance", "val_gini", "val_calibration_ratio"}
    assert prefixed["val_gini"] == metrics["gini"]
