"""Shared evaluation metrics for training (Role B) and monitoring (Role D).

ADR-0002 requires one evaluation module for both, so a metric means the same thing on the
validation split, on the test split, and on live traffic. Every function takes the same three
1-D arrays of equal length:

- ``y_freq``: observed claim frequency, ``ClaimNb / Exposure``.
- ``pred_freq``: predicted claim frequency, strictly positive.
- ``exposure``: policy exposure in years, strictly positive, used as the weight.

Metric names are ``contracts.model.MetricName``; ``with_prefix`` turns them into MLflow keys
such as ``val_poisson_deviance``.
"""

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import auc, mean_poisson_deviance

from claimfreq.contracts.model import MetricName, MetricPrefix


def poisson_deviance(y_freq: ArrayLike, pred_freq: ArrayLike, exposure: ArrayLike) -> float:
    """Exposure-weighted mean Poisson deviance. Lower is better; the selection metric."""
    return float(mean_poisson_deviance(y_freq, pred_freq, sample_weight=exposure))


def gini(y_freq: ArrayLike, pred_freq: ArrayLike, exposure: ArrayLike) -> float:
    """Exposure-weighted Gini coefficient of the Lorenz curve. Higher is better.

    Policies are sorted by predicted frequency, safest first. The Lorenz curve plots the
    cumulative share of exposure (x) against the cumulative share of observed claims (y), from
    an origin at (0, 0). Gini is ``1 - 2 * AUC``: 0 for random ranking, positive when the model
    puts claims in the riskiest policies, negative when the ranking is reversed. This follows
    scikit-learn's Poisson regression example.

    Policies with the same prediction form a single step of the curve. Otherwise the row order
    inside a tie would change the result, and a constant model, which ranks nothing, could
    score far from 0. With ties merged, a constant model scores exactly 0 and shuffling the
    rows does not change the Gini.

    Returns NaN when there are no observed claims, because the curve is undefined.
    """
    pred = np.asarray(pred_freq, dtype=float)
    exposure = np.asarray(exposure, dtype=float)
    claims = np.asarray(y_freq, dtype=float) * exposure
    if claims.sum() == 0:
        return float("nan")
    order = np.argsort(pred, kind="stable")
    # Start index of each run of equal predictions in sorted order; sum each run into one point.
    _, starts = np.unique(pred[order], return_index=True)
    tied_exposure = np.add.reduceat(exposure[order], starts)
    tied_claims = np.add.reduceat(claims[order], starts)
    x = np.concatenate([[0.0], np.cumsum(tied_exposure) / tied_exposure.sum()])
    y = np.concatenate([[0.0], np.cumsum(tied_claims) / tied_claims.sum()])
    return float(1 - 2 * auc(x, y))


def calibration_ratio(y_freq: ArrayLike, pred_freq: ArrayLike, exposure: ArrayLike) -> float:
    """Predicted claims divided by observed claims; 1.0 is perfect.

    Below 1 the portfolio is under-priced: premiums will not cover the claims. In monitoring this
    is the concept-drift signal. Returns NaN when there are no observed claims.
    """
    observed = float(np.dot(y_freq, exposure))
    if observed == 0:
        return float("nan")
    return float(np.dot(pred_freq, exposure)) / observed


def compute_all(y_freq: ArrayLike, pred_freq: ArrayLike, exposure: ArrayLike) -> dict[str, float]:
    """Compute every metric, keyed by its base name (``poisson_deviance``, ``gini``, ...)."""
    return {
        MetricName.POISSON_DEVIANCE.value: poisson_deviance(y_freq, pred_freq, exposure),
        MetricName.GINI.value: gini(y_freq, pred_freq, exposure),
        MetricName.CALIBRATION_RATIO.value: calibration_ratio(y_freq, pred_freq, exposure),
    }


def with_prefix(metrics: dict[str, float], prefix: MetricPrefix) -> dict[str, float]:
    """Prefix metric names for MLflow, e.g. ``with_prefix(m, MetricPrefix.VAL)``."""
    return {f"{prefix}{name}": value for name, value in metrics.items()}
