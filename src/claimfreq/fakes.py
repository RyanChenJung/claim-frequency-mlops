"""Deterministic fake data, so every role can build and test against the contracts on day 1.

Nothing here touches the network, MLflow, or a database. The fakes satisfy the contracts:

- ``make_policies``: a clean-stage policy frame that passes ``validate_frame(stage="clean")``.
- ``TrueFrequencyModel``: a ready-made ``FrequencyModel`` that predicts ``true_frequency``.
- ``make_reference_frame``: a reference dataset with ``REFERENCE_COLUMNS``.
- ``make_predict_request``: a valid ``PredictRequest`` body.

The claim counts follow a known log-linear frequency (``true_frequency``), so a model trained
on fake policies has real signal to learn and should beat the constant baseline.
"""

import numpy as np
import pandas as pd

from claimfreq.contracts.api import MAX_POLICY_ID, PolicyInput, PredictRequest
from claimfreq.contracts.model import (
    PREDICTED_FREQUENCY,
    REFERENCE_COLUMNS,
    FrequencyModel,
)
from claimfreq.contracts.schema import (
    CLAIM_NB,
    DEFAULT_CLAIM_NB_CAP,
    EXPOSURE,
    FEATURE_COLUMNS,
    ID_COLUMN,
    Area,
    Region,
    VehBrand,
    VehGas,
    coerce_features,
    coerce_frame,
)

# Coefficients of the true log frequency. The intercept is tuned so the fake portfolio's
# frequency is about 0.10 claims per policy-year, like the real data (0.1006 after capping).
_INTERCEPT = -3.04
_YOUNG_DRIVER = 0.6  # added for DrivAge < 25
_LOG_BONUS_MALUS = 1.5  # times log(BonusMalus / 50)
_LOG_DENSITY = 0.05  # times log(Density), so urban policies claim more often

# Upper bounds of the Density bins for Area A-E; F takes the rest. In the real data Area is
# also a coarse function of density, which keeps the two features consistent.
_AREA_DENSITY_EDGES = [50, 200, 500, 2_000, 10_000]


def true_frequency(df: pd.DataFrame) -> np.ndarray:
    """Return the known claim frequency that ``make_policies`` draws claims from.

    It rises for young drivers, with bonus-malus, and slightly with density.
    """
    log_frequency = (
        _INTERCEPT
        + _YOUNG_DRIVER * (df["DrivAge"].to_numpy() < 25)
        + _LOG_BONUS_MALUS * np.log(df["BonusMalus"].to_numpy() / 50)
        + _LOG_DENSITY * np.log(df["Density"].to_numpy())
    )
    return np.exp(log_frequency)


class TrueFrequencyModel:
    """A ``FrequencyModel`` that predicts ``true_frequency``; it needs no fitting.

    It stands in for a trained champion: its predictions vary by policy, so monitoring has a
    non-constant reference for prediction drift and Gini, and Role D can serve it before any
    real model exists. ``ConstantFrequencyModel`` remains the baseline every candidate must beat.
    """

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return the true claim frequency of each row of ``X``."""
        return true_frequency(X)


def make_policies(n: int, seed: int = 0, id_start: int = 1) -> pd.DataFrame:
    """Return ``n`` synthetic clean-stage policies, identical for the same arguments.

    Marginals are roughly plausible for French motor policies: most drivers are 25 to 65, about
    half sit at the best bonus-malus of 50, density is log-normal, and a third of policies have
    a full year of exposure. ``ClaimNb`` is Poisson with mean ``true_frequency * Exposure``,
    clipped to the default clean-stage cap.

    ``IDpol`` runs from ``id_start`` to ``id_start + n - 1``. Give fake train, validation, and
    test files different ``id_start`` values so they do not share ids, as real splits do not.
    A range outside [1, ``api.MAX_POLICY_ID``] raises ``ValueError``, because numpy would
    silently wrap it around past int64's maximum.
    """
    last_id = id_start + n - 1
    if id_start < 1 or last_id > MAX_POLICY_ID:
        raise ValueError(f"IDpol range {id_start}..{last_id} is outside 1..{MAX_POLICY_ID}")
    rng = np.random.default_rng(seed)
    drv_age = np.clip(np.round(rng.normal(45, 14, n)), 18, 100)
    bonus_malus = np.where(
        rng.random(n) < 0.5, 50, np.clip(np.round(50 + rng.exponential(25, n)), 50, 230)
    )
    density = np.clip(np.round(np.exp(rng.normal(6, 2, n))), 1, 27_000)
    df = pd.DataFrame(
        {
            # Not np.arange(id_start, id_start + n): a stop past int64's maximum makes it float.
            ID_COLUMN: id_start + np.arange(n),
            CLAIM_NB: 0,  # drawn below, once the features exist
            EXPOSURE: np.where(rng.random(n) < 1 / 3, 1.0, rng.uniform(0.01, 1.0, n)),
            "VehPower": np.clip(4 + rng.poisson(2.5, n), 4, 15),
            "VehAge": np.clip(np.floor(rng.exponential(7, n)), 0, 100),
            "DrivAge": drv_age,
            "BonusMalus": bonus_malus,
            "VehBrand": rng.choice([b.value for b in VehBrand], n),
            "VehGas": rng.choice([g.value for g in VehGas], n),
            "Area": np.array([a.value for a in Area])[np.digitize(density, _AREA_DENSITY_EDGES)],
            "Density": density,
            "Region": rng.choice([r.value for r in Region], n),
        }
    )
    expected_claims = true_frequency(df) * df[EXPOSURE].to_numpy()
    df[CLAIM_NB] = np.minimum(rng.poisson(expected_claims), DEFAULT_CLAIM_NB_CAP)
    return coerce_frame(df)


def make_reference_frame(
    n: int, seed: int = 0, model: FrequencyModel | None = None
) -> pd.DataFrame:
    """Return a fake reference dataset: fake policies plus a model's predictions.

    ``model`` defaults to ``TrueFrequencyModel``, so ``predicted_frequency`` varies by policy
    like a real champion's would. Pass a fitted ``ConstantFrequencyModel`` for a flat reference.
    """
    policies = make_policies(n, seed)
    if model is None:
        model = TrueFrequencyModel()
    reference = policies.assign(**{PREDICTED_FREQUENCY: model.predict(coerce_features(policies))})
    return reference[REFERENCE_COLUMNS]


def make_predict_request(n: int, seed: int = 0) -> PredictRequest:
    """Return a valid ``PredictRequest`` with ``n`` fake policies (at most 1000)."""
    df = make_policies(n, seed)
    # tolist and to_dict return native Python ints, which the strict integer fields require.
    ids = df[ID_COLUMN].tolist()
    features = df[FEATURE_COLUMNS].to_dict("records")
    return PredictRequest(
        batch_id=f"fake-{seed}",
        policies=[PolicyInput(policy_id=i, **f) for i, f in zip(ids, features, strict=True)],
    )
