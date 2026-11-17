"""Tests for the API contract models."""

import math
import uuid
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from claimfreq.contracts.api import (
    MAX_BATCH_SIZE,
    MAX_POLICY_ID,
    ModelInfoResponse,
    ModelRef,
    PolicyInput,
    Prediction,
    PredictRequest,
    PredictResponse,
)
from claimfreq.fakes import make_predict_request

MODEL_INFO = {
    "name": "claim-frequency",
    "version": "3",
    "semver": "1.2.0",
    "alias": "champion",
    "run_id": "0123456789abcdef",
    "trained_at": datetime(2026, 11, 20, 9, 30, tzinfo=UTC),
    "schema_hash": "aab3e74fe93a",
    "data_hash": "900150983cd24fb0d6963f7d28e17f72",
    "git_commit": "82538e9",
    "algorithm": "lightgbm",
    "metrics": {"val_poisson_deviance": 0.31, "val_gini": 0.28},
}


def make_response(request: PredictRequest) -> PredictResponse:
    """Return a valid response to ``request``, with one prediction of 0.1 per policy."""
    return PredictResponse(
        request_id=str(uuid.uuid4()),
        model=ModelRef(name="claim-frequency", version="3", semver="1.2.0"),
        predictions=[
            Prediction(
                prediction_id=str(uuid.uuid4()),
                policy_id=policy.policy_id,
                predicted_frequency=0.1,
            )
            for policy in request.policies
        ],
    )


def test_request_round_trip():
    request = make_predict_request(10)
    assert PredictRequest.model_validate_json(request.model_dump_json()) == request


def test_response_round_trip():
    response = make_response(make_predict_request(3))
    assert PredictResponse.model_validate_json(response.model_dump_json()) == response


def test_batch_size_limits():
    policies = make_predict_request(1).model_dump(mode="json")["policies"]
    PredictRequest(policies=policies * MAX_BATCH_SIZE)
    with pytest.raises(ValidationError):
        PredictRequest(policies=policies * (MAX_BATCH_SIZE + 1))
    with pytest.raises(ValidationError):
        PredictRequest(policies=[])


def test_batch_id_length():
    policies = make_predict_request(1).model_dump(mode="json")["policies"]
    PredictRequest(batch_id="x" * 64, policies=policies)
    with pytest.raises(ValidationError):
        PredictRequest(batch_id="x" * 65, policies=policies)


def test_request_rejects_unknown_fields():
    body = make_predict_request(1).model_dump(mode="json")
    with pytest.raises(ValidationError):
        PredictRequest.model_validate({**body, "model": "other"})
    body["policies"][0]["Colour"] = "red"
    with pytest.raises(ValidationError):
        PredictRequest.model_validate(body)


@pytest.mark.parametrize("frequency", [0.0, math.inf, math.nan])
def test_prediction_must_be_positive_and_finite(frequency):
    with pytest.raises(ValidationError):
        Prediction(prediction_id=str(uuid.uuid4()), policy_id=None, predicted_frequency=frequency)


def test_prediction_id_must_be_a_uuid4():
    with pytest.raises(ValidationError):
        Prediction(prediction_id="not-a-uuid", policy_id=None, predicted_frequency=0.1)


def test_policy_id_fits_a_bigint():
    policy = make_predict_request(1).policies[0].model_dump()
    assert PolicyInput(**{**policy, "policy_id": MAX_POLICY_ID}).policy_id == 2**63 - 1
    with pytest.raises(ValidationError):
        PolicyInput(**{**policy, "policy_id": MAX_POLICY_ID + 1})


def test_model_info_round_trip():
    info = ModelInfoResponse(**MODEL_INFO)
    assert ModelInfoResponse.model_validate_json(info.model_dump_json()) == info


@pytest.mark.parametrize(
    "override",
    [
        {"metrics": {"val_gini": math.nan}},  # the API omits non-finite metrics instead
        {"metrics": {"val_poisson_deviance": math.inf}},
        {"trained_at": datetime(2026, 11, 20, 9, 30)},  # naive
    ],
)
def test_model_info_rejects_non_finite_metrics_and_naive_times(override):
    with pytest.raises(ValidationError):
        ModelInfoResponse(**{**MODEL_INFO, **override})
