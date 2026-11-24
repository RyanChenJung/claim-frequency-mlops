"""The prediction API: ``POST /predict``, ``GET /health``, and ``GET /model-info``.

Run it with ``uvicorn claimfreq.serving.app:app --host 0.0.0.0 --port 8000``.

How it works:

- At startup the app connects to ``MONITORING_DB_URL`` and creates the log tables.
- Then it gets a model. Tests pass one to ``create_app``. Otherwise a background thread loads
  ``claim-frequency@champion`` from MLflow, and tries again every 30 seconds until it works.
- Until a model is loaded, ``/health`` returns 503 ``degraded``, and ``/predict`` and
  ``/model-info`` return 503 ``{"detail": "model not loaded"}``.
- ``/predict`` writes one ``prediction_log`` row per policy and commits it **before** it
  returns 200, so a client can write ``ground_truth`` for those rows right away.
- Every 422 keeps FastAPI's default body and is also written to ``rejected_request_log``.

The bodies and paths are fixed in ``claimfreq.contracts.api``.
"""

import json
import logging
import threading
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from sqlalchemy import insert
from sqlalchemy.engine import Engine
from starlette.concurrency import run_in_threadpool

from claimfreq.contracts import env
from claimfreq.contracts.api import (
    MAX_BATCH_ID_LENGTH,
    PATH_HEALTH,
    PATH_MODEL_INFO,
    PATH_PREDICT,
    HealthResponse,
    ModelInfoResponse,
    ModelRef,
    Prediction,
    PredictRequest,
    PredictResponse,
)
from claimfreq.contracts.logs import (
    create_all,
    make_engine,
    prediction_log,
    prediction_log_rows,
    rejected_request_log,
)
from claimfreq.contracts.model import REGISTERED_MODEL_NAME, FrequencyModel
from claimfreq.contracts.schema import FEATURE_COLUMNS, coerce_features
from claimfreq.serving.champion import load_champion

logger = logging.getLogger(__name__)

# How long to wait before trying to load the champion again.
CHAMPION_RETRY_SECONDS = 30.0
MODEL_NOT_LOADED = "model not loaded"
# The value of every lineage field when a model is injected (docs/contracts.md, "The app").
DEV = "dev"


def dev_model_info() -> ModelInfoResponse:
    """Return the identity the app reports for an injected model without a ``model_ref``."""
    return ModelInfoResponse(
        name=REGISTERED_MODEL_NAME,
        version="0",
        semver="0.0.0",
        alias=DEV,
        run_id=DEV,
        trained_at=datetime.now(UTC),
        schema_hash=DEV,
        data_hash=DEV,
        git_commit=DEV,
        algorithm=DEV,
        metrics={},
    )


@dataclass(frozen=True)
class LoadedModel:
    """A model together with the identity that the API reports for it."""

    model: FrequencyModel
    info: ModelInfoResponse


class _State:
    """What the app needs at request time: the database engine and the loaded model.

    ``loaded`` is replaced in one step, so a request always sees a matching model and info.
    """

    def __init__(self) -> None:
        self.engine: Engine | None = None
        self.loaded: LoadedModel | None = None


def _load_champion_until_done(state: _State, stop: threading.Event) -> None:
    """Try to load the champion, then every ``CHAMPION_RETRY_SECONDS`` until it works."""
    tracking_uri = env.get(env.MLFLOW_TRACKING_URI)
    while not stop.is_set():
        try:
            model, info = load_champion(tracking_uri)
        except Exception as error:  # noqa: BLE001 - any failure means "try again later"
            logger.warning(
                "Could not load the champion (%s). Retrying in %.0f s.",
                error,
                CHAMPION_RETRY_SECONDS,
            )
            stop.wait(CHAMPION_RETRY_SECONDS)
            continue
        state.loaded = LoadedModel(model, info)
        logger.info("Loaded %s version %s (semver %s).", info.name, info.version, info.semver)
        return


def _rejected_row(errors: list[dict[str, Any]], body: Any, received_at: datetime) -> dict[str, Any]:
    """Build the ``rejected_request_log`` row for one 422."""
    # The body is a dict when it was valid JSON; otherwise it is raw text or bytes.
    batch_id = body.get("batch_id") if isinstance(body, dict) else None
    if not isinstance(batch_id, str) or len(batch_id) > MAX_BATCH_ID_LENGTH:
        batch_id = None
    policies = body.get("policies") if isinstance(body, dict) else None
    return {
        "request_id": str(uuid.uuid4()),
        "received_at": received_at,
        "batch_id": batch_id,
        "status_code": 422,
        "error_type": str(errors[0]["type"]) if errors else "unknown",
        "error_detail": json.dumps(errors),
        "n_policies": len(policies) if isinstance(policies, list) else None,
    }


def _insert(engine: Engine, table: Any, rows: list[dict[str, Any]]) -> None:
    """Insert ``rows`` into ``table`` and commit."""
    with engine.begin() as connection:
        connection.execute(insert(table), rows)


def create_app(
    model: FrequencyModel | None = None, model_ref: ModelInfoResponse | None = None
) -> FastAPI:
    """Build the API.

    Args:
        model: A model to serve instead of the champion, for tests and local development.
            When it is None, the app loads ``claim-frequency@champion`` from MLflow.
        model_ref: The identity to report for ``model``. When it is None, the app reports the
            dev identity (version ``"0"``, semver ``"0.0.0"``, alias ``dev``). It is only used
            together with ``model``.
    """
    state = _State()

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        state.engine = make_engine(env.get(env.MONITORING_DB_URL))
        # Create the tables before the app can report ready.
        create_all(state.engine)
        stop = threading.Event()
        if model is not None:
            state.loaded = LoadedModel(model, model_ref or dev_model_info())
        else:
            # A daemon thread, so a slow MLflow call never blocks shutdown.
            threading.Thread(
                target=_load_champion_until_done,
                args=(state, stop),
                name="load-champion",
                daemon=True,
            ).start()
        yield
        stop.set()
        state.engine.dispose()

    app = FastAPI(title="Claim frequency API", lifespan=lifespan)

    @app.exception_handler(RequestValidationError)
    async def log_rejected_request(request: Request, error: RequestValidationError) -> Response:
        """Return FastAPI's default 422 body, and write the rejection to the log."""
        received_at = datetime.now(UTC)
        response = await request_validation_exception_handler(request, error)
        row = _rejected_row(jsonable_encoder(error.errors()), error.body, received_at)
        await run_in_threadpool(_insert, state.engine, rejected_request_log, [row])
        return response

    @app.post(PATH_PREDICT, response_model=PredictResponse)
    def predict(request: PredictRequest) -> PredictResponse:
        """Score 1 to 1000 policies, log every prediction, and return them in input order."""
        received_at = datetime.now(UTC)
        loaded = state.loaded
        if loaded is None:
            raise HTTPException(status_code=503, detail=MODEL_NOT_LOADED)

        raw = [
            policy.model_dump(mode="json", include=set(FEATURE_COLUMNS))
            for policy in request.policies
        ]
        features = coerce_features(pd.DataFrame(raw, columns=FEATURE_COLUMNS))
        predicted = np.asarray(loaded.model.predict(features), dtype=float).ravel()
        if len(predicted) != len(request.policies):
            raise RuntimeError(
                f"the model returned {len(predicted)} predictions "
                f"for {len(request.policies)} policies"
            )

        response = PredictResponse(
            request_id=str(uuid.uuid4()),
            model=ModelRef(
                name=loaded.info.name, version=loaded.info.version, semver=loaded.info.semver
            ),
            # Prediction rejects a value that is not finite and > 0, so a bad model gives a 500.
            predictions=[
                Prediction(
                    prediction_id=str(uuid.uuid4()),
                    policy_id=policy.policy_id,
                    predicted_frequency=float(value),
                )
                for policy, value in zip(request.policies, predicted, strict=True)
            ],
        )
        # Commit before returning, so the client can write ground_truth for these rows at once.
        _insert(state.engine, prediction_log, prediction_log_rows(request, response, received_at))
        return response

    @app.get(PATH_HEALTH, response_model=HealthResponse)
    def health() -> JSONResponse:
        """Readiness: 200 once a model is loaded, 503 before."""
        if state.loaded is None:
            body = HealthResponse(status="degraded", model_loaded=False)
            return JSONResponse(status_code=503, content=body.model_dump())
        return JSONResponse(content=HealthResponse(status="ok", model_loaded=True).model_dump())

    @app.get(PATH_MODEL_INFO, response_model=ModelInfoResponse)
    def model_info() -> ModelInfoResponse:
        """The served model version, its lineage, and its metrics."""
        loaded = state.loaded
        if loaded is None:
            raise HTTPException(status_code=503, detail=MODEL_NOT_LOADED)
        return loaded.info

    return app


app = create_app()
"""The app that uvicorn serves. It loads the champion from MLflow at startup."""
