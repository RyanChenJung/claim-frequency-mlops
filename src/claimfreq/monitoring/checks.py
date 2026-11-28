"""The five monitoring checks (docs/contracts.md, What the monitor must catch).

| Check | Runs when | Fails when |
|---|---|---|
| rejection_rate | the window has at least one request | rejected / all requests > max |
| data_drift | at least ``min_predictions`` predictions | share of drifted factors >= max |
| prediction_drift | as data_drift | ``predicted_frequency`` drifts, and failing on it is on |
| calibration | at least one prediction has ground truth | ratio outside [min, max] |
| live_deviance | as calibration | live / reference Poisson deviance > max |

A check that cannot run is reported as skipped, and a skipped check never fails the run.
The two drift checks use Evidently. Calibration and deviance use ``claimfreq.metrics``, the same
code that training uses.
"""

import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import pandas as pd

from claimfreq.contracts.model import PREDICTED_FREQUENCY
from claimfreq.contracts.params import MonitorParams
from claimfreq.contracts.schema import (
    CATEGORICAL_FEATURES,
    CLAIM_NB,
    EXPOSURE,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
)
from claimfreq.metrics import calibration_ratio, poisson_deviance
from claimfreq.monitoring.window import WindowData

REJECTION_RATE = "rejection_rate"
DATA_DRIFT = "data_drift"
PREDICTION_DRIFT = "prediction_drift"
CALIBRATION = "calibration"
LIVE_DEVIANCE = "live_deviance"

SKIPPED_TOO_FEW_PREDICTIONS = "skipped (too few predictions)"
SKIPPED_NO_GROUND_TRUTH = "skipped (no ground truth yet)"


class Status(StrEnum):
    """The outcome of one check."""

    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass(frozen=True)
class CheckResult:
    """The outcome of one check, with a short human-readable detail."""

    name: str
    status: Status
    detail: str
    value: float | None = None

    def line(self) -> str:
        """One log line, e.g. ``calibration: failed (ratio 0.71, allowed [0.9, 1.1])``."""
        return f"{self.name}: {self.status} ({self.detail})"


@dataclass
class MonitorReport:
    """The results of every check, in table order, and the Evidently snapshot if one ran."""

    checks: list[CheckResult]
    # The Evidently snapshot of the drift checks; None when they were skipped.
    snapshot: Any = field(default=None, repr=False)

    @property
    def failed(self) -> list[CheckResult]:
        """The checks that ran and failed."""
        return [check for check in self.checks if check.status is Status.FAILED]


def _y_freq(frame: pd.DataFrame) -> pd.Series:
    """Observed claim frequency: claims per policy-year."""
    return frame[CLAIM_NB] / frame[EXPOSURE]


def check_rejection_rate(data: WindowData, params: MonitorParams) -> CheckResult:
    """Fail when too many requests were rejected with HTTP 422."""
    rate = data.n_rejected / data.n_requests
    detail = (
        f"{data.n_rejected} of {data.n_requests} requests rejected, rate {rate:.3f}, "
        f"max {params.rejection_rate_max}"
    )
    status = Status.FAILED if rate > params.rejection_rate_max else Status.PASSED
    return CheckResult(REJECTION_RATE, status, detail, rate)


def _drift_columns(snapshot_dict: dict[str, Any], columns: list[str]) -> dict[str, float]:
    """Return the ``DriftedColumnsCount`` value (count and share) computed on ``columns``."""
    for metric in snapshot_dict["metrics"]:
        config = metric["config"]
        if config["type"].endswith("DriftedColumnsCount") and config["columns"] == columns:
            return metric["value"]
    raise LookupError(f"no DriftedColumnsCount for {columns} in the Evidently snapshot")


def run_drift_report(current: pd.DataFrame, reference: pd.DataFrame) -> Any:
    """Run the Evidently drift report of ``current`` against ``reference``.

    It holds one ``DriftedColumnsCount`` over the rating factors, one over
    ``predicted_frequency``, and a ``ValueDrift`` per column for the dashboard. Evidently picks
    each column's statistical test from its type and the sample size.
    """
    # Imported here, because Evidently is slow to import.
    from evidently import DataDefinition, Dataset, Report
    from evidently.metrics import DriftedColumnsCount, ValueDrift

    columns = [*FEATURE_COLUMNS, PREDICTED_FREQUENCY]
    definition = DataDefinition(
        numerical_columns=[*NUMERIC_FEATURES, PREDICTED_FREQUENCY],
        categorical_columns=CATEGORICAL_FEATURES,
    )
    report = Report(
        [
            DriftedColumnsCount(columns=FEATURE_COLUMNS),
            DriftedColumnsCount(columns=[PREDICTED_FREQUENCY]),
            *(ValueDrift(column=column) for column in columns),
        ]
    )
    return report.run(
        Dataset.from_pandas(current[columns], data_definition=definition),
        Dataset.from_pandas(reference[columns], data_definition=definition),
    )


def check_drift(
    data: WindowData, reference: pd.DataFrame, params: MonitorParams
) -> tuple[CheckResult, CheckResult, Any]:
    """Run the data drift and prediction drift checks. Returns both results and the snapshot."""
    n = len(data.predictions)
    if n < params.min_predictions:
        detail = f"{SKIPPED_TOO_FEW_PREDICTIONS}: {n} < {params.min_predictions}"
        return (
            CheckResult(DATA_DRIFT, Status.SKIPPED, detail),
            CheckResult(PREDICTION_DRIFT, Status.SKIPPED, detail),
            None,
        )

    snapshot = run_drift_report(data.predictions, reference)
    as_dict = snapshot.dict()

    features = _drift_columns(as_dict, FEATURE_COLUMNS)
    share = float(features["share"])
    data_drift = CheckResult(
        DATA_DRIFT,
        Status.FAILED if share >= params.drift_share_max else Status.PASSED,
        f"{int(features['count'])} of {len(FEATURE_COLUMNS)} rating factors drifted, "
        f"share {share:.3f}, fails at >= {params.drift_share_max}",
        share,
    )

    drifted = _drift_columns(as_dict, [PREDICTED_FREQUENCY])["count"] > 0
    if not drifted:
        status, detail = Status.PASSED, "no drift on predicted_frequency"
    elif params.fail_on_prediction_drift:
        status, detail = Status.FAILED, "drift on predicted_frequency"
    else:
        status = Status.PASSED
        detail = "drift on predicted_frequency, ignored (fail_on_prediction_drift is false)"
    prediction_drift = CheckResult(PREDICTION_DRIFT, status, detail, float(drifted))
    return data_drift, prediction_drift, snapshot


def check_calibration(labelled: pd.DataFrame, params: MonitorParams) -> CheckResult:
    """Fail when predicted claims / observed claims is outside the allowed range."""
    if labelled.empty:
        return CheckResult(CALIBRATION, Status.SKIPPED, SKIPPED_NO_GROUND_TRUTH)
    ratio = calibration_ratio(_y_freq(labelled), labelled[PREDICTED_FREQUENCY], labelled[EXPOSURE])
    allowed = f"allowed [{params.calibration_min}, {params.calibration_max}]"
    if math.isnan(ratio):
        # No observed claims, so the ratio is undefined: the prices cannot be checked.
        return CheckResult(
            CALIBRATION,
            Status.FAILED,
            f"no observed claims in {len(labelled)} policies, {allowed}",
            ratio,
        )
    inside = params.calibration_min <= ratio <= params.calibration_max
    return CheckResult(
        CALIBRATION,
        Status.PASSED if inside else Status.FAILED,
        f"ratio {ratio:.3f} on {len(labelled)} policies, {allowed}",
        ratio,
    )


def reference_deviance(reference: pd.DataFrame) -> float:
    """Poisson deviance of the champion on its own reference dataset."""
    return poisson_deviance(_y_freq(reference), reference[PREDICTED_FREQUENCY], reference[EXPOSURE])


def check_live_deviance(
    labelled: pd.DataFrame, reference: pd.DataFrame, params: MonitorParams
) -> CheckResult:
    """Fail when live Poisson deviance is too high compared with the reference."""
    if labelled.empty:
        return CheckResult(LIVE_DEVIANCE, Status.SKIPPED, SKIPPED_NO_GROUND_TRUTH)
    live = poisson_deviance(_y_freq(labelled), labelled[PREDICTED_FREQUENCY], labelled[EXPOSURE])
    baseline = reference_deviance(reference)
    ratio = live / baseline
    return CheckResult(
        LIVE_DEVIANCE,
        Status.FAILED if ratio > params.live_deviance_max_ratio else Status.PASSED,
        f"live {live:.4f} / reference {baseline:.4f} = {ratio:.3f}, "
        f"max {params.live_deviance_max_ratio}",
        ratio,
    )


def run_checks(data: WindowData, reference: pd.DataFrame, params: MonitorParams) -> MonitorReport:
    """Run all five checks on a window that has at least one request."""
    if data.is_empty:
        raise ValueError("run_checks needs a window with at least one request")
    data_drift, prediction_drift, snapshot = check_drift(data, reference, params)
    labelled = data.with_ground_truth
    return MonitorReport(
        checks=[
            check_rejection_rate(data, params),
            data_drift,
            prediction_drift,
            check_calibration(labelled, params),
            check_live_deviance(labelled, reference, params),
        ],
        snapshot=snapshot,
    )
