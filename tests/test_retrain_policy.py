"""Tests for the drift-triggered retraining rule (dags/retrain_policy.py), without Airflow."""

import sys
from pathlib import Path

import pytest

# The DAG folder is not a package; Airflow puts it on sys.path, and so does this test.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dags"))

from retrain_policy import EXIT_CHECK_FAILED, exit_code_from_message, should_retrain  # noqa: E402

from claimfreq.cli import EXIT_CHECK_FAILED as CLI_EXIT_CHECK_FAILED  # noqa: E402


def test_exit_code_matches_the_cli_contract():
    # The Airflow image cannot import claimfreq, so the constant is mirrored; keep them equal.
    assert EXIT_CHECK_FAILED == CLI_EXIT_CHECK_FAILED == 10


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("Docker container failed: {'StatusCode': 10}", 10),
        ("Docker container failed: {'StatusCode': 3, 'Error': None}", 3),
        ("Docker container failed: {'Error': None, 'StatusCode': 137}", 137),
        ("something else entirely", None),
    ],
)
def test_exit_code_from_message(message, expected):
    assert exit_code_from_message(message) == expected


@pytest.mark.parametrize(
    ("exit_code", "enabled", "expected"),
    [
        (10, True, True),  # a failed check: drift or decay detected
        (10, False, False),  # retraining switched off for this run
        (1, True, False),  # a crash: retraining would not fix it
        (3, True, False),  # a stub: nothing to retrain from
        (None, True, False),  # no exit code recorded
    ],
)
def test_should_retrain_only_on_a_failed_check(exit_code, enabled, expected):
    assert should_retrain(exit_code, enabled) is expected
