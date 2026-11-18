"""When the `monitor` DAG retrains automatically (stretch goal; docs/contracts.md, DAGs).

Standard library only, so pytest checks it on the host without Airflow.

The rule: retrain only when `claimfreq monitor` exited 10 ("check failed", i.e. drift or
decay was detected) and the run's ``retrain_on_drift`` param is on. A crash (1), a usage
error (2), or a stub (3) is a platform problem, and retraining would not fix it. Promotion
stays a human decision (ADR-0007): retraining only produces a new challenger.
"""

from __future__ import annotations

import re

# Exit code of a failed check (docs/contracts.md, CLI). Mirrors cli.EXIT_CHECK_FAILED, which
# the Airflow image cannot import.
EXIT_CHECK_FAILED = 10

# XCom key under which the monitor task records its container's exit code.
EXIT_CODE_XCOM_KEY = "exit_code"

# DockerOperator reports a failed container as "Docker container failed: {'StatusCode': N,
# ...}"; the exception carries no separate code attribute.
_STATUS_CODE = re.compile(r"'StatusCode':\s*(-?\d+)")


def exit_code_from_message(message: str) -> int | None:
    """Return the container exit code in a DockerOperator failure message, or None."""
    match = _STATUS_CODE.search(message)
    return int(match.group(1)) if match else None


def should_retrain(exit_code: int | None, enabled: bool) -> bool:
    """Return True only for a failed check (exit 10) with retraining enabled."""
    return bool(enabled) and exit_code == EXIT_CHECK_FAILED
