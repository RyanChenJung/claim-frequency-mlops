"""CLI contract: every ``claimfreq`` command, with its final name, arguments, and exit codes.

The Airflow DAGs (Role C) and DVC stages (Role A) call these commands inside the ``trainer``
image, so the command line is the only interface they depend on (ADR-0003). Each command
already exists with its final arguments. Commands not implemented yet are stubs that exit with
``EXIT_NOT_IMPLEMENTED`` until their owner replaces the stub with a real implementation, without
changing the command line; ``tests/test_cli.py`` lists the remaining stubs in ``STUBS``.

Exit codes (any nonzero code makes Airflow mark the task failed):

- ``0`` OK.
- ``1`` unexpected error: any unhandled exception. ``main`` prints the traceback to stderr;
  Python's own default for a crash is also 1.
- ``2`` usage error (argparse's default for bad arguments).
- ``3`` not implemented yet.
- ``10`` a check failed (data validation or a monitoring test), kept distinct from a crash.
  ``main`` turns an uncaught ``schema.SchemaError`` into 10 and prints its message without a
  traceback, so a command can simply let ``validate_frame`` raise.

Every command runs with the repo root as the working directory (docs/contracts.md, Runtime).
Run it as ``claimfreq <command>`` or ``python -m claimfreq <command>``: bare inside an image,
behind ``uv run`` on the host (docs/contracts.md, Runtime).
"""

import argparse
import sys
import traceback
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from claimfreq.contracts import paths

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_USAGE = 2
EXIT_NOT_IMPLEMENTED = 3
EXIT_CHECK_FAILED = 10

ROLE_DATA = "Role A (Data)"
ROLE_MODELING = "Role B (Modeling)"
ROLE_SERVING = "Role D (Serving and monitoring)"

Handler = Callable[[argparse.Namespace], int]

# Which `monitor` checks run, and what a window without data means (docs/contracts.md, What the
# monitor must catch). Shown by `claimfreq monitor --help`.
MONITOR_SEMANTICS = (
    "No data: no row in prediction_log or in rejected_request_log for the window (and the "
    "--batch-id filter); monitor logs 'no data in window' and exits 0. Rejection rate runs "
    "whenever the window has at least one request. Data drift and prediction drift run only "
    "with at least monitor.min_predictions predictions; otherwise they are reported as "
    "'skipped (too few predictions)'. Calibration and live deviance run only on the "
    "predictions that have ground truth; with none, they are reported as 'skipped (no ground "
    "truth yet)'. A skipped check never fails the run: exit 10 if any check that ran failed, "
    "else 0."
)


def _not_implemented(owner: str) -> Handler:
    """Return a stub handler that names the command's owner and exits 3."""

    def handler(args: argparse.Namespace) -> int:
        print(
            f"{args.command}: not implemented yet — owner: {owner}; see docs/contracts.md",
            file=sys.stderr,
        )
        return EXIT_NOT_IMPLEMENTED

    return handler


def _ingest(args: argparse.Namespace) -> int:
    """Download OpenML 41214, check it, and write the raw-stage parquet file."""
    # Imported here so `--help` and the stubs stay fast and need no pandas or scikit-learn.
    from claimfreq.ingest import CountCheckError, ingest

    try:
        out, n_rows = ingest()
    except CountCheckError as error:
        # Wrong totals are a failed check (exit 10), like a SchemaError, not a crash.
        print(error, file=sys.stderr)
        return EXIT_CHECK_FAILED
    print(f"ingest: wrote {n_rows} policies to {out}")
    return EXIT_OK


def _fake_data(args: argparse.Namespace) -> int:
    """Write schema-valid synthetic clean-stage policies to a parquet file."""
    # Imported here so `--help` and the stubs stay fast and need no pandas or scikit-learn.
    from claimfreq.contracts.api import MAX_POLICY_ID
    from claimfreq.fakes import make_policies

    # Every IDpol must fit the API's policy_id. Checked here, because it depends on two options.
    if args.id_start + args.rows - 1 > MAX_POLICY_ID:
        print(
            f"fake-data: error: --id-start + --rows - 1 must be at most {MAX_POLICY_ID}, "
            "the largest policy_id",
            file=sys.stderr,
        )
        return EXIT_USAGE
    out = args.out or paths.data_path(paths.FAKE)
    out.parent.mkdir(parents=True, exist_ok=True)
    policies = make_policies(args.rows, seed=args.seed, id_start=args.id_start)
    policies.to_parquet(out, index=False)
    print(f"wrote {args.rows} fake policies to {out}")
    return EXIT_OK


def _train(args: argparse.Namespace) -> int:
    """Train every candidate and log one parent MLflow run with a child run per candidate."""
    # Imported here so `--help` and the stubs stay fast and need no mlflow.
    from claimfreq.training import train

    parent_run_id = train()
    print(f"train: logged candidate set {parent_run_id}")
    return EXIT_OK


# argparse types raise ArgumentTypeError: for any other error, argparse would print the
# function's name ("invalid _positive_int value") instead of the message.


def _positive_int(value: str) -> int:
    """argparse type for a count or an id that must be at least 1."""
    try:
        number = int(value)
        if number >= 1:
            return number
    except ValueError:
        pass
    raise argparse.ArgumentTypeError(f"expected a positive integer, got {value!r}")


def _seed(value: str) -> int:
    """argparse type for a seed, in the same range as the ``seed`` fields of params.yaml."""
    # Imported here, not at the top, so `--help` and the stubs need no pydantic or sklearn.
    from claimfreq.contracts.params import SEED_MAX

    try:
        number = int(value)
        if 0 <= number <= SEED_MAX:
            return number
    except ValueError:
        pass
    raise argparse.ArgumentTypeError(f"expected an integer from 0 to {SEED_MAX}, got {value!r}")


def _batch_id(value: str) -> str:
    """argparse type for a batch id: at most ``api.MAX_BATCH_ID_LENGTH`` characters."""
    # Imported here, not at the top, so `--help` and the stubs need no pydantic or pandas.
    from claimfreq.contracts.api import MAX_BATCH_ID_LENGTH

    if len(value) <= MAX_BATCH_ID_LENGTH:
        return value
    raise argparse.ArgumentTypeError(
        f"expected at most {MAX_BATCH_ID_LENGTH} characters, got {len(value)}"
    )


def _utc_datetime(value: str) -> datetime:
    """argparse type for an ISO 8601 time, returned timezone-aware in UTC.

    A naive value (no offset) is read as UTC, so a DAG and a developer get the same window.
    """
    try:
        moment = datetime.fromisoformat(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected an ISO 8601 datetime, got {value!r}") from None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return moment.astimezone(UTC)


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with every subcommand of the CLI contract."""
    parser = argparse.ArgumentParser(
        prog="claimfreq",
        description="Claim-frequency MLOps pipeline. Data paths are relative to DATA_DIR "
        "(default: data); tunable values come from params.yaml.",
    )
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    def add(
        name: str, help_text: str, handler: Handler, epilog: str | None = None
    ) -> argparse.ArgumentParser:
        command = commands.add_parser(name, help=help_text, description=help_text, epilog=epilog)
        command.set_defaults(handler=handler)
        return command

    # Role A (Data): DVC stages.
    add(
        "ingest",
        "Download OpenML dataset 41214 and write raw/freMTPL2freq.parquet (raw stage).",
        _ingest,
    )
    add(
        "clean",
        "Read raw/, cap ClaimNb and Exposure, and write processed/policies.parquet.",
        _not_implemented(ROLE_DATA),
    )
    add(
        "split",
        "Read processed/ and write splits/train, splits/validation, and splits/test parquet.",
        _not_implemented(ROLE_DATA),
    )
    add(
        "make-drift",
        "Read splits/test.parquet and write all five drift/<scenario>.parquet files.",
        _not_implemented(ROLE_DATA),
    )

    # Role B (Modeling): Airflow tasks, plus promote (run by a human).
    add(
        "train",
        "Train candidates on splits/train, evaluate on splits/validation, and log each as a "
        "child run of one parent MLflow run per invocation.",
        _train,
    )
    add(
        "register",
        "Register the best candidate of the latest train run by selection_metric with alias "
        "'challenger', its tags, and the reference dataset, and log its comparison with the "
        "'champion'.",
        _not_implemented(ROLE_MODELING),
    )
    promote = add(
        "promote",
        "Move the 'champion' alias to a registered model version (a human decision).",
        _not_implemented(ROLE_MODELING),
    )
    promote.add_argument(
        "--version", type=_positive_int, required=True, metavar="N", help="registry version"
    )
    add(
        "evaluate-test",
        "Evaluate the champion on splits/test and log test_* metrics.",
        _not_implemented(ROLE_MODELING),
    )

    # Role A (Data) owns send-traffic: it replays Role A's test and drift files through the API.
    send_traffic = add(
        "send-traffic",
        "POST a data source to the API in batches of traffic.batch_size, then write "
        "ground_truth for the scored rows. Exit 0 when every batch got HTTP 200 or 422 (the "
        "counts are printed); exit 1 on connection errors or 5xx after retries.",
        _not_implemented(ROLE_DATA),
    )
    send_traffic.add_argument(
        "--source",
        required=True,
        choices=["test", *(scenario.value for scenario in paths.DriftScenario)],
        help="'test' for splits/test.parquet, or a drift scenario for drift/<scenario>.parquet",
    )
    send_traffic.add_argument(
        "--batch-id",
        type=_batch_id,
        metavar="ID",
        help="id of this run, sent as batch_id with every request (at most 64 chars; default: "
        "<source>-<UTC time>, e.g. portfolio_shift-20261201T153000Z)",
    )
    monitor = add(
        "monitor",
        "Compare logged predictions, ground truth, and rejections with the champion's "
        "reference dataset; push results to the Evidently UI; exit 10 if any check that ran "
        "failed and post to ALERT_WEBHOOK_URL if set.",
        _not_implemented(ROLE_SERVING),
        epilog=MONITOR_SEMANTICS,
    )
    monitor.add_argument(
        "--since",
        type=_utc_datetime,
        metavar="ISO",
        help="start of the received_at window, inclusive (ISO 8601; naive means UTC; "
        "default: unbounded)",
    )
    monitor.add_argument(
        "--until",
        type=_utc_datetime,
        metavar="ISO",
        help="end of the received_at window, exclusive (ISO 8601; naive means UTC; "
        "default: unbounded)",
    )
    monitor.add_argument(
        "--batch-id",
        type=_batch_id,
        metavar="ID",
        help="only the requests of the send-traffic run with this batch_id (default: all runs)",
    )

    # Contracts: works now.
    fake_data = add(
        "fake-data",
        "Write schema-valid synthetic clean-stage policies to a parquet file.",
        _fake_data,
    )
    fake_data.add_argument(
        "--rows", type=_positive_int, required=True, metavar="N", help="number of policies"
    )
    fake_data.add_argument(
        "--seed", type=_seed, default=0, metavar="S", help="random seed, 0 to 2**32 - 1"
    )
    fake_data.add_argument(
        "--id-start",
        type=_positive_int,
        default=1,
        metavar="ID",
        help="first IDpol (default: 1); give each fake split its own range of ids",
    )
    fake_data.add_argument(
        "--out",
        type=Path,
        metavar="PATH",
        help="output parquet path (default: DATA_DIR/fake/policies.parquet)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse ``argv`` (default: ``sys.argv[1:]``), run the command, and return its exit code.

    A usage error exits 2 through argparse's ``SystemExit``. An uncaught ``SchemaError`` is a
    failed check: its message (every violation) goes to stderr without a traceback, and the
    exit code is ``EXIT_CHECK_FAILED`` (10). Any other exception is a crash: its traceback goes
    to stderr and the exit code is ``EXIT_ERROR`` (1), never 10, so a bug is not mistaken for a
    failed check.
    """
    args = build_parser().parse_args(argv)
    try:
        return args.handler(args)
    except Exception as error:
        # Imported here, like the handlers' own imports, so `--help` and the stubs need no
        # pandas. Whatever raised a SchemaError has already imported the module.
        from claimfreq.contracts.schema import SchemaError

        if isinstance(error, SchemaError):
            print(error, file=sys.stderr)
            return EXIT_CHECK_FAILED
        traceback.print_exc()
        return EXIT_ERROR
