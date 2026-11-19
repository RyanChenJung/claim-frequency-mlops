"""Tests for the CLI contract: command names, stubs, exit codes, and fake-data."""

import subprocess
import sys
from datetime import UTC, datetime

import pandas as pd
import pytest

from claimfreq import cli
from claimfreq.cli import (
    EXIT_CHECK_FAILED,
    EXIT_ERROR,
    EXIT_NOT_IMPLEMENTED,
    EXIT_OK,
    EXIT_USAGE,
    build_parser,
    main,
)
from claimfreq.contracts.api import MAX_BATCH_ID_LENGTH, MAX_POLICY_ID
from claimfreq.contracts.params import SEED_MAX
from claimfreq.contracts.schema import validate_frame
from claimfreq.fakes import make_policies

# (argv, owner) for every stubbed command, with its required arguments. When a role implements
# a command, it removes that command's entries here (docs/contracts.md, Develop against fakes).
STUBS = [
    (["ingest"], "Role A"),
    (["clean"], "Role A"),
    (["split"], "Role A"),
    (["make-drift"], "Role A"),
    (["send-traffic", "--source", "test"], "Role A"),
    (["send-traffic", "--source", "concept_drift", "--batch-id", "run-1"], "Role A"),
    (["monitor"], "Role D"),
    (["monitor", "--since", "2026-11-18T00:00:00+00:00", "--until", "2026-11-19"], "Role D"),
    (["monitor", "--batch-id", "column_swap-1"], "Role D"),
]
# Every command of the CLI contract, stubbed or not.
COMMANDS = {
    "ingest",
    "clean",
    "split",
    "make-drift",
    "train",
    "register",
    "promote",
    "evaluate-test",
    "send-traffic",
    "monitor",
    "fake-data",
}


def test_exit_codes():
    codes = (EXIT_OK, EXIT_ERROR, EXIT_USAGE, EXIT_NOT_IMPLEMENTED, EXIT_CHECK_FAILED)
    assert codes == (0, 1, 2, 3, 10)


def test_help_lists_every_command(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])
    assert excinfo.value.code == EXIT_OK
    out = capsys.readouterr().out
    for command in COMMANDS:
        assert command in out


@pytest.mark.parametrize(("argv", "owner"), STUBS)
def test_stub_exits_3_and_names_its_owner(argv, owner, capsys):
    assert main(argv) == EXIT_NOT_IMPLEMENTED
    err = capsys.readouterr().err
    assert "not implemented yet" in err
    assert owner in err


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["promote"],  # --version is required
        ["send-traffic", "--source", "nope"],
        ["monitor", "--since", "yesterday"],
        ["fake-data"],  # --rows is required
        ["fake-data", "--rows", "0"],
        ["fake-data", "--rows", "10", "--id-start", "0"],
        ["fake-data", "--rows", "10", "--seed", "-1"],
        ["fake-data", "--rows", "10", "--seed", str(SEED_MAX + 1)],
        ["send-traffic", "--source", "test", "--batch-id", "x" * (MAX_BATCH_ID_LENGTH + 1)],
        ["monitor", "--batch-id", "x" * (MAX_BATCH_ID_LENGTH + 1)],
    ],
)
def test_usage_errors_exit_2(argv, capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(argv)
    assert excinfo.value.code == EXIT_USAGE


def test_batch_id_of_maximum_length_is_accepted():
    batch_id = "x" * MAX_BATCH_ID_LENGTH
    for argv in (["send-traffic", "--source", "test"], ["monitor"]):
        assert build_parser().parse_args([*argv, "--batch-id", batch_id]).batch_id == batch_id


def test_fake_data_ids_must_fit_a_policy_id(tmp_path, capsys):
    out = tmp_path / "fake.parquet"
    # The last IDpol, id_start + rows - 1, may equal MAX_POLICY_ID but not exceed it.
    argv = ["fake-data", "--rows", "3", "--out", str(out), "--id-start"]
    assert main([*argv, str(MAX_POLICY_ID - 1)]) == EXIT_USAGE
    assert "the largest policy_id" in capsys.readouterr().err
    assert not out.exists()
    assert main([*argv, str(MAX_POLICY_ID - 2)]) == EXIT_OK
    assert pd.read_parquet(out)["IDpol"].max() == MAX_POLICY_ID


def test_usage_error_message_names_the_expectation(capsys):
    with pytest.raises(SystemExit):
        main(["fake-data", "--rows", "ten"])
    err = capsys.readouterr().err
    assert "expected a positive integer, got 'ten'" in err
    assert "_positive_int" not in err


def test_monitor_window_is_utc_aware():
    args = build_parser().parse_args(
        ["monitor", "--since", "2026-11-18T01:00:00+02:00", "--until", "2026-11-19"]
    )
    # An offset is converted to UTC; a naive value is read as UTC.
    assert args.since == datetime(2026, 11, 17, 23, 0, tzinfo=UTC)
    assert args.until == datetime(2026, 11, 19, 0, 0, tzinfo=UTC)
    assert args.since.tzinfo is UTC and args.until.tzinfo is UTC


def test_unhandled_exception_exits_1_with_traceback(tmp_path, capsys):
    blocker = tmp_path / "file"
    blocker.write_text("not a directory")
    # The output's parent directory cannot be created, so fake-data crashes.
    assert main(["fake-data", "--rows", "5", "--out", str(blocker / "x.parquet")]) == EXIT_ERROR
    assert "Traceback" in capsys.readouterr().err


def test_schema_error_exits_10_without_traceback(monkeypatch, capsys):
    def failing_check(args):
        """Stand-in for a command that lets validate_frame raise on an invalid frame."""
        validate_frame(make_policies(5).drop(columns="DrivAge"), "clean")
        return EXIT_OK

    # build_parser reads the module attribute, so fake-data now runs the failing check.
    monkeypatch.setattr(cli, "_fake_data", failing_check)
    assert main(["fake-data", "--rows", "5"]) == EXIT_CHECK_FAILED
    err = capsys.readouterr().err
    assert "missing column 'DrivAge'" in err
    assert "Traceback" not in err


def test_fake_data_writes_a_valid_parquet(tmp_path):
    out = tmp_path / "nested" / "fake.parquet"
    argv = ["fake-data", "--rows", "100", "--seed", str(SEED_MAX), "--id-start", "900000"]
    assert main([*argv, "--out", str(out)]) == EXIT_OK
    df = pd.read_parquet(out)
    assert len(df) == 100
    assert df["IDpol"].tolist() == list(range(900_000, 900_100))
    validate_frame(df, "clean")


def test_fake_data_defaults_to_data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    assert main(["fake-data", "--rows", "10"]) == EXIT_OK
    assert (tmp_path / "fake" / "policies.parquet").exists()


def test_python_dash_m_works():
    result = subprocess.run(
        [sys.executable, "-m", "claimfreq", "--help"], capture_output=True, text=True
    )
    assert result.returncode == EXIT_OK
    assert "fake-data" in result.stdout
