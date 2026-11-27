"""Tests for ``claimfreq.results``: the Markdown table behind ``docs/results.md``."""

from types import SimpleNamespace

import pytest

from claimfreq.cli import EXIT_OK, main
from claimfreq.results import DOES_NOT_BEAT, candidate_table, render
from claimfreq.results import main as results_main


def _run(run_id: str, algorithm: str, deviance: float) -> SimpleNamespace:
    """A stand-in for an MLflow Run with only the fields the table reads."""
    return SimpleNamespace(
        info=SimpleNamespace(run_id=run_id, run_name=algorithm),
        data=SimpleNamespace(
            tags={"algorithm": algorithm},
            metrics={"val_poisson_deviance": deviance, "val_gini": 0.1},
        ),
    )


def test_table_flags_candidates_that_do_not_beat_the_constant():
    runs = [_run("a", "constant", 0.46), _run("b", "poisson_glm", 0.45), _run("c", "lgbm", 0.47)]
    rows = candidate_table(runs, "poisson_deviance")[2:]
    assert rows[0].startswith("| constant |") and rows[0].endswith("| baseline |")
    assert "**poisson_glm** (best)" in rows[1] and rows[1].endswith("| yes |")
    assert rows[2].endswith(f"| {DOES_NOT_BEAT} |")
    # A missing metric is shown, not crashed on.
    assert "n/a" in rows[0]


def test_a_tie_with_the_constant_does_not_beat_it():
    runs = [_run("a", "constant", 0.46), _run("b", "poisson_glm", 0.46)]
    assert candidate_table(runs, "poisson_deviance")[-1].endswith(f"| {DOES_NOT_BEAT} |")


def test_report_before_promotion_has_no_champion(workspace):
    assert main(["train"]) == EXIT_OK
    report = render()
    assert "| constant |" in report and "| **poisson_glm** (best) |" in report
    assert "No champion yet" in report


def test_report_after_evaluate_test_shows_the_champions_test_metrics(workspace):
    for argv in (["train"], ["register"], ["promote", "--version", "1"]):
        assert main(argv) == EXIT_OK
    assert "Not evaluated yet" in render()
    assert main(["evaluate-test"]) == EXIT_OK
    report = render()
    assert "version 1 (1.0.0, poisson_glm)" in report
    assert "| test |" in report and "| val |" in report


def test_report_warns_when_the_champion_is_from_an_older_candidate_set(workspace):
    for argv in (["train"], ["register"], ["promote", "--version", "1"]):
        assert main(argv) == EXIT_OK
    assert "**Note:**" not in render()
    assert main(["train"]) == EXIT_OK  # a newer candidate set, e.g. a drift-triggered retrain
    report = render()
    assert "**Note:** the champion comes from candidate set" in report


def test_output_is_written_and_a_failure_keeps_the_old_file(
    workspace, tmp_path, capsys, monkeypatch
):
    out = tmp_path / "docs" / "results.md"
    # Nothing trained yet: exit 10 with a results: message, and no file.
    assert results_main(["--output", str(out)]) == 10
    err = capsys.readouterr().err
    assert err.startswith("results: ") and "register:" not in err
    assert not out.exists()
    assert main(["train"]) == EXIT_OK
    assert results_main(["--output", str(out)]) == 0
    first = out.read_text()
    assert first.startswith("# Results")
    # A crash while writing (here: the text cannot be encoded) leaves the old report intact.
    monkeypatch.setattr("claimfreq.results.render", lambda: "# Results \ud800\n")
    with pytest.raises(UnicodeEncodeError):
        results_main(["--output", str(out)])
    assert out.read_text() == first
    assert list(out.parent.iterdir()) == [out]  # no temporary file left behind
