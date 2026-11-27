"""Tests for ``claimfreq.results``: the Markdown table behind ``docs/results.md``."""

from types import SimpleNamespace

from claimfreq.cli import EXIT_OK, main
from claimfreq.results import DOES_NOT_BEAT, candidate_table, render


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
