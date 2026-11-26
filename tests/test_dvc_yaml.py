"""Guard tests for the DVC pipeline, ``dvc.yaml`` (#12; docs/contracts.md, DAGs and Paths).

``dvc repro`` is never run in CI (it downloads OpenML), so these tests check the file itself:
the stage order, each bare ``claimfreq <stage>`` command, that each stage reads the previous
stage's output, the code and params each stage depends on, and that every out is a path from
``contracts/paths.py``. In particular, ``split`` must declare its three files as separate outs,
or ``dvc.lock`` would record a directory hash instead of the md5 that ``data_hash`` carries.
"""

from itertools import pairwise
from pathlib import Path

import pytest
import yaml

from claimfreq.contracts import env, paths
from claimfreq.contracts.params import Params
from claimfreq.contracts.paths import DriftScenario

REPO = Path(__file__).resolve().parents[1]
DVC_YAML = REPO / "dvc.yaml"
# dvc.yaml paths assume the default DATA_DIR, whatever the test environment sets.
DATA = Path(env.DEFAULTS[env.DATA_DIR])
SCHEMA = "src/claimfreq/contracts/schema.py"


def data(relative: Path) -> str:
    """Return a paths.py path as dvc.yaml writes it, under the default DATA_DIR."""
    return (DATA / relative).as_posix()


# The expected pipeline, in run order: each stage's code module, data input, params, and outs.
EXPECTED = {
    "ingest": {
        "module": "src/claimfreq/ingest.py",
        "inputs": [],  # OpenML 41214, outside DVC
        "params": [],
        "outs": [data(paths.RAW)],
    },
    "clean": {
        "module": "src/claimfreq/clean.py",
        "inputs": [data(paths.RAW)],
        "params": ["clean"],
        "outs": [data(paths.PROCESSED)],
    },
    "split": {
        "module": "src/claimfreq/split.py",
        "inputs": [data(paths.PROCESSED)],
        "params": ["seed", "split", "clean"],
        "outs": [data(paths.TRAIN), data(paths.VALIDATION), data(paths.TEST)],
    },
    "make-drift": {
        "module": "src/claimfreq/make_drift.py",
        "inputs": [data(paths.TEST)],
        "params": ["seed", "drift", "clean"],
        "outs": [data(paths.DRIFT_DIR / f"{s}.parquet") for s in DriftScenario],
    },
}


@pytest.fixture(scope="module")
def stages() -> dict[str, dict]:
    """Return the ``stages`` mapping of dvc.yaml, in file order."""
    return yaml.safe_load(DVC_YAML.read_text())["stages"]


def data_deps(stage: dict) -> list[str]:
    """Return a stage's deps that are data files (under DATA_DIR) rather than code."""
    return [d for d in stage.get("deps", []) if Path(d).is_relative_to(DATA)]


def test_stages_are_the_pipeline_in_order(stages):
    assert list(stages) == list(EXPECTED)


@pytest.mark.parametrize("name", EXPECTED)
def test_cmd_is_bare_claimfreq(stages, name):
    # Bare, never `uv run`: inside the trainer image the venv is already on PATH.
    assert stages[name]["cmd"] == f"claimfreq {name}"


@pytest.mark.parametrize("name", EXPECTED)
def test_outs_are_the_paths_contract_files(stages, name):
    # Exact file lists: a splits/ or drift/ directory out would fail here.
    assert stages[name]["outs"] == EXPECTED[name]["outs"]


def test_each_stage_reads_the_previous_stage_output(stages):
    names = list(EXPECTED)
    assert data_deps(stages[names[0]]) == []
    for previous, name in pairwise(names):
        assert data_deps(stages[name]) == EXPECTED[name]["inputs"]
        assert set(data_deps(stages[name])) <= set(stages[previous]["outs"])


@pytest.mark.parametrize("name", EXPECTED)
def test_code_deps_are_the_module_and_the_schema(stages, name):
    code = [d for d in stages[name]["deps"] if d not in data_deps(stages[name])]
    assert code == [EXPECTED[name]["module"], SCHEMA]
    for dep in code:
        assert (REPO / dep).is_file(), f"{name} depends on a missing file: {dep}"


@pytest.mark.parametrize("name", EXPECTED)
def test_stage_has_only_the_expected_keys(stages, name):
    # No frozen, always_changed, wdir, or other stage keys.
    expected = {"cmd", "deps", "outs"} | ({"params"} if EXPECTED[name]["params"] else set())
    assert set(stages[name]) == expected


@pytest.mark.parametrize("name", EXPECTED)
def test_params_are_the_sections_the_stage_reads(stages, name):
    listed = stages[name].get("params", [])
    assert listed == EXPECTED[name]["params"]
    assert set(listed) <= set(Params.model_fields)
