"""Guard tests for the committed ``dvc.lock`` (#12; docs/contracts.md, Model -> data_hash).

``dvc.lock`` records the md5 of every stage out. ``train`` tags each model with
``data_hash = model.file_md5(train.parquet)``, and the two must agree, so a model can be traced
to the data version a git commit points to.

- The sync tests run everywhere, CI included, and need no data. Every out in ``dvc.yaml`` has a
  plain md5 in ``dvc.lock``, and each stage's locked cmd, deps, params values, and code md5s
  equal the committed ``dvc.yaml``, ``params.yaml``, and source files. So a change committed
  without re-running ``dvc repro`` fails, as ADR-0004 requires: a ``dvc.lock`` change lands with
  the change that caused it. Because ``contracts/schema.py`` is a dep of every stage, even a
  docstring edit there needs ``uv run dvc repro`` and a ``dvc.lock`` commit in the same PR.
- The md5 tests need the real data, so they are skipped when a file is not on disk (as in CI).
  Locally, a failure means the files on disk are not the version this ``dvc.lock`` points to.
"""

from pathlib import Path

import pytest
import yaml

from claimfreq.contracts.model import file_md5

REPO = Path(__file__).resolve().parents[1]
STAGES = yaml.safe_load((REPO / "dvc.yaml").read_text())["stages"]
LOCK = yaml.safe_load((REPO / "dvc.lock").read_text())["stages"]
PARAMS = yaml.safe_load((REPO / "params.yaml").read_text())
LOCKED_OUT_MD5 = {o["path"]: o["md5"] for s in LOCK.values() for o in s["outs"]}
# Every out recorded in dvc.lock, as (path, md5).
LOCKED_OUTS = [(out["path"], out["md5"]) for stage in LOCK.values() for out in stage["outs"]]


def test_lock_records_every_out_of_dvc_yaml_with_a_plain_md5():
    # Compared as sets: DVC writes each stage's outs to dvc.lock sorted by path.
    assert set(LOCK) == set(STAGES)
    for name, stage in STAGES.items():
        locked = LOCK[name]["outs"]
        assert {out["path"] for out in locked} == set(stage["outs"]), f"re-run `dvc repro` ({name})"
        # A plain file md5 (not a directory's ".dir" hash) is what data_hash can equal.
        assert all(out["hash"] == "md5" and len(out["md5"]) == 32 for out in locked)


@pytest.mark.parametrize("name", STAGES)
def test_lock_matches_dvc_yaml_params_and_code(name):
    stage, locked = STAGES[name], LOCK[name]
    hint = f"re-run `uv run dvc repro` and commit dvc.lock ({name})"
    assert locked["cmd"] == stage["cmd"], hint
    assert {d["path"] for d in locked["deps"]} == set(stage["deps"]), hint
    want = {key: PARAMS[key] for key in stage.get("params", [])}
    assert locked.get("params", {}).get("params.yaml", {}) == want, hint
    for dep in locked["deps"]:
        if dep["path"] in LOCKED_OUT_MD5:
            # A data dep must be the version its upstream stage's out records.
            assert dep["md5"] == LOCKED_OUT_MD5[dep["path"]], hint
        else:
            assert file_md5(REPO / dep["path"]) == dep["md5"], f"{dep['path']} changed: {hint}"


@pytest.mark.parametrize(("path", "md5"), LOCKED_OUTS, ids=[p for p, _ in LOCKED_OUTS])
def test_file_on_disk_matches_its_md5_in_dvc_lock(path, md5):
    # data/splits/train.parquet is the one data_hash carries; the others keep the set consistent.
    file = REPO / path
    if not file.is_file():
        pytest.skip(f"{path} is not on disk (get it with `uv run dvc pull` or `uv run dvc repro`)")
    assert file_md5(file) == md5, (
        f"{path} differs from the version dvc.lock records. Run `uv run dvc pull` to fetch that "
        "version, or `uv run dvc repro` to rebuild it. If a rebuild changes dvc.lock with no "
        "code, params, or uv.lock change, that is a reproducibility bug: report it, don't commit."
    )
