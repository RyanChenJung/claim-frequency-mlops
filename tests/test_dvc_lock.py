"""Guard tests for the committed ``dvc.lock`` (#12; docs/contracts.md, Model -> data_hash).

``dvc.lock`` records the md5 of every stage out. ``train`` tags each model with
``data_hash = model.file_md5(train.parquet)``, and the two must agree, so a model can be traced
to the data version a git commit points to.

- The sync test runs everywhere, CI included: every out in ``dvc.yaml`` has a plain md5 in
  ``dvc.lock``, so a ``dvc.yaml`` edit committed without re-running ``dvc repro`` fails.
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


@pytest.mark.parametrize(("path", "md5"), LOCKED_OUTS, ids=[p for p, _ in LOCKED_OUTS])
def test_file_on_disk_matches_its_md5_in_dvc_lock(path, md5):
    # data/splits/train.parquet is the one data_hash carries; the others keep the set consistent.
    file = REPO / path
    if not file.is_file():
        pytest.skip(f"{path} is not on disk (get it with `uv run dvc pull` or `uv run dvc repro`)")
    assert file_md5(file) == md5, (
        f"{path} differs from the version dvc.lock records. Run `uv run dvc pull` to fetch that "
        "version, or `uv run dvc repro` to rebuild it (then commit dvc.lock if it changed)."
    )
