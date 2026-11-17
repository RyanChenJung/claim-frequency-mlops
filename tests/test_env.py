"""Tests for the environment contract: git_commit() for the lineage tag."""

import subprocess

import pytest

from claimfreq.contracts import env


@pytest.fixture
def outside_any_repo(tmp_path, monkeypatch):
    """Run in an empty directory that git cannot resolve to a repository, with GIT_COMMIT unset."""
    monkeypatch.delenv(env.GIT_COMMIT, raising=False)
    monkeypatch.delenv("GIT_DIR", raising=False)
    # Stop git from finding a repository in a parent directory of tmp_path.
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_git_commit_prefers_the_variable(outside_any_repo, monkeypatch):
    monkeypatch.setenv(env.GIT_COMMIT, "abc1234")
    assert env.git_commit() == "abc1234"


def test_git_commit_falls_back_to_git(outside_any_repo, monkeypatch):
    monkeypatch.setenv(env.GIT_COMMIT, "")  # empty counts as unset here
    # Self-contained identity, and no signing, whatever the developer's global git config says.
    git = ["git", "-c", "user.name=test", "-c", "user.email=test@example.com"]
    git += ["-c", "commit.gpgsign=false"]
    subprocess.run([*git, "init", "-q"], check=True)
    subprocess.run([*git, "commit", "-q", "--allow-empty", "-m", "init"], check=True)
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    assert env.git_commit() == head
    assert len(head) == 40


def test_git_commit_fails_without_either(outside_any_repo):
    with pytest.raises(RuntimeError, match="GIT_COMMIT"):
        env.git_commit()
