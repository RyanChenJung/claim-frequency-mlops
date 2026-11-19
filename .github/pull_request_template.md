Closes #

## What and why

<!-- One or two sentences: what this PR changes, and why. -->

## Checklist

- [ ] Comments and docstrings on public code
- [ ] Tests added for the new behavior
- [ ] `STUBS` in `tests/test_cli.py` updated (if this PR implements a stubbed command)
- [ ] `requirements.txt` re-exported if dependencies changed (`uv lock`, then `uv export --no-dev --no-hashes -o requirements.txt`; commit `pyproject.toml`, `uv.lock`, and `requirements.txt` together)
- [ ] `contract-change` label, and a review from every affected role, if a contract in `docs/contracts.md` changed
