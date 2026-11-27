Closes #

## What and why

<!-- One or two sentences: what this PR changes, and why. -->

## Checklist

- [ ] `uv run ruff check .`, `uv run ruff format --check .`, and `uv run pytest` pass locally
- [ ] Comments and docstrings on public code
- [ ] Tests added for the new behavior
- [ ] `STUBS` in `tests/test_cli.py` updated (if this PR implements a stubbed command)
- [ ] `requirements.txt` re-exported if dependencies changed (`uv lock`, then `uv export --no-dev --no-hashes -o requirements.txt`; commit `pyproject.toml`, `uv.lock`, and `requirements.txt` together)
- [ ] `contract-change` label and a review from every affected role, if this PR changes anything in `src/claimfreq/contracts/`, a CLI command or exit code, an environment variable, or a cross-role `params.yaml` key
