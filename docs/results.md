# Results

Pending the real splits (#10, #12). This file is generated, not written by hand. After
`uv run dvc pull`, run `train`, `register`, `promote --version N`, and `evaluate-test`, then:

```bash
uv run python -m claimfreq.results > docs/results.md
```

The MLflow UI screenshot comparing the candidates goes in `docs/evidence/` (slide d).
