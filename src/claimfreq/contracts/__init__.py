"""Shared contracts between the four roles. See docs/contracts.md for the overview.

- ``schema``: policy columns, dtypes, bounds, categories, frame validation, ``schema_hash``.
- ``model``: registry names, aliases, tags, metric keys, model I/O, the constant baseline.
- ``api``: endpoint paths and request and response bodies.
- ``logs``: tables of the monitoring database.
- ``paths``: data file locations and drift scenarios.
- ``params``: the validated view of params.yaml.
- ``env``: environment variable names and defaults.

Submodules are not imported here, so importing ``paths`` or ``env`` stays cheap for the CLI.
"""
