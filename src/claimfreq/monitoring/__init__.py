"""Monitoring (Role D): compare logged traffic with the champion's reference dataset.

- ``window`` reads one time window of the monitoring logs.
- ``checks`` runs the five checks on it.
- ``reference`` loads the reference dataset from the champion's MLflow run.

``claimfreq monitor`` (in ``claimfreq.cli``) puts the three together.
"""
