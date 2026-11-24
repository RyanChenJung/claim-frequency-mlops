"""Load the reference dataset from the champion's MLflow run.

``register`` (Role B) logs it on the run of each registered version, at
``reference/reference.parquet``, with ``REFERENCE_COLUMNS``: the validation split plus the
model's ``predicted_frequency`` (docs/contracts.md, Model).
"""

import tempfile
from pathlib import Path

import pandas as pd

from claimfreq.contracts.model import (
    ALIAS_CHAMPION,
    REFERENCE_ARTIFACT_PATH,
    REFERENCE_COLUMNS,
    REGISTERED_MODEL_NAME,
)


def load_reference(tracking_uri: str) -> tuple[pd.DataFrame, str]:
    """Return the champion's reference dataset and the champion's version number.

    Raises any MLflow error, for example when there is no champion yet, and ``ValueError`` when
    the file lacks a reference column.
    """
    # Imported here, because MLflow is slow to import.
    import mlflow
    from mlflow import MlflowClient

    client = MlflowClient(tracking_uri=tracking_uri)
    version = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, ALIAS_CHAMPION)
    with tempfile.TemporaryDirectory() as folder:
        path = mlflow.artifacts.download_artifacts(
            run_id=version.run_id,
            artifact_path=REFERENCE_ARTIFACT_PATH,
            dst_path=folder,
            tracking_uri=tracking_uri,
        )
        reference = pd.read_parquet(Path(path))
    missing = [column for column in REFERENCE_COLUMNS if column not in reference.columns]
    if missing:
        raise ValueError(f"the reference dataset lacks columns: {', '.join(missing)}")
    return reference[REFERENCE_COLUMNS], str(version.version)
