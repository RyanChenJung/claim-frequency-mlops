"""Tests for the params contract and the shipped params.yaml."""

from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import BaseModel, ValidationError

from claimfreq.contracts.params import SEED_MAX, Params, load_params

PARAMS_YAML = Path(__file__).resolve().parents[1] / "params.yaml"


def field_tree(model: type[BaseModel]) -> dict[str, Any]:
    """Return the nested field names of a params model; leaves map to None."""
    tree: dict[str, Any] = {}
    for name, field in model.model_fields.items():
        annotation = field.annotation
        is_section = isinstance(annotation, type) and issubclass(annotation, BaseModel)
        tree[name] = field_tree(annotation) if is_section else None
    return tree


def key_tree(data: dict[str, Any]) -> dict[str, Any]:
    """Return the nested keys of a parsed YAML mapping, in the same shape as ``field_tree``."""
    return {k: key_tree(v) if isinstance(v, dict) else None for k, v in data.items()}


def test_params_yaml_validates_and_sets_every_field():
    load_params(PARAMS_YAML)
    # Every field is explicit in the file, so no value silently comes from a code default.
    assert key_tree(yaml.safe_load(PARAMS_YAML.read_text())) == field_tree(Params)


def write_params(tmp_path: Path, override: dict) -> Path:
    """Write params.yaml with some section keys overridden, and return its path."""
    data = yaml.safe_load(PARAMS_YAML.read_text())
    for section, values in override.items():
        data[section].update(values)
    path = tmp_path / "params.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


def test_invalid_selection_metric_is_rejected(tmp_path):
    path = write_params(tmp_path, {"training": {"selection_metric": "rmse"}})
    with pytest.raises(ValidationError):
        load_params(path)


@pytest.mark.parametrize(
    "override",
    [
        {"split": {"test_fraction": 0.6, "validation_fraction": 0.4}},
        {"monitor": {"calibration_min": 1.2}},
        {"traffic": {"batch_size": 1001}},
        {"clean": {"claim_nb_cap": 4.0}},  # strict integer
        {"clean": {"claim_nb_capp": 4}},  # typo in a key
        {"training": {"flaml_time_budget_s": True}},  # YAML true is not 1
        {"monitor": {"fail_on_prediction_drift": 1}},  # 1 is not true
        {"monitor": {"drift_share_max": True}},  # YAML true is not a float
        {"monitor": {"drift_share_max": "0.2"}},  # a quoted string is not a float
        {"monitor": {"min_predictions": 0}},  # at least 1
    ],
)
def test_invalid_values_are_rejected(tmp_path, override):
    with pytest.raises(ValidationError):
        load_params(write_params(tmp_path, override))


def test_strict_floats_accept_integers():
    params = Params.model_validate({"clean": {"exposure_cap": 1}})
    assert params.clean.exposure_cap == 1.0


@pytest.mark.parametrize("seed", [-1, SEED_MAX + 1, True])
def test_seeds_are_strict_and_bounded(seed):
    with pytest.raises(ValidationError):
        Params.model_validate({"seed": seed})
    with pytest.raises(ValidationError):
        Params.model_validate({"training": {"seed": seed}})


def test_seed_range_boundaries_are_accepted():
    assert Params.model_validate({"seed": 0, "training": {"seed": SEED_MAX}}).seed == 0


@pytest.mark.parametrize(
    "glm",
    [
        {"solver": "saga"},  # not a PoissonRegressor solver
        {"alpha": -1.0},
        {"max_iter": 0},
        {"vehage_bins": 1},
        {"drivage_edges": []},
        {"drivage_edges": [25, 21]},  # must increase
        {"drivage_edges": [21, 21]},  # strictly
        {"drivage_edges": ["21"]},  # strict integers
        {"lr": 0.1},  # unknown key
    ],
)
def test_invalid_glm_params_are_rejected(glm):
    with pytest.raises(ValidationError):
        Params.model_validate({"training": {"glm": glm}})
