"""Tests for CARLTrainer -- no GPU required."""

import pytest
from carl_studio.training.trainer import CARLTrainer
from carl_studio.types.config import TrainingConfig
from carl_studio.types.run import RunPhase


def _make_config(**overrides):
    defaults = {
        "run_name": "test-run",
        "base_model": "Qwen/Qwen3-8B",
        "output_repo": "test/test",
        "method": "sft",
        "dataset_repo": "trl-lib/Capybara",
        "compute_target": "local",
    }
    defaults.update(overrides)
    return TrainingConfig(**defaults)


def test_trainer_init():
    config = _make_config()
    trainer = CARLTrainer(config)
    assert trainer.run.phase == RunPhase.INITIALIZING
    assert trainer.run.id  # UUID generated


def test_trainer_parse_timeout():
    assert CARLTrainer._parse_timeout("3h") == 10800
    assert CARLTrainer._parse_timeout("90m") == 5400
    assert CARLTrainer._parse_timeout("14400") == 14400


def test_trainer_parse_timeout_fractional():
    assert CARLTrainer._parse_timeout("1.5h") == 5400
    assert CARLTrainer._parse_timeout("2.5m") == 150


def test_trainer_parse_timeout_compound():
    assert CARLTrainer._parse_timeout("2h30m") == 9000


def test_trainer_parse_timeout_empty_returns_default():
    assert CARLTrainer._parse_timeout("") == 10800  # default 3h


def test_trainer_remote_dispatch():
    """Test that non-LOCAL targets dispatch to remote mode."""
    config = _make_config(compute_target="l4x1")
    trainer = CARLTrainer(config)
    assert trainer.config.compute_target.value == "l4x1"
    assert trainer.is_remote is True


def test_trainer_local_dispatch():
    """Test that LOCAL target is not remote."""
    config = _make_config(compute_target="local")
    trainer = CARLTrainer(config)
    assert trainer.is_remote is False


def test_trainer_run_id_unique():
    c1 = CARLTrainer(_make_config())
    c2 = CARLTrainer(_make_config())
    assert c1.run.id != c2.run.id


def test_trainer_run_has_config():
    config = _make_config(run_name="my-special-run")
    trainer = CARLTrainer(config)
    assert trainer.run.config.run_name == "my-special-run"


def test_trainer_all_compute_targets():
    """Every non-LOCAL compute target should be remote."""
    for target in ["l4x1", "l4x4", "a10g-large", "a10g-largex2", "a100-large"]:
        config = _make_config(compute_target=target)
        trainer = CARLTrainer(config)
        assert trainer.is_remote is True, f"{target} should be remote"


def test_trainer_initial_step_zero():
    trainer = CARLTrainer(_make_config())
    assert trainer.run.current_step == 0
    assert trainer.run.total_steps == 0
