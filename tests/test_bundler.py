"""Test the bundler generates valid scripts."""
import pytest
from carl_studio.types.config import TrainingConfig
from carl_studio.bundler import Bundler


def _make_config(**overrides):
    defaults = {
        "run_name": "test",
        "base_model": "Qwen/Qwen3-8B",
        "output_repo": "test/test",
        "method": "sft",
        "dataset_repo": "trl-lib/Capybara",
    }
    defaults.update(overrides)
    return TrainingConfig(**defaults)


def test_sft_bundle_valid_python():
    config = _make_config(method="sft")
    script = Bundler().generate(config)
    # Should be valid Python
    compile(script, "<bundle>", "exec")
    assert "# /// script" in script
    assert "Qwen/Qwen3-8B" in script
    assert "SFTTrainer" in script


def test_sft_bundle_has_deps():
    config = _make_config()
    script = Bundler().generate(config)
    assert "trl" in script
    assert "peft" in script
    assert "trackio" in script


def test_grpo_bundle_valid_python():
    config = _make_config(method="grpo")
    script = Bundler().generate(config)
    compile(script, "<bundle>", "exec")
    assert "GRPOTrainer" in script
    assert "KAPPA" in script
    assert "DEFECT_THRESHOLD" in script
    assert "tool_call_format_reward" in script


def test_grpo_bundle_has_carl():
    config = _make_config(method="grpo")
    script = Bundler().generate(config)
    assert "carl_composite_reward" in script or "make_carl_reward" in script
    assert "CascadeRewardManager" in script
    assert "CoherenceMonitorCallback" in script


def test_grpo_bundle_has_all_rewards():
    config = _make_config(method="grpo")
    script = Bundler().generate(config)
    assert "tool_call_format_reward" in script
    assert "tool_selection_reward" in script
    assert "conciseness_reward" in script


def test_bundle_unsupported_method():
    config = _make_config(method="dpo")
    with pytest.raises((ValueError, NotImplementedError)):
        Bundler().generate(config)
