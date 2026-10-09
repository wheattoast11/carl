"""Private drafts remain private across public evaluation and training rewards."""

from __future__ import annotations

import pytest

from carl_studio.eval.runner import EvalConfig, EvalGate, EvalReport, PrivacyBoundaryGate
from carl_studio.training.rewards.privacy import privacy_rewards, suppress_private_rewards


@pytest.mark.parametrize(
    "event",
    [
        {"phase": "public", "text": "draft-123"},
        {"phase": "public", "payload": [{"parent": "draft-123"}]},
        {"phase": "public", "draft-123": "value"},
        {"phase": "public", "nested": {"phase": "private", "text": "draft-123"}},
    ],
)
def test_private_identifiers_in_public_events_fail(event):
    gate = PrivacyBoundaryGate()
    assert not gate.audit_trajectory([event], {"draft-123"})
    assert gate.terminal_reward([event], {"draft-123"}) == -3.0


def test_private_events_and_clean_publication_pass():
    trajectory = [
        {"phase": "private", "text": "draft-123"},
        {"phase": "public", "text": "published answer"},
    ]
    assert PrivacyBoundaryGate().audit_trajectory(trajectory, {"draft-123"})
    assert PrivacyBoundaryGate().terminal_reward(trajectory, {"draft-123"}) == 0.0


@pytest.mark.parametrize(
    "trajectory", [None, 1, [], [{}], [{"phase": "unknown"}], [{"phase": []}], [{"phase": {}}], [None]]
)
def test_malformed_trajectories_fail(trajectory):
    assert not PrivacyBoundaryGate().audit_trajectory(trajectory, {"draft-123"})


def test_eval_cannot_pass_with_a_private_identifier():
    report = EvalReport(
        checkpoint="fixture",
        phase="2prime",
        n_samples=1,
        primary_metric="success",
        primary_value=1.0,
        threshold=0.5,
        passed=True,
        detail=[{"phase": "public", "text": "draft-123"}],
    )
    config = EvalConfig(
        checkpoint="fixture", private_keys={"draft-123"}, require_coherence_gate=False
    )
    assert not EvalGate(config=config).check(report)
    assert report.metrics["privacy_terminal_reward"] == -3.0
    assert "draft-123" not in report.gate_reason
    assert "private_keys" not in config.model_dump()
    report.detail = [{"phase": "public", "text": "answer"}]
    assert EvalGate(config=config).check(report)
    assert report.metrics["privacy_terminal_reward"] == 0.0


@pytest.mark.parametrize("active_reward", [0.0, 100.0])
def test_terminal_penalty_is_not_masked_or_offset(active_reward):
    completions = ["draft-123 leaked", "clean answer"]
    metadata = {"private_keys": [["draft-123"], ["draft-123"]]}
    weighted = suppress_private_rewards(lambda **kwargs: [active_reward, active_reward])
    rewards = weighted(completions, **metadata)
    terminal = privacy_rewards(completions, **metadata)
    assert [a + b for a, b in zip(rewards, terminal)] == [-3.0, active_reward]


def test_privacy_reward_requires_aligned_metadata():
    with pytest.raises(ValueError, match="one identifier list"):
        privacy_rewards(["answer"], private_keys=[])
    with pytest.raises(ValueError, match="nonempty strings"):
        privacy_rewards(["answer"], private_keys=[[""]])
    with pytest.raises(ValueError, match="one event list"):
        privacy_rewards(["answer"], private_keys=[["draft"]], trajectory=[])
    assert privacy_rewards(["answer"]) == [0.0]


@pytest.mark.parametrize("stage", ["A", "B"])
def test_trainer_attaches_terminal_penalty_outside_cascade(stage, monkeypatch):
    import importlib
    from types import SimpleNamespace

    from carl_studio.training.trainer import CARLTrainer
    from carl_studio.types.config import TrainingConfig

    monkeypatch.setattr(
        importlib.import_module("carl_studio.training.rewards.composite"),
        "make_carl_reward",
        lambda **kwargs: lambda **kwargs: [100.0, 100.0],
    )
    trainer = CARLTrainer(
        TrainingConfig(
            run_name="privacy",
            method="grpo",
            dataset_repo="fixture/data",
            base_model="fixture",
            output_repo="fixture/output",
            cascade={"carl_start": 2, "gate_mode": "metric"},
        ),
        task_reward_funcs=[lambda **kwargs: [100.0, 100.0]],
        task_reward_weights=[50.0],
    )
    functions = trainer._build_rewards(SimpleNamespace(), SimpleNamespace(vocab_size=16))
    if stage == "B":
        trainer._cascade_manager._step = 20
    assert trainer._cascade_manager.get_stage() == stage
    completions = ["draft-123 leaked", "answer"]
    kwargs = {"private_keys": [["draft-123"], ["draft-123"]]}
    values = [function(completions=completions, **kwargs) for function in functions]
    assert sum(row[0] for row in values) == -3.0
    assert sum(row[1] for row in values) > 0.0


def test_numeric_identifiers_and_cyclic_payloads_fail():
    gate = PrivacyBoundaryGate()
    assert gate.terminal_reward([{"phase": "public", "parent": 123}], {"123"}) == -3.0
    payload = {"phase": "public"}
    payload["cycle"] = payload
    assert gate.terminal_reward([payload], {"draft"}) == -3.0
