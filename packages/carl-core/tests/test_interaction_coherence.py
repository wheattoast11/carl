"""Tests for the Step cross-channel coherence extension (SEM-007).

These tests pin down:
  * ``Step`` accepts ``phi`` / ``kuramoto_r`` / ``channel_coherence``
  * Existing callsites that never knew these fields still work.
  * ``InteractionChain.coherence_trajectory`` reduces a chain to a
    well-defined phi-vs-step series.
"""
from __future__ import annotations

from carl_core.interaction import ActionType, InteractionChain, Step


def test_step_accepts_phi_kuramoto_channel_coherence() -> None:
    cc = {
        "phi_mean": 0.42,
        "cloud_quality": 0.31,
        "success_rate": 0.9,
        "latency_ms": 75.0,
    }
    step = Step(
        action=ActionType.EXTERNAL,
        name="demo.step",
        phi=0.42,
        kuramoto_r=0.77,
        channel_coherence=cc,
    )
    assert step.phi == 0.42
    assert step.kuramoto_r == 0.77
    assert step.channel_coherence == cc

    # Round-trip via chain.to_dict -> from_dict.
    chain = InteractionChain()
    chain.steps.append(step)
    revived = InteractionChain.from_dict(chain.to_dict())
    assert len(revived.steps) == 1
    r = revived.steps[0]
    assert r.phi == 0.42
    assert r.kuramoto_r == 0.77
    assert r.channel_coherence == cc


def test_step_defaults_preserve_legacy_callsite() -> None:
    # No new kwargs — the field defaults must keep every existing
    # construction shape valid.
    step = Step(action=ActionType.CLI_CMD, name="legacy.call")
    assert step.phi is None
    assert step.kuramoto_r is None
    assert step.channel_coherence is None

    d = step.to_dict()
    assert d["phi"] is None
    assert d["kuramoto_r"] is None
    assert d["channel_coherence"] is None

    # And InteractionChain.record defaults behave identically.
    chain = InteractionChain()
    s = chain.record(ActionType.TOOL_CALL, "tool.call")
    assert s.phi is None
    assert s.kuramoto_r is None
    assert s.channel_coherence is None


def test_coherence_trajectory_returns_per_step_phi() -> None:
    chain = InteractionChain()
    chain.record(ActionType.TRAINING_STEP, "train.0", phi=0.10)
    chain.record(ActionType.TRAINING_STEP, "train.1", phi=0.35)
    chain.record(ActionType.EVAL_PHASE, "eval.0", phi=0.80)

    traj = chain.coherence_trajectory()
    assert len(traj) == 3
    assert [phi for _, phi in traj] == [0.10, 0.35, 0.80]
    # Timestamps are the started_at isoformat of each step.
    for (ts, _), step in zip(traj, chain.steps):
        assert ts == step.started_at.isoformat()


def test_chain_trajectory_yields_none_for_missing_phi() -> None:
    chain = InteractionChain()
    chain.record(ActionType.USER_INPUT, "user.hello")  # no phi
    chain.record(ActionType.TRAINING_STEP, "train.0", phi=0.42)
    chain.record(ActionType.EXTERNAL, "external.ping")  # no phi

    traj = chain.coherence_trajectory()
    assert [phi for _, phi in traj] == [None, 0.42, None]
