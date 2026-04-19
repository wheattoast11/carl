"""SEM-006: CascadeRewardManager must gate on crystallization events.

Reward-volume thresholds (the existing metric gate) can be gamed by
trivial patterns. Crystallization events — ``delta_phi > DEFECT_THRESHOLD``
at low prior phi — cannot be gamed; they are a phase-transition property
of the Phi field itself.

These tests exercise the new ``gate_mode="crystallization"`` path, the
unchanged ``gate_mode="metric"`` path, the rolling-window behavior, and
the trace-reading path through ``CascadeCallback``.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from carl_studio.training.cascade import (
    CascadeCallback,
    CascadeRewardManager,
)


# ---------------------------------------------------------------------------
# Crystallization gating
# ---------------------------------------------------------------------------


def test_cascade_crystallization_mode_fires_on_events():
    """Gate fires as soon as the windowed crystallization sum meets the bar."""
    cascade = CascadeRewardManager(
        gate_mode="crystallization",
        n_crystallizations_required=3,
        crystallization_window=10,
    )
    assert cascade._mode == "adaptive"
    assert cascade.get_stage() == "A"

    cascade._step = 1
    cascade.record_crystallizations(1)
    assert not cascade._gate_fired
    cascade._step = 2
    cascade.record_crystallizations(1)
    assert not cascade._gate_fired

    cascade._step = 3
    cascade.record_crystallizations(1)  # rolling sum 3 >= 3 -> fire
    assert cascade._gate_fired
    assert cascade._gate_fired_at == 3
    assert cascade.get_stage() == "B"


def test_cascade_metric_mode_still_works():
    """Backward compat: metric-mode gating unchanged by the SEM-006 refactor."""
    cascade = CascadeRewardManager(
        gate_metric="task_completion",
        gate_percentile=0.6,
        gate_window=5,
        gate_min_above=3,
        gate_min_steps=5,
    )
    assert cascade._mode == "adaptive"
    assert cascade._gate_mode == "metric"
    assert cascade.get_stage() == "A"

    for _ in range(6):
        cascade.record_metric(0.5)
        cascade._step += 1

    assert cascade._gate_fired
    assert cascade.get_stage() == "B"


def test_cascade_insufficient_crystallizations_holds_stage():
    """If the rolling sum never reaches the bar, the gate never fires."""
    cascade = CascadeRewardManager(
        gate_mode="crystallization",
        n_crystallizations_required=5,
        crystallization_window=10,
    )
    for i in range(20):
        cascade._step = i
        # One event every other step — total in any 10-step window is <= 5
        cascade.record_crystallizations(1 if i % 3 == 0 else 0)

    # Over 20 steps with maxlen=10 window + 1-every-3-steps we stay < 5.
    assert not cascade._gate_fired
    assert cascade.get_stage() == "A"


def test_cascade_window_rolls_correctly():
    """Old events roll off the deque; gate can NOT fire on stale counts."""
    cascade = CascadeRewardManager(
        gate_mode="crystallization",
        n_crystallizations_required=4,
        crystallization_window=3,
    )

    # Fill the window with counts that sum to exactly the bar — then fire.
    cascade._step = 1
    cascade.record_crystallizations(2)
    cascade._step = 2
    cascade.record_crystallizations(1)
    assert not cascade._gate_fired  # sum = 3 < 4

    cascade._step = 3
    cascade.record_crystallizations(1)  # sum = 4 -> fire
    assert cascade._gate_fired
    assert cascade._gate_fired_at == 3

    # Now prove the window ROLLS with a fresh cascade: old counts age out,
    # later pushes cannot accidentally carry them forward.
    cascade2 = CascadeRewardManager(
        gate_mode="crystallization",
        n_crystallizations_required=4,
        crystallization_window=3,
    )
    cascade2._step = 1
    cascade2.record_crystallizations(3)  # window=[3]
    cascade2._step = 2
    cascade2.record_crystallizations(0)  # window=[3,0]
    cascade2._step = 3
    cascade2.record_crystallizations(0)  # window=[3,0,0] sum=3 < 4
    cascade2._step = 4
    cascade2.record_crystallizations(0)  # window=[0,0,0] sum=0
    cascade2._step = 5
    cascade2.record_crystallizations(0)  # still 0
    cascade2._step = 6
    cascade2.record_crystallizations(0)
    assert not cascade2._gate_fired  # old '3' aged out; gate never fires


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


def test_cascade_rejects_unknown_gate_mode():
    with pytest.raises(ValueError, match="gate_mode"):
        CascadeRewardManager(gate_mode="nonsense")  # type: ignore[arg-type]


def test_cascade_rejects_zero_crystallizations_required():
    with pytest.raises(ValueError, match="n_crystallizations_required"):
        CascadeRewardManager(
            gate_mode="crystallization",
            n_crystallizations_required=0,
        )


def test_cascade_rejects_zero_window():
    with pytest.raises(ValueError, match="crystallization_window"):
        CascadeRewardManager(
            gate_mode="crystallization",
            crystallization_window=0,
        )


# ---------------------------------------------------------------------------
# Callback integration: traces flow from reward_fn -> cascade -> stage change
# ---------------------------------------------------------------------------


def _make_stub_reward(traces):
    """Build a reward-function stand-in exposing the CARLReward contract."""

    def reward_fn(completions, **kwargs):
        return [0.0] * len(completions)

    reward_fn._last_traces = [traces]  # type: ignore[attr-defined]
    return reward_fn


def test_callback_reads_traces_and_fires_gate():
    """The callback path end-to-end: traces -> record -> gate fire."""
    cascade = CascadeRewardManager(
        gate_mode="crystallization",
        n_crystallizations_required=2,
        crystallization_window=5,
    )
    # Two traces in the batch -> total crystallizations = 2 -> fire.
    trace_a = SimpleNamespace(n_crystallizations=1)
    trace_b = SimpleNamespace(n_crystallizations=1)
    reward_fn = _make_stub_reward([trace_a, trace_b])
    cascade.attach_reward_fn(reward_fn)

    callback = CascadeCallback(cascade)
    state = SimpleNamespace(global_step=1)
    callback.on_step_end(args=None, state=state, control=None)

    assert cascade._gate_fired
    assert cascade.get_stage_weight({"B"}) > 0.0


def test_callback_no_op_when_traces_missing():
    """Missing _last_traces must not crash — gate simply holds."""
    cascade = CascadeRewardManager(
        gate_mode="crystallization",
        n_crystallizations_required=1,
        crystallization_window=3,
    )

    def bare_reward(completions, **kwargs):
        return [0.0] * len(completions)

    cascade.attach_reward_fn(bare_reward)  # no _last_traces attribute
    callback = CascadeCallback(cascade)
    state = SimpleNamespace(global_step=1)
    callback.on_step_end(args=None, state=state, control=None)
    assert not cascade._gate_fired


def test_callback_no_op_when_no_reward_bound():
    """If no reward function was attached, the callback silently skips."""
    cascade = CascadeRewardManager(
        gate_mode="crystallization",
        n_crystallizations_required=1,
    )
    callback = CascadeCallback(cascade)
    state = SimpleNamespace(global_step=1)
    # Never called attach_reward_fn — callback should simply no-op.
    callback.on_step_end(args=None, state=state, control=None)
    assert not cascade._gate_fired


def test_callback_logs_crystallization_state():
    """on_log exposes the new crystallization diagnostics."""
    cascade = CascadeRewardManager(
        gate_mode="crystallization",
        n_crystallizations_required=4,
        crystallization_window=5,
    )
    cascade._step = 2
    cascade.record_crystallizations(1)
    cascade.record_crystallizations(1)

    callback = CascadeCallback(cascade)
    state = SimpleNamespace(global_step=2)
    logs: dict[str, float] = {}
    callback.on_log(args=None, state=state, control=None, logs=logs)
    assert logs["cascade/gate_mode"] == "crystallization"
    assert logs["cascade/recent_crystallizations"] == 2
    assert logs["cascade/crystallizations_required"] == 4
    assert logs["cascade/stage"] == 0  # still stage A


# ---------------------------------------------------------------------------
# wrap_reward auto-binds the reward function in crystallization mode
# ---------------------------------------------------------------------------


def test_wrap_reward_auto_binds_in_crystallization_mode():
    cascade = CascadeRewardManager(
        gate_mode="crystallization",
        n_crystallizations_required=1,
    )
    trace = SimpleNamespace(n_crystallizations=1)
    reward_fn = _make_stub_reward([trace])

    wrapped = cascade.wrap_reward(reward_fn, active_in_stages={"B"})
    # Auto-bind is what matters here: the callback must now find _last_traces.
    assert cascade._reward_fn is reward_fn

    # And the traces propagate to the wrapped callable (pre-existing behavior).
    assert getattr(wrapped, "_last_traces", None) is reward_fn._last_traces


def test_record_metric_is_ignored_in_crystallization_mode():
    """Scalar metric reads are irrelevant when gating on crystallizations."""
    cascade = CascadeRewardManager(
        gate_mode="crystallization",
        n_crystallizations_required=2,
    )
    for _ in range(20):
        cascade.record_metric(10.0)  # would fire a metric-mode gate easily
    assert not cascade._gate_fired
    assert cascade._gate_history == []


def test_record_crystallizations_clamps_negative_and_nonfinite():
    """Defensive: negative or non-int counts get normalized to 0."""
    cascade = CascadeRewardManager(
        gate_mode="crystallization",
        n_crystallizations_required=5,
        crystallization_window=10,
    )
    cascade.record_crystallizations(-3)
    cascade.record_crystallizations(float("nan"))  # type: ignore[arg-type]
    assert sum(cascade._recent_crystallizations) == 0
    assert not cascade._gate_fired
