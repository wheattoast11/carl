"""Test cascade stage management (2-stage: A/B)."""
import math

import pytest

from carl_studio.training.cascade import CascadeRewardManager
from carl_studio.training.rewards.multiscale import (
    clamp_counts,
    reset_clamp_counts,
)


@pytest.fixture(autouse=True)
def _fresh_clamp_state():
    reset_clamp_counts()
    yield
    reset_clamp_counts()


def test_stages():
    cascade = CascadeRewardManager(carl_start=100)
    cascade._step = 0
    assert cascade.get_stage() == "A"
    cascade._step = 99
    assert cascade.get_stage() == "A"
    cascade._step = 100
    assert cascade.get_stage() == "B"
    cascade._step = 200
    assert cascade.get_stage() == "B"


def test_warmup():
    cascade = CascadeRewardManager(carl_start=100, warmup_steps=10)

    # Before stage B: weight = 0
    cascade._step = 99
    assert cascade.get_stage_weight({"B"}) == 0.0

    # First step of B: weight = 0.1 (1/10 warmup)
    cascade._step = 100
    weight = cascade.get_stage_weight({"B"})
    assert 0.0 < weight < 0.5

    # After warmup: weight = 1.0
    cascade._step = 115
    assert cascade.get_stage_weight({"B"}) == 1.0


def test_wrap_reward():
    cascade = CascadeRewardManager(carl_start=10)

    def dummy_reward(completions, **kwargs):
        return [1.0] * len(completions)

    wrapped = cascade.wrap_reward(dummy_reward, active_in_stages={"B"})

    # Stage A: returns zeros
    cascade._step = 5
    assert wrapped(["x"]) == [0.0]

    # Stage B after warmup (warmup_steps=10 default, carl_start=10 -> fully warm at step 20)
    cascade._step = 25
    assert wrapped(["x"]) == [1.0]


def test_adaptive_gate():
    cascade = CascadeRewardManager(
        gate_metric="task_completion",
        gate_percentile=0.6,
        gate_window=5,
        gate_min_above=3,
        gate_min_steps=5,
    )
    assert cascade._mode == "adaptive"
    assert cascade.get_stage() == "A"  # gate hasn't fired

    # Feed enough nonzero metrics to trigger the gate
    for _ in range(6):
        cascade.record_metric(0.5)
        cascade._step += 1

    # Gate should fire: 60th percentile of [0.5]*6 = 0.5 > 0, min_above met
    assert cascade._gate_fired
    assert cascade.get_stage() == "B"


def test_default_carl_start():
    cascade = CascadeRewardManager()
    assert cascade.carl_start == 50


# ---------------------------------------------------------------------------
# WS-T3 — clamping through wrap_reward
# ---------------------------------------------------------------------------


def test_wrap_reward_clamps_nan():
    cascade = CascadeRewardManager(carl_start=0, warmup_steps=1)
    cascade._step = 10  # fully warm, stage B

    def bad_reward(completions, **kwargs):
        return [float("nan"), float("inf"), 1e9, -1e9, 0.5]

    wrapped = cascade.wrap_reward(bad_reward, active_in_stages={"B"})
    out = wrapped(["a", "b", "c", "d", "e"])
    assert out[0] == 0.0  # NaN clamped
    assert out[1] == 0.0  # +inf clamped
    assert out[2] == 100.0  # upper cap
    assert out[3] == -100.0  # lower cap
    assert out[4] == 0.5   # pass-through


def test_wrap_reward_isolates_exception():
    cascade = CascadeRewardManager(carl_start=0, warmup_steps=1)
    cascade._step = 10

    def broken(completions, **kwargs):
        raise RuntimeError("reward crashed")

    wrapped = cascade.wrap_reward(broken, active_in_stages={"B"})
    out = wrapped(["x", "y", "z"])
    assert out == [0.0, 0.0, 0.0]


def test_wrap_reward_warmup_still_clamps():
    cascade = CascadeRewardManager(carl_start=10, warmup_steps=10)
    cascade._step = 10  # first warmup tick

    def blow_up(completions, **kwargs):
        return [1e12] * len(completions)

    wrapped = cascade.wrap_reward(blow_up, active_in_stages={"B"})
    out = wrapped(["a"])
    # Warmup weight = 0.1, but clamp applies both before and after scaling.
    assert -100.0 <= out[0] <= 100.0


# ---------------------------------------------------------------------------
# WS-T4 — empty gate history guard
# ---------------------------------------------------------------------------


def test_empty_gate_history_returns_inf():
    cascade = CascadeRewardManager(
        gate_metric="task_completion",
        gate_min_steps=1,
    )
    assert math.isinf(cascade._compute_adaptive_threshold())


def test_fixed_mode_has_empty_gate_history():
    cascade = CascadeRewardManager(carl_start=100)
    # Fixed-mode instances still expose the attribute, empty.
    assert hasattr(cascade, "_gate_history")
    assert cascade._gate_history == []
    assert math.isinf(cascade._compute_adaptive_threshold())


def test_record_metric_clamps_nonfinite_values():
    cascade = CascadeRewardManager(
        gate_metric="task_completion",
        gate_min_steps=2,
    )
    cascade.record_metric(float("nan"))
    cascade.record_metric(float("inf"))
    # Both should have been clamped to 0.0 and appended.
    assert cascade._gate_history == [0.0, 0.0]
    assert clamp_counts()["nonfinite"] == 2
