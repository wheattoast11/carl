"""Test cascade stage management (2-stage: A/B)."""
from carl_studio.training.cascade import CascadeRewardManager


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
