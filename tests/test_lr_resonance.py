"""Test resonance-aware LR modulation."""
import math
import threading

import numpy as np
import pytest

from carl_studio.primitives.constants import KAPPA
from carl_studio.training.lr_resonance import ResonanceLRCallback


def _make_mock_carl_fn(R_value: float = 0.5):
    """Mock CARL reward fn with traces that produce a specific Kuramoto R."""
    from carl_studio.primitives.coherence_trace import CoherenceTrace

    # Build a trace with known Phi values that produce target R
    # For half-circle mapping: R = |mean(exp(i*pi*phi))|
    # Uniform phi → R depends on distribution
    # For simplicity, use constant phi (R = 1 for constant phi)
    T = 64
    if R_value > 0.9:
        phi = np.full(T, 0.8)
    else:
        phi = np.random.default_rng(42).uniform(0, 1, T).astype(np.float64)

    trace = CoherenceTrace(
        phi=phi,
        entropy=np.ones(T) * 0.5,
        selected_prob=np.ones(T) * 0.1,
        delta_phi=np.diff(phi),
        vocab_size=1000,
        n_tokens=T,
        step=5,
    )

    def carl_fn(completions, **kwargs):
        return [0.5] * len(completions)

    carl_fn._last_traces = [[trace]]
    carl_fn._metrics_lock = threading.Lock()
    carl_fn._step = [5]
    return carl_fn


class MockOptimizer:
    def __init__(self, lr=5e-6):
        self.param_groups = [{"lr": lr}]


class MockState:
    def __init__(self, step=0):
        self.global_step = step


class TestConservationEnvelope:
    def test_step_zero_is_full(self):
        cb = ResonanceLRCallback(d_model=4096, tokens_per_step=2400)
        assert cb._conservation_envelope(0) == 1.0

    def test_t_star_is_zero(self):
        cb = ResonanceLRCallback(d_model=4096, tokens_per_step=2400)
        assert cb._conservation_envelope(cb._t_star_steps) == 0.0

    def test_mid_point_is_half(self):
        cb = ResonanceLRCallback(d_model=4096, tokens_per_step=2400)
        mid = cb._t_star_steps // 2
        envelope = cb._conservation_envelope(mid)
        assert 0.4 < envelope < 0.6  # roughly 0.5

    def test_t_star_calculation(self):
        cb = ResonanceLRCallback(d_model=4096, tokens_per_step=2400)
        expected = int(KAPPA * 4096 / 2400)
        assert cb._t_star_steps == expected

    def test_disabled(self):
        cb = ResonanceLRCallback(envelope_enabled=False)
        assert cb._conservation_envelope(0) == 1.0
        assert cb._conservation_envelope(1000) == 1.0


class TestPhiGate:
    def test_high_R_base_lr(self):
        cb = ResonanceLRCallback(phi_alpha=0.5)
        # R=1.0 → multiplier = 1.0 (base LR, model is locked)
        assert cb._phi_gate(1.0) == 1.0

    def test_low_R_higher_lr(self):
        cb = ResonanceLRCallback(phi_alpha=0.5)
        # R=0.0 → multiplier = 1.5 (50% boost, model restructuring)
        assert cb._phi_gate(0.0) == 1.5

    def test_mid_R(self):
        cb = ResonanceLRCallback(phi_alpha=0.5)
        assert cb._phi_gate(0.5) == 1.25

    def test_disabled(self):
        cb = ResonanceLRCallback(phi_alpha=0.0)
        assert cb._phi_gate(0.0) == 1.0
        assert cb._phi_gate(1.0) == 1.0


class TestEnvSignal:
    def test_with_reward_std(self):
        cb = ResonanceLRCallback(env_beta=0.3)
        mult = cb._env_signal({"reward_std": 1.0})
        assert mult != 1.0  # should modulate

    def test_disabled(self):
        cb = ResonanceLRCallback(env_beta=0.0)
        assert cb._env_signal({"reward_std": 5.0}) == 1.0

    def test_empty_logs(self):
        cb = ResonanceLRCallback(env_beta=0.3)
        assert cb._env_signal({}) == 1.0  # no signal → neutral


class TestOnStepBegin:
    def test_modulates_optimizer_lr(self):
        carl_fn = _make_mock_carl_fn(R_value=0.5)
        cb = ResonanceLRCallback(
            carl_reward_fn=carl_fn,
            d_model=4096,
            tokens_per_step=2400,
            phi_alpha=0.5,
            envelope_enabled=True,
        )
        opt = MockOptimizer(lr=5e-6)
        state = MockState(step=10)

        cb.on_step_begin(args=None, state=state, control=None, optimizer=opt)

        # LR should be modified from base
        assert opt.param_groups[0]["lr"] != 5e-6
        # Should be less than base (envelope decay) but not zero (phi gate boost)
        assert 0 < opt.param_groups[0]["lr"] < 5e-6

    def test_respects_min_multiplier(self):
        cb = ResonanceLRCallback(
            d_model=100, tokens_per_step=2400,  # very small T* → envelope at 0
            min_multiplier=0.2,
        )
        opt = MockOptimizer(lr=1e-5)
        state = MockState(step=1000)  # way past T*

        cb.on_step_begin(args=None, state=state, control=None, optimizer=opt)

        assert opt.param_groups[0]["lr"] >= 1e-5 * 0.2

    def test_respects_max_multiplier(self):
        cb = ResonanceLRCallback(
            phi_alpha=2.0,  # aggressive boost
            envelope_enabled=False,
            max_multiplier=1.5,
        )
        opt = MockOptimizer(lr=1e-5)
        state = MockState(step=0)

        cb.on_step_begin(args=None, state=state, control=None, optimizer=opt)

        assert opt.param_groups[0]["lr"] <= 1e-5 * 1.5


class TestOnLog:
    def test_logs_resonance_metrics(self):
        carl_fn = _make_mock_carl_fn()
        cb = ResonanceLRCallback(carl_reward_fn=carl_fn, phi_alpha=0.5)
        # Trigger R read
        cb._read_kuramoto_R()

        logs = {}
        cb.on_log(args=None, state=MockState(5), control=None, logs=logs)

        assert "lr_resonance/envelope" in logs
        assert "lr_resonance/phi_gate" in logs
        assert "lr_resonance/total_multiplier" in logs
        assert "lr_resonance/kuramoto_R" in logs
        assert "lr_resonance/t_star_steps" in logs


class TestImportable:
    def test_from_training_package(self):
        from carl_studio.training import ResonanceLRCallback as RLR
        assert RLR is not None
