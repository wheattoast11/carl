"""Tests for v0.19.2 AnticipatoryReward (studio PAID minimal)."""
from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from carl_core import CoherenceForecast
from carl_studio.training.rewards.anticipatory import (
    AnticipatoryReward,
    AnticipatoryRewardComponents,
    make_anticipatory_reward,
)


def _fc(phi: list[float]) -> CoherenceForecast:
    h = len(phi)
    return CoherenceForecast(
        horizon_steps=h,
        phi_predicted=np.asarray(phi, dtype=np.float64),
        subspace_prior=np.zeros((h, 2)),
        substrate_health=np.ones(h),
        confidence=np.full(h, 0.9),
        scale_band=1.0,
        method="linear",
    )


# ----------------------------------------------------------------------
# basic reward computation
# ----------------------------------------------------------------------


def test_zero_gap_yields_baseline_reward() -> None:
    reward = AnticipatoryReward(baseline=0.5, scale=1.0)
    fc = _fc([0.5, 0.5, 0.5, 0.5])
    obs = np.array([0.5, 0.5, 0.5, 0.5])
    r, meta = reward(fc, obs)
    assert r == pytest.approx(0.5, abs=1e-9)
    assert meta.realizability_gap == pytest.approx(0.0, abs=1e-9)


def test_positive_gap_lowers_reward_below_baseline() -> None:
    reward = AnticipatoryReward(baseline=1.0, scale=2.0)
    fc = _fc([0.5, 0.5, 0.5, 0.5])
    obs = np.array([0.1, 0.1, 0.1, 0.7])
    r, meta = reward(fc, obs)
    assert r < 1.0
    assert meta.realizability_gap > 0.0


def test_returns_components_dataclass() -> None:
    reward = AnticipatoryReward()
    fc = _fc([0.5, 0.6, 0.7])
    obs = np.array([0.5, 0.5, 0.5])
    _r, meta = reward(fc, obs)
    assert isinstance(meta, AnticipatoryRewardComponents)
    assert meta.scale == 1.0
    assert meta.baseline == 0.0


# ----------------------------------------------------------------------
# clipping
# ----------------------------------------------------------------------


def test_clip_min_floors_reward() -> None:
    reward = AnticipatoryReward(scale=1000.0, clip_min=-5.0)
    fc = _fc([0.5, 0.5, 0.5, 0.5])
    obs = np.array([0.99, 0.01, 0.99, 0.01])
    r, _ = reward(fc, obs)
    assert r >= -5.0


def test_clip_max_ceils_reward() -> None:
    reward = AnticipatoryReward(baseline=10.0, clip_max=2.0)
    fc = _fc([0.5, 0.5])
    obs = np.array([0.5, 0.5])
    r, _ = reward(fc, obs)
    assert r <= 2.0


# ----------------------------------------------------------------------
# horizon-specific
# ----------------------------------------------------------------------


def test_horizon_specific_call_uses_only_that_step() -> None:
    reward = AnticipatoryReward()
    fc = _fc([0.5, 0.6, 0.7, 0.8])
    obs = np.array([0.5, 0.5, 0.5, 0.5])
    # at horizon 0: forecast(0.5) == observed(0.5), gap == 0
    _r0, meta0 = reward(fc, obs, horizon=0)
    assert meta0.realizability_gap == pytest.approx(0.0, abs=1e-9)
    assert meta0.horizon == 0


# ----------------------------------------------------------------------
# chain_reward
# ----------------------------------------------------------------------


def test_chain_reward_zero_for_perfect_forecasts() -> None:
    reward = AnticipatoryReward(baseline=0.0, scale=1.0)
    fc1 = _fc([0.5, 0.5, 0.5])
    fc2 = _fc([0.6, 0.6, 0.6])
    r, meta = reward.chain_reward(
        [fc1, fc2],
        [np.array([0.5, 0.5, 0.5]), np.array([0.6, 0.6, 0.6])],
        [None, None],
    )
    assert r == pytest.approx(0.0, abs=1e-9)
    assert meta.realizability_gap == pytest.approx(0.0, abs=1e-9)


def test_chain_reward_aggregates_gaps() -> None:
    reward = AnticipatoryReward(baseline=0.0, scale=1.0)
    fc1 = _fc([0.5, 0.5, 0.5])
    fc2 = _fc([0.5, 0.5, 0.5])
    obs_match = np.array([0.5, 0.5, 0.5])
    obs_diverge = np.array([0.1, 0.1, 0.7])
    r_single, _ = reward(fc2, obs_diverge)
    r_chain, _ = reward.chain_reward(
        [fc1, fc2], [obs_match, obs_diverge], [None, None]
    )
    # chain reward = baseline (0) - scale (1) * sum_of_gaps
    # which equals r_single because obs_match has zero gap
    assert r_chain == pytest.approx(r_single, abs=1e-9)


# ----------------------------------------------------------------------
# factory + immutability
# ----------------------------------------------------------------------


def test_factory_constructs_correctly() -> None:
    r = make_anticipatory_reward(scale=2.5, baseline=0.5, clip_min=-1.0, clip_max=1.0)
    assert r.scale == 2.5
    assert r.baseline == 0.5
    assert r.clip_min == -1.0
    assert r.clip_max == 1.0


def test_reward_is_frozen() -> None:
    r = AnticipatoryReward(scale=1.0)
    with pytest.raises((AttributeError, TypeError)):
        r.scale = 2.0  # type: ignore[misc]


# ----------------------------------------------------------------------
# property: realizability gap is monotone in observed-divergence
# ----------------------------------------------------------------------


@given(
    base=st.floats(min_value=0.2, max_value=0.8),
    n=st.integers(min_value=2, max_value=8),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@settings(max_examples=30, deadline=None)
def test_reward_decreases_as_observed_diverges(
    base: float, n: int, seed: int
) -> None:
    """Larger divergence ⇒ larger gap ⇒ smaller reward (monotone)."""
    reward = AnticipatoryReward(baseline=0.0, scale=1.0)
    rng = np.random.default_rng(seed)
    forecast_phi = np.full(n, base)
    fc = CoherenceForecast(
        horizon_steps=n,
        phi_predicted=forecast_phi,
        subspace_prior=np.zeros((n, 2)),
        substrate_health=np.ones(n),
        confidence=np.full(n, 0.9),
        scale_band=1.0,
        method="linear",
    )
    obs_match = forecast_phi.copy()
    obs_perturbed = np.clip(
        forecast_phi + rng.normal(0, 0.2, n), 0.01, 0.99
    )
    r_match, _ = reward(fc, obs_match)
    r_pert, _ = reward(fc, obs_perturbed)
    assert r_match >= r_pert - 1e-9


# ----------------------------------------------------------------------
# property: reward is non-positive when scale > 0 and baseline = 0
# ----------------------------------------------------------------------


@given(
    base=st.floats(min_value=0.1, max_value=0.9),
    n=st.integers(min_value=2, max_value=6),
)
@settings(max_examples=20, deadline=None)
def test_reward_non_positive_with_zero_baseline_and_positive_scale(
    base: float, n: int
) -> None:
    reward = AnticipatoryReward(baseline=0.0, scale=1.0)
    fc = CoherenceForecast(
        horizon_steps=n,
        phi_predicted=np.full(n, base),
        subspace_prior=np.zeros((n, 2)),
        substrate_health=np.ones(n),
        confidence=np.full(n, 0.9),
        scale_band=1.0,
        method="linear",
    )
    obs = np.full(n, base + 0.1)  # diverges
    r, _ = reward(fc, obs)
    # gap >= 0, scale > 0, baseline = 0 ⇒ reward <= 0
    assert r <= 1e-9
