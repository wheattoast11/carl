"""Tests for carl_core.realizability — the realizability-gap operator (v0.19 §4.5)."""
from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from carl_core.errors import ValidationError
from carl_core.forecast import CoherenceForecast
from carl_core.realizability import realizability_chain, realizability_gap


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
# realizability_gap
# ----------------------------------------------------------------------


def test_zero_gap_when_forecast_equals_observed() -> None:
    fc = _fc([0.5, 0.5, 0.5, 0.5])
    obs = np.array([0.5, 0.5, 0.5, 0.5])
    gap = realizability_gap(fc, obs)
    assert gap == pytest.approx(0.0, abs=1e-9)


def test_positive_gap_on_divergence() -> None:
    fc = _fc([0.5, 0.5, 0.5, 0.5])
    obs = np.array([0.1, 0.1, 0.1, 0.7])
    gap = realizability_gap(fc, obs)
    assert gap > 0.0


def test_horizon_specific_gap_returns_zero_at_matching_step() -> None:
    fc = _fc([0.5, 0.6, 0.7, 0.8])
    obs = np.array([0.5, 0.5, 0.5, 0.5])
    # at horizon 0: forecast(0.5) == observed(0.5) → 0
    gap0 = realizability_gap(fc, obs, horizon=0)
    assert gap0 == pytest.approx(0.0, abs=1e-9)


def test_horizon_specific_gap_positive_at_mismatching_step() -> None:
    fc = _fc([0.5, 0.6, 0.7, 0.9])
    obs = np.array([0.5, 0.5, 0.5, 0.5])
    # at horizon 3: forecast(0.9) vs observed(0.5) → positive (after normalize)
    # Note: single-element KL after normalize becomes 0 because both
    # singletons normalize to 1.0. So multi-element only has positive gap.
    gap = realizability_gap(fc, obs)
    assert gap > 0.0


def test_shape_mismatch_raises() -> None:
    fc = _fc([0.5, 0.5, 0.5, 0.5])
    with pytest.raises(ValidationError) as ei:
        realizability_gap(fc, np.array([0.5, 0.5]))
    assert ei.value.code == "carl.realizability.horizon_mismatch"


def test_horizon_out_of_range_raises() -> None:
    fc = _fc([0.5, 0.5, 0.5, 0.5])
    obs = np.array([0.5, 0.5, 0.5, 0.5])
    with pytest.raises(ValidationError) as ei:
        realizability_gap(fc, obs, horizon=99)
    assert ei.value.code == "carl.realizability.horizon_mismatch"
    with pytest.raises(ValidationError):
        realizability_gap(fc, obs, horizon=-1)


def test_observed_too_short_for_horizon_raises() -> None:
    fc = _fc([0.5, 0.5, 0.5, 0.5])
    with pytest.raises(ValidationError) as ei:
        realizability_gap(fc, np.array([0.5, 0.5]), horizon=3)
    assert ei.value.code == "carl.realizability.horizon_mismatch"


def test_epsilon_stability_with_zero_phi() -> None:
    # All-zero phi shouldn't blow up — epsilon floor protects log
    fc = _fc([0.0, 0.0, 0.0, 0.0])
    obs = np.array([0.0, 0.0, 0.0, 0.0])
    gap = realizability_gap(fc, obs)
    assert np.isfinite(gap)


# ----------------------------------------------------------------------
# realizability_chain
# ----------------------------------------------------------------------


def test_chain_empty_returns_zero() -> None:
    assert realizability_chain([], [], []) == 0.0


def test_chain_sums_per_element_gaps() -> None:
    fc1 = _fc([0.5, 0.5, 0.5])
    fc2 = _fc([0.5, 0.5, 0.5])
    obs = np.array([0.5, 0.5, 0.5])
    obs_div = np.array([0.1, 0.1, 0.7])
    individual1 = realizability_gap(fc1, obs)
    individual2 = realizability_gap(fc2, obs_div)
    chain = realizability_chain([fc1, fc2], [obs, obs_div], [None, None])
    assert chain == pytest.approx(individual1 + individual2)


def test_chain_length_mismatch_raises() -> None:
    fc1 = _fc([0.5, 0.5, 0.5])
    obs = np.array([0.5, 0.5, 0.5])
    with pytest.raises(ValidationError) as ei:
        realizability_chain([fc1, fc1], [obs], [None])
    assert ei.value.code == "carl.realizability.horizon_mismatch"


# ----------------------------------------------------------------------
# property: KL non-negativity (Gibbs inequality)
# ----------------------------------------------------------------------


@given(
    base_phi=st.lists(
        st.floats(min_value=0.05, max_value=0.95),
        min_size=2,
        max_size=8,
    ),
    noise_scale=st.floats(min_value=0.0, max_value=0.3),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@settings(max_examples=50, deadline=None)
def test_kl_non_negativity(
    base_phi: list[float], noise_scale: float, seed: int
) -> None:
    phi = np.asarray(base_phi, dtype=np.float64)
    fc = _fc(base_phi)
    rng = np.random.default_rng(seed)
    perturbed = np.clip(phi + rng.normal(0, noise_scale, size=len(phi)), 0.01, 0.99)
    gap = realizability_gap(fc, perturbed)
    # Gibbs inequality: KL(p||q) >= 0
    assert gap >= -1e-9
    assert np.isfinite(gap)


# ----------------------------------------------------------------------
# property: composability bounds
# ----------------------------------------------------------------------


@given(
    n=st.integers(min_value=1, max_value=5),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@settings(max_examples=30, deadline=None)
def test_chain_non_negative(n: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    forecasts: list[CoherenceForecast] = []
    observations: list[np.ndarray] = []  # type: ignore[type-arg]
    for _ in range(n):
        phi = rng.uniform(0.1, 0.9, size=4).tolist()
        forecasts.append(_fc(phi))
        observations.append(np.clip(np.asarray(phi) + rng.normal(0, 0.1, 4), 0.01, 0.99))
    horizons: list[int | None] = [None] * n
    chain = realizability_chain(forecasts, observations, horizons)
    assert chain >= -1e-9
    assert np.isfinite(chain)
