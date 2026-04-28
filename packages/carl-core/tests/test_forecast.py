"""Tests for carl_core.forecast — CoherenceForecast primitive (v0.19 §4.1)."""
from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from carl_core.errors import ValidationError
from carl_core.forecast import CoherenceForecast


# ----------------------------------------------------------------------
# construction validation
# ----------------------------------------------------------------------


def _ok_forecast(horizon: int = 4, scale: float = 1.0, method: str = "linear") -> CoherenceForecast:
    return CoherenceForecast(
        horizon_steps=horizon,
        phi_predicted=np.linspace(0.5, 0.6, horizon),
        subspace_prior=np.zeros((horizon, 3)),
        substrate_health=np.ones(horizon),
        confidence=np.full(horizon, 0.9),
        scale_band=scale,
        method=method,
    )


def test_construct_valid_forecast() -> None:
    fc = _ok_forecast()
    assert fc.horizon_steps == 4
    assert fc.scale_band == 1.0
    assert fc.method == "linear"
    assert fc.phi_predicted.shape == (4,)
    assert fc.subspace_prior.shape == (4, 3)


def test_horizon_must_be_positive() -> None:
    with pytest.raises(ValidationError) as ei:
        CoherenceForecast(
            horizon_steps=0,
            phi_predicted=np.array([]),
            subspace_prior=np.zeros((0, 3)),
            substrate_health=np.array([]),
            confidence=np.array([]),
            scale_band=1.0,
            method="linear",
        )
    assert ei.value.code == "carl.forecast.horizon_invalid"


def test_negative_horizon_rejected() -> None:
    with pytest.raises(ValidationError) as ei:
        CoherenceForecast(
            horizon_steps=-2,
            phi_predicted=np.array([0.5, 0.5]),
            subspace_prior=np.zeros((2, 3)),
            substrate_health=np.ones(2),
            confidence=np.full(2, 0.5),
            scale_band=1.0,
            method="linear",
        )
    assert ei.value.code == "carl.forecast.horizon_invalid"


def test_array_length_must_match_horizon() -> None:
    with pytest.raises(ValidationError) as ei:
        CoherenceForecast(
            horizon_steps=4,
            phi_predicted=np.linspace(0.5, 0.6, 3),  # mismatch
            subspace_prior=np.zeros((4, 3)),
            substrate_health=np.ones(4),
            confidence=np.full(4, 0.9),
            scale_band=1.0,
            method="linear",
        )
    assert ei.value.code == "carl.forecast.horizon_invalid"


def test_unknown_method_rejected() -> None:
    with pytest.raises(ValidationError) as ei:
        CoherenceForecast(
            horizon_steps=4,
            phi_predicted=np.linspace(0.5, 0.6, 4),
            subspace_prior=np.zeros((4, 3)),
            substrate_health=np.ones(4),
            confidence=np.full(4, 0.9),
            scale_band=1.0,
            method="bogus",
        )
    assert ei.value.code == "carl.forecast.method_unknown"


def test_confidence_must_be_in_unit_interval() -> None:
    with pytest.raises(ValidationError) as ei:
        CoherenceForecast(
            horizon_steps=2,
            phi_predicted=np.array([0.5, 0.6]),
            subspace_prior=np.zeros((2, 3)),
            substrate_health=np.array([0.9, 0.9]),
            confidence=np.array([0.5, 1.5]),  # > 1
            scale_band=1.0,
            method="linear",
        )
    assert ei.value.code == "carl.forecast.uncertainty_overflow"


def test_negative_confidence_rejected() -> None:
    with pytest.raises(ValidationError) as ei:
        CoherenceForecast(
            horizon_steps=2,
            phi_predicted=np.array([0.5, 0.6]),
            subspace_prior=np.zeros((2, 3)),
            substrate_health=np.array([0.9, 0.9]),
            confidence=np.array([-0.1, 0.5]),
            scale_band=1.0,
            method="linear",
        )
    assert ei.value.code == "carl.forecast.uncertainty_overflow"


# ----------------------------------------------------------------------
# early_warning
# ----------------------------------------------------------------------


def test_early_warning_no_warning_when_above_threshold() -> None:
    fc = CoherenceForecast(
        horizon_steps=5,
        phi_predicted=np.full(5, 0.8),
        subspace_prior=np.zeros((5, 2)),
        substrate_health=np.ones(5),
        confidence=np.full(5, 0.9),
        scale_band=1.0,
        method="linear",
    )
    assert fc.early_warning(threshold=0.5) is None


def test_early_warning_fires_at_step_zero() -> None:
    fc = CoherenceForecast(
        horizon_steps=3,
        phi_predicted=np.array([0.1, 0.8, 0.8]),
        subspace_prior=np.zeros((3, 2)),
        substrate_health=np.ones(3),
        confidence=np.full(3, 0.9),
        scale_band=1.0,
        method="linear",
    )
    assert fc.early_warning(threshold=0.5) == 0


def test_early_warning_fires_at_last_step() -> None:
    fc = CoherenceForecast(
        horizon_steps=3,
        phi_predicted=np.array([0.8, 0.8, 0.1]),
        subspace_prior=np.zeros((3, 2)),
        substrate_health=np.ones(3),
        confidence=np.full(3, 0.9),
        scale_band=1.0,
        method="linear",
    )
    assert fc.early_warning(threshold=0.5) == 2


def test_early_warning_threshold_zero_never_fires() -> None:
    fc = CoherenceForecast(
        horizon_steps=3,
        phi_predicted=np.array([0.0, 0.0, 0.0]),
        subspace_prior=np.zeros((3, 2)),
        substrate_health=np.ones(3),
        confidence=np.full(3, 0.9),
        scale_band=1.0,
        method="linear",
    )
    # threshold=0 means "warn when phi < 0" — never true for phi >= 0
    assert fc.early_warning(threshold=0.0) is None


def test_early_warning_threshold_one_fires_immediately() -> None:
    fc = _ok_forecast()
    # threshold=1 means "warn when phi < 1" — true everywhere unless phi is 1.0
    assert fc.early_warning(threshold=1.0) == 0


def test_early_warning_threshold_out_of_range_raises() -> None:
    fc = _ok_forecast()
    with pytest.raises(ValidationError) as ei:
        fc.early_warning(threshold=-0.1)
    assert ei.value.code == "carl.forecast.uncertainty_overflow"
    with pytest.raises(ValidationError):
        fc.early_warning(threshold=1.5)


# ----------------------------------------------------------------------
# lyapunov_drift
# ----------------------------------------------------------------------


def test_lyapunov_drift_positive_for_growing_phi() -> None:
    fc = CoherenceForecast(
        horizon_steps=10,
        phi_predicted=np.linspace(0.1, 0.9, 10),
        subspace_prior=np.zeros((10, 2)),
        substrate_health=np.ones(10),
        confidence=np.full(10, 0.9),
        scale_band=1.0,
        method="linear",
    )
    assert fc.lyapunov_drift() > 0.0


def test_lyapunov_drift_negative_for_shrinking_phi() -> None:
    fc = CoherenceForecast(
        horizon_steps=10,
        phi_predicted=np.linspace(0.9, 0.1, 10),
        subspace_prior=np.zeros((10, 2)),
        substrate_health=np.ones(10),
        confidence=np.full(10, 0.9),
        scale_band=1.0,
        method="linear",
    )
    assert fc.lyapunov_drift() < 0.0


def test_lyapunov_drift_zero_for_constant_phi() -> None:
    fc = CoherenceForecast(
        horizon_steps=10,
        phi_predicted=np.full(10, 0.5),
        subspace_prior=np.zeros((10, 2)),
        substrate_health=np.ones(10),
        confidence=np.full(10, 0.9),
        scale_band=1.0,
        method="linear",
    )
    assert fc.lyapunov_drift() == pytest.approx(0.0, abs=1e-12)


def test_lyapunov_drift_zero_series_returns_zero() -> None:
    fc = CoherenceForecast(
        horizon_steps=5,
        phi_predicted=np.zeros(5),
        subspace_prior=np.zeros((5, 2)),
        substrate_health=np.ones(5),
        confidence=np.full(5, 0.9),
        scale_band=1.0,
        method="linear",
    )
    # all-zero is constant; finite-time Lyapunov is 0 by convention
    assert fc.lyapunov_drift() == pytest.approx(0.0, abs=1e-12)


def test_lyapunov_drift_single_step_returns_zero() -> None:
    fc = _ok_forecast(horizon=1)
    assert fc.lyapunov_drift() == pytest.approx(0.0, abs=1e-12)


# ----------------------------------------------------------------------
# confidence_interval
# ----------------------------------------------------------------------


def test_confidence_interval_default_alpha() -> None:
    fc = _ok_forecast()
    lo, hi = fc.confidence_interval()
    assert lo.shape == fc.phi_predicted.shape
    assert hi.shape == fc.phi_predicted.shape
    assert np.all(lo <= fc.phi_predicted)
    assert np.all(hi >= fc.phi_predicted)


def test_confidence_interval_alpha_bounds() -> None:
    fc = _ok_forecast()
    lo1, hi1 = fc.confidence_interval(alpha=0.05)
    lo2, hi2 = fc.confidence_interval(alpha=0.32)
    # smaller alpha (95% CI) -> wider interval than larger alpha (68% CI)
    assert np.all((hi1 - lo1) >= (hi2 - lo2) - 1e-9)


def test_confidence_interval_alpha_out_of_range_raises() -> None:
    fc = _ok_forecast()
    with pytest.raises(ValidationError) as ei:
        fc.confidence_interval(alpha=0.0)
    assert ei.value.code == "carl.forecast.uncertainty_overflow"
    with pytest.raises(ValidationError):
        fc.confidence_interval(alpha=1.0)
    with pytest.raises(ValidationError):
        fc.confidence_interval(alpha=-0.1)


# ----------------------------------------------------------------------
# phi_at
# ----------------------------------------------------------------------


def test_phi_at_in_range() -> None:
    fc = CoherenceForecast(
        horizon_steps=4,
        phi_predicted=np.array([0.1, 0.2, 0.3, 0.4]),
        subspace_prior=np.zeros((4, 2)),
        substrate_health=np.ones(4),
        confidence=np.full(4, 0.9),
        scale_band=1.0,
        method="linear",
    )
    assert fc.phi_at(0) == pytest.approx(0.1)
    assert fc.phi_at(3) == pytest.approx(0.4)


def test_phi_at_out_of_range_raises() -> None:
    fc = _ok_forecast()
    with pytest.raises(ValidationError) as ei:
        fc.phi_at(99)
    assert ei.value.code == "carl.forecast.horizon_invalid"
    with pytest.raises(ValidationError):
        fc.phi_at(-1)


# ----------------------------------------------------------------------
# serialization round-trip
# ----------------------------------------------------------------------


def test_to_dict_round_trip() -> None:
    fc = _ok_forecast()
    payload = fc.to_dict()
    rebuilt = CoherenceForecast.from_dict(payload)
    assert rebuilt.horizon_steps == fc.horizon_steps
    assert rebuilt.scale_band == fc.scale_band
    assert rebuilt.method == fc.method
    np.testing.assert_array_equal(rebuilt.phi_predicted, fc.phi_predicted)
    np.testing.assert_array_equal(rebuilt.subspace_prior, fc.subspace_prior)
    np.testing.assert_array_equal(rebuilt.substrate_health, fc.substrate_health)
    np.testing.assert_array_equal(rebuilt.confidence, fc.confidence)


# ----------------------------------------------------------------------
# property: lyapunov drift sign matches monotonicity
# ----------------------------------------------------------------------


@given(
    base=st.floats(min_value=0.05, max_value=0.5),
    delta=st.floats(min_value=-0.4, max_value=0.4),
)
@settings(max_examples=50, deadline=None)
def test_lyapunov_drift_sign_matches_phi_monotone(base: float, delta: float) -> None:
    horizon = 8
    phi = np.linspace(base, base + delta, horizon)
    phi = np.clip(phi, 1e-6, 1.0)  # keep strictly positive for log
    fc = CoherenceForecast(
        horizon_steps=horizon,
        phi_predicted=phi,
        subspace_prior=np.zeros((horizon, 2)),
        substrate_health=np.ones(horizon),
        confidence=np.full(horizon, 0.9),
        scale_band=1.0,
        method="linear",
    )
    drift = fc.lyapunov_drift()
    if delta > 1e-3:
        assert drift > -1e-9
    elif delta < -1e-3:
        assert drift < 1e-9
    else:
        assert abs(drift) < 0.5  # near-constant -> small magnitude
    assert math.isfinite(drift)
