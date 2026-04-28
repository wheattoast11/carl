"""Tests for carl_core.forecast_fractal — phi-spaced bands (v0.19 §4.4)."""
from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from carl_core.constants import PHI
from carl_core.errors import ValidationError
from carl_core.forecast import CoherenceForecast
from carl_core.forecast_fractal import FractalForecast, phi_scale_bands


# ----------------------------------------------------------------------
# phi_scale_bands
# ----------------------------------------------------------------------


def test_phi_scale_bands_default_returns_four() -> None:
    bands = phi_scale_bands()
    assert len(bands) == 4
    assert bands[0] == pytest.approx(1.0)
    assert bands[1] == pytest.approx(PHI)
    assert bands[2] == pytest.approx(PHI**2)
    assert bands[3] == pytest.approx(PHI**3)


def test_phi_scale_bands_strictly_monotone_increasing() -> None:
    bands = phi_scale_bands(6)
    for i in range(len(bands) - 1):
        assert bands[i] < bands[i + 1]


def test_phi_scale_bands_zero_count_raises() -> None:
    with pytest.raises(ValidationError) as ei:
        phi_scale_bands(0)
    assert ei.value.code == "carl.fractal.band_count_invalid"


def test_phi_scale_bands_negative_count_raises() -> None:
    with pytest.raises(ValidationError) as ei:
        phi_scale_bands(-3)
    assert ei.value.code == "carl.fractal.band_count_invalid"


def test_phi_scale_bands_overflow_raises() -> None:
    with pytest.raises(ValidationError) as ei:
        phi_scale_bands(20)
    assert ei.value.code == "carl.fractal.scale_overflow"


@given(num=st.integers(min_value=1, max_value=10))
@settings(max_examples=20, deadline=None)
def test_phi_scale_bands_property_first_is_one(num: int) -> None:
    bands = phi_scale_bands(num)
    assert bands[0] == pytest.approx(1.0)
    assert len(bands) == num


# ----------------------------------------------------------------------
# FractalForecast
# ----------------------------------------------------------------------


def _decaying_base(
    _state: Any, *, horizon: int, scale_band: float
) -> CoherenceForecast:
    return CoherenceForecast(
        horizon_steps=horizon,
        phi_predicted=np.linspace(0.85, 0.20, horizon),
        subspace_prior=np.zeros((horizon, 2)),
        substrate_health=np.ones(horizon),
        confidence=np.full(horizon, 0.9),
        scale_band=scale_band,
        method="linear",
    )


def _healthy_base(
    _state: Any, *, horizon: int, scale_band: float
) -> CoherenceForecast:
    return CoherenceForecast(
        horizon_steps=horizon,
        phi_predicted=np.full(horizon, 0.9),
        subspace_prior=np.zeros((horizon, 2)),
        substrate_health=np.ones(horizon),
        confidence=np.full(horizon, 0.9),
        scale_band=scale_band,
        method="linear",
    )


def test_fractal_forecast_produces_one_per_band() -> None:
    ff = FractalForecast(_decaying_base, num_bands=4)
    forecasts = ff.forecast(state={}, base_horizon=4)
    assert len(forecasts) == 4


def test_fractal_forecast_horizon_scaled_by_phi() -> None:
    ff = FractalForecast(_decaying_base, num_bands=4)
    forecasts = ff.forecast(state={}, base_horizon=4)
    assert forecasts[0].horizon_steps == 4
    assert forecasts[1].horizon_steps == max(1, int(4 * PHI))
    assert forecasts[2].horizon_steps == max(1, int(4 * PHI**2))
    assert forecasts[3].horizon_steps == max(1, int(4 * PHI**3))


def test_fractal_forecast_band_attribute_matches() -> None:
    ff = FractalForecast(_decaying_base, num_bands=4)
    forecasts = ff.forecast(state={}, base_horizon=4)
    bands = phi_scale_bands(4)
    for fc, band in zip(forecasts, bands):
        assert fc.scale_band == pytest.approx(band)


def test_fractal_forecast_zero_horizon_raises() -> None:
    ff = FractalForecast(_decaying_base, num_bands=4)
    with pytest.raises(ValidationError):
        ff.forecast(state={}, base_horizon=0)


def test_aggregate_warning_no_bands_fire_on_healthy() -> None:
    ff = FractalForecast(_healthy_base, num_bands=4)
    forecasts = ff.forecast(state={}, base_horizon=4)
    warnings = ff.aggregate_warning(forecasts, threshold=0.5)
    assert len(warnings) == 4
    assert all(v is None for v in warnings.values())


def test_aggregate_warning_fires_on_decay() -> None:
    ff = FractalForecast(_decaying_base, num_bands=4)
    forecasts = ff.forecast(state={}, base_horizon=4)
    warnings = ff.aggregate_warning(forecasts, threshold=0.5)
    assert any(v is not None for v in warnings.values())


def test_any_band_fires_returns_true_on_decay() -> None:
    ff = FractalForecast(_decaying_base, num_bands=4)
    forecasts = ff.forecast(state={}, base_horizon=4)
    assert ff.any_band_fires(forecasts, threshold=0.5) is True


def test_any_band_fires_returns_false_on_healthy() -> None:
    ff = FractalForecast(_healthy_base, num_bands=4)
    forecasts = ff.forecast(state={}, base_horizon=4)
    assert ff.any_band_fires(forecasts, threshold=0.5) is False
