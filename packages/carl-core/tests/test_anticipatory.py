"""Tests for carl_core.anticipatory — AnticipatoryGate (v0.19 §4.3)."""
from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from carl_core.anticipatory import (
    AnticipatoryGate,
    AnticipatoryPredicate,
    GateResult,
)
from carl_core.errors import ValidationError
from carl_core.forecast import CoherenceForecast


def _healthy_forecast(horizon: int = 4) -> CoherenceForecast:
    return CoherenceForecast(
        horizon_steps=horizon,
        phi_predicted=np.full(horizon, 0.85),
        subspace_prior=np.zeros((horizon, 2)),
        substrate_health=np.ones(horizon),
        confidence=np.full(horizon, 0.9),
        scale_band=1.0,
        method="linear",
    )


def _decaying_forecast(horizon: int = 4) -> CoherenceForecast:
    return CoherenceForecast(
        horizon_steps=horizon,
        phi_predicted=np.linspace(0.85, 0.05, horizon),
        subspace_prior=np.zeros((horizon, 2)),
        substrate_health=np.linspace(1.0, 0.3, horizon),
        confidence=np.full(horizon, 0.9),
        scale_band=1.0,
        method="lyapunov",
    )


# ----------------------------------------------------------------------
# construction
# ----------------------------------------------------------------------


def test_gate_construct_with_defaults() -> None:
    gate = AnticipatoryGate(horizon=4, threshold=0.5, drift_threshold=0.1)
    assert gate.horizon == 4
    assert gate.threshold == 0.5
    assert gate.drift_threshold == 0.1


def test_gate_threshold_out_of_range_raises() -> None:
    with pytest.raises(ValidationError) as ei:
        AnticipatoryGate(horizon=4, threshold=1.5, drift_threshold=0.1)
    assert ei.value.code == "carl.gate.anticipatory_threshold_invalid"
    with pytest.raises(ValidationError):
        AnticipatoryGate(horizon=4, threshold=-0.1, drift_threshold=0.1)


def test_gate_horizon_must_be_positive() -> None:
    with pytest.raises(ValidationError) as ei:
        AnticipatoryGate(horizon=0, threshold=0.5, drift_threshold=0.1)
    assert ei.value.code == "carl.gate.anticipatory_threshold_invalid"


# ----------------------------------------------------------------------
# evaluate: gate fires on warning, allows on healthy forecast
# ----------------------------------------------------------------------


def test_gate_allows_when_forecast_healthy() -> None:
    gate = AnticipatoryGate(horizon=4, threshold=0.5, drift_threshold=0.1)
    result = gate.evaluate({}, forecast_fn=lambda _: _healthy_forecast())
    assert isinstance(result, GateResult)
    assert result.allowed is True
    assert result.warning_step is None


def test_gate_denies_when_forecast_warns() -> None:
    gate = AnticipatoryGate(horizon=4, threshold=0.5, drift_threshold=0.1)
    result = gate.evaluate({}, forecast_fn=lambda _: _decaying_forecast())
    assert result.allowed is False
    assert result.warning_step is not None
    assert result.warning_step >= 0
    assert "anticipatory" in result.reason.lower() or "warning" in result.reason.lower()


def test_gate_denies_when_drift_too_negative() -> None:
    # Forecast above threshold everywhere, but drift is steeply negative
    horizon = 5
    fc = CoherenceForecast(
        horizon_steps=horizon,
        phi_predicted=np.array([0.95, 0.9, 0.85, 0.8, 0.75]),
        subspace_prior=np.zeros((horizon, 2)),
        substrate_health=np.ones(horizon),
        confidence=np.full(horizon, 0.9),
        scale_band=1.0,
        method="lyapunov",
    )
    gate = AnticipatoryGate(horizon=5, threshold=0.5, drift_threshold=0.05)
    result = gate.evaluate({}, forecast_fn=lambda _: fc)
    # phi never crosses 0.5, but drift is sharply negative -> denied
    assert result.allowed is False


# ----------------------------------------------------------------------
# graceful failure on missing forecast_fn / errors
# ----------------------------------------------------------------------


def test_gate_raises_when_forecast_fn_returns_none() -> None:
    gate = AnticipatoryGate(horizon=4, threshold=0.5, drift_threshold=0.1)
    with pytest.raises(ValidationError) as ei:
        gate.evaluate({}, forecast_fn=lambda _: None)  # type: ignore[arg-type, return-value]
    assert ei.value.code == "carl.gate.forecast_unavailable"


def test_gate_raises_when_forecast_fn_throws() -> None:
    gate = AnticipatoryGate(horizon=4, threshold=0.5, drift_threshold=0.1)

    def boom(_: Any) -> CoherenceForecast:
        raise RuntimeError("provider down")

    with pytest.raises(ValidationError) as ei:
        gate.evaluate({}, forecast_fn=boom)
    assert ei.value.code == "carl.gate.forecast_unavailable"


def test_gate_raises_when_forecast_horizon_mismatch() -> None:
    gate = AnticipatoryGate(horizon=4, threshold=0.5, drift_threshold=0.1)
    # forecast horizon=2, gate horizon=4 -> mismatch
    short = CoherenceForecast(
        horizon_steps=2,
        phi_predicted=np.array([0.8, 0.8]),
        subspace_prior=np.zeros((2, 2)),
        substrate_health=np.ones(2),
        confidence=np.full(2, 0.9),
        scale_band=1.0,
        method="linear",
    )
    with pytest.raises(ValidationError) as ei:
        gate.evaluate({}, forecast_fn=lambda _: short)
    assert ei.value.code == "carl.gate.forecast_unavailable"


# ----------------------------------------------------------------------
# AnticipatoryPredicate exposed for composition with carl_studio gates
# ----------------------------------------------------------------------


def test_predicate_check_returns_allowed_tuple() -> None:
    pred = AnticipatoryPredicate(
        forecast=_healthy_forecast(),
        threshold=0.5,
        drift_threshold=0.1,
    )
    allowed, reason = pred.check()
    assert allowed is True
    assert isinstance(reason, str)


def test_predicate_check_denies_decaying() -> None:
    pred = AnticipatoryPredicate(
        forecast=_decaying_forecast(),
        threshold=0.5,
        drift_threshold=0.1,
    )
    allowed, reason = pred.check()
    assert allowed is False
    assert isinstance(reason, str)


def test_predicate_name_is_stable() -> None:
    pred = AnticipatoryPredicate(
        forecast=_healthy_forecast(),
        threshold=0.5,
        drift_threshold=0.1,
    )
    assert pred.name.startswith("anticipatory")
