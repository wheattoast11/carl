"""Tests for carl_core.substrate — SubstrateChannel/SubstrateState (v0.19 §4.2)."""
from __future__ import annotations

import pytest

from carl_core.errors import ValidationError
from carl_core.substrate import SubstrateChannel, SubstrateState


# ----------------------------------------------------------------------
# SubstrateChannel construction
# ----------------------------------------------------------------------


def test_construct_channel() -> None:
    ch = SubstrateChannel(
        name="memory",
        health=0.85,
        drift_rate=-0.01,
        drift_acceleration=-0.001,
    )
    assert ch.name == "memory"
    assert ch.health == 0.85
    # leading_indicator is computed
    assert ch.leading_indicator is False  # dPsi/dt < 0


def test_construct_channel_health_out_of_range_raises() -> None:
    with pytest.raises(ValidationError) as ei:
        SubstrateChannel(name="memory", health=1.5, drift_rate=0.0, drift_acceleration=0.0)
    assert ei.value.code == "carl.substrate.channel_unknown"
    with pytest.raises(ValidationError):
        SubstrateChannel(name="memory", health=-0.1, drift_rate=0.0, drift_acceleration=0.0)


def test_channel_name_required() -> None:
    with pytest.raises(ValidationError) as ei:
        SubstrateChannel(name="", health=0.5, drift_rate=0.0, drift_acceleration=0.0)
    assert ei.value.code == "carl.substrate.channel_unknown"


# ----------------------------------------------------------------------
# leading_indicator algebra: d²Ψ/dt² < 0  AND  dΨ/dt > 0  ⇒  imminent collapse
# ----------------------------------------------------------------------


def test_leading_indicator_fires_when_derivative_positive_acceleration_negative() -> None:
    ch = SubstrateChannel(
        name="memory",
        health=0.9,
        drift_rate=0.05,         # dPsi/dt > 0 (rising)
        drift_acceleration=-0.01,  # d²Psi/dt² < 0 (decelerating)
    )
    assert ch.leading_indicator is True


def test_leading_indicator_quiet_when_acceleration_positive() -> None:
    ch = SubstrateChannel(
        name="memory",
        health=0.9,
        drift_rate=0.05,
        drift_acceleration=0.01,
    )
    assert ch.leading_indicator is False


def test_leading_indicator_quiet_when_derivative_negative() -> None:
    ch = SubstrateChannel(
        name="memory",
        health=0.9,
        drift_rate=-0.01,
        drift_acceleration=-0.05,
    )
    assert ch.leading_indicator is False


def test_leading_indicator_quiet_at_exact_zero_boundary() -> None:
    ch = SubstrateChannel(
        name="memory",
        health=0.9,
        drift_rate=0.0,           # boundary; not strictly > 0
        drift_acceleration=-0.01,
    )
    assert ch.leading_indicator is False


# ----------------------------------------------------------------------
# SubstrateState
# ----------------------------------------------------------------------


def _make_state(
    *,
    healthy: bool = True,
    timestamp: float = 0.0,
) -> SubstrateState:
    if healthy:
        chans = {
            "memory": SubstrateChannel("memory", 0.9, -0.001, 0.0001),
            "compute": SubstrateChannel("compute", 0.95, 0.0, 0.0),
        }
        psi = 0.92
    else:
        chans = {
            "memory": SubstrateChannel("memory", 0.4, 0.05, -0.02),  # leading-indicator
            "compute": SubstrateChannel("compute", 0.95, 0.0, 0.0),
        }
        psi = 0.6
    return SubstrateState(timestamp=timestamp, channels=chans, overall_psi=psi)


def test_construct_substrate_state() -> None:
    s = _make_state()
    assert s.timestamp == 0.0
    assert "memory" in s.channels
    assert s.overall_psi == 0.92


def test_critical_channels_empty_for_healthy_state() -> None:
    s = _make_state(healthy=True)
    assert s.critical_channels() == []


def test_critical_channels_lists_indicator_firing_channels() -> None:
    s = _make_state(healthy=False)
    crit = s.critical_channels()
    assert "memory" in crit
    assert "compute" not in crit


def test_overall_psi_must_be_in_unit_interval() -> None:
    chans = {"a": SubstrateChannel("a", 0.5, 0.0, 0.0)}
    with pytest.raises(ValidationError) as ei:
        SubstrateState(timestamp=0.0, channels=chans, overall_psi=1.2)
    assert ei.value.code == "carl.substrate.channel_unknown"


# ----------------------------------------------------------------------
# lyapunov_exponent_estimate
# ----------------------------------------------------------------------


def test_lyapunov_exponent_empty_history_raises() -> None:
    s = _make_state()
    with pytest.raises(ValidationError) as ei:
        s.lyapunov_exponent_estimate([])
    assert ei.value.code == "carl.substrate.history_too_short"


def test_lyapunov_exponent_short_history_raises() -> None:
    s = _make_state(timestamp=2.0)
    with pytest.raises(ValidationError):
        s.lyapunov_exponent_estimate([_make_state(timestamp=0.0)])


def test_lyapunov_exponent_for_decaying_psi() -> None:
    # Psi decays: 1.0 -> 0.5 -> 0.25 -> 0.125 (each step halves -> negative lambda)
    history: list[SubstrateState] = []
    for i, psi in enumerate([1.0, 0.5, 0.25]):
        chans = {"a": SubstrateChannel("a", 0.5, 0.0, 0.0)}
        history.append(SubstrateState(timestamp=float(i), channels=chans, overall_psi=psi))
    current_chans = {"a": SubstrateChannel("a", 0.5, 0.0, 0.0)}
    current = SubstrateState(timestamp=3.0, channels=current_chans, overall_psi=0.125)
    lam = current.lyapunov_exponent_estimate(history)
    assert lam < 0.0


def test_lyapunov_exponent_for_growing_psi() -> None:
    # Psi grows: 0.1 -> 0.2 -> 0.4 -> 0.8 (positive lambda)
    history: list[SubstrateState] = []
    for i, psi in enumerate([0.1, 0.2, 0.4]):
        chans = {"a": SubstrateChannel("a", 0.5, 0.0, 0.0)}
        history.append(SubstrateState(timestamp=float(i), channels=chans, overall_psi=psi))
    current_chans = {"a": SubstrateChannel("a", 0.5, 0.0, 0.0)}
    current = SubstrateState(timestamp=3.0, channels=current_chans, overall_psi=0.8)
    lam = current.lyapunov_exponent_estimate(history)
    assert lam > 0.0


def test_lyapunov_exponent_for_constant_psi() -> None:
    history: list[SubstrateState] = []
    for i in range(3):
        chans = {"a": SubstrateChannel("a", 0.5, 0.0, 0.0)}
        history.append(SubstrateState(timestamp=float(i), channels=chans, overall_psi=0.5))
    current_chans = {"a": SubstrateChannel("a", 0.5, 0.0, 0.0)}
    current = SubstrateState(timestamp=3.0, channels=current_chans, overall_psi=0.5)
    lam = current.lyapunov_exponent_estimate(history)
    assert lam == pytest.approx(0.0, abs=1e-9)
