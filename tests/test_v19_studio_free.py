"""Tests for v0.19 studio FREE surface (forecast + substrate CLI + observe)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from carl_core import CoherenceForecast, SubstrateChannel, SubstrateState
from carl_studio.cli.apps import app as root_app
from carl_studio.cli.forecast import forecast_app
from carl_studio.cli.substrate import substrate_app
from carl_studio.observe.forecast_dashboard import (
    TrinityView,
    linear_forecast,
    lyapunov_forecast,
    make_trinity_view,
)
from carl_studio.observe.substrate_monitor import SubstrateMonitor

# Import wiring so the sub-apps are registered on root_app.
import carl_studio.cli.wiring  # noqa: F401


runner = CliRunner()


# ----------------------------------------------------------------------
# linear_forecast
# ----------------------------------------------------------------------


def test_linear_forecast_extrapolates_increasing_history() -> None:
    history = [0.1, 0.2, 0.3, 0.4]
    fc = linear_forecast(history, horizon=3)
    assert isinstance(fc, CoherenceForecast)
    assert fc.horizon_steps == 3
    # Linear extension: phi at step 0 should be > 0.4 (continuing growth)
    assert fc.phi_at(0) > 0.4
    assert fc.phi_at(1) > fc.phi_at(0)


def test_linear_forecast_decreasing_history_gives_decay() -> None:
    history = [0.9, 0.8, 0.7, 0.6]
    fc = linear_forecast(history, horizon=3)
    assert fc.phi_at(0) < 0.6
    assert fc.lyapunov_drift() < 0.0


def test_linear_forecast_empty_history_returns_flat_default() -> None:
    fc = linear_forecast([], horizon=4)
    assert fc.horizon_steps == 4
    assert fc.phi_at(0) == pytest.approx(0.5)


def test_linear_forecast_single_value_history_returns_flat() -> None:
    fc = linear_forecast([0.7], horizon=4)
    assert fc.phi_at(0) == pytest.approx(0.7)
    assert fc.phi_at(3) == pytest.approx(0.7)


def test_linear_forecast_zero_horizon_raises() -> None:
    with pytest.raises(ValueError):
        linear_forecast([0.5, 0.6], horizon=0)


def test_linear_forecast_clamps_to_unit_interval() -> None:
    # Steeply rising history would extrapolate beyond 1.0; must clamp
    history = [0.6, 0.7, 0.8, 0.9, 0.95]
    fc = linear_forecast(history, horizon=10)
    assert np.all(fc.phi_predicted <= 1.0)
    assert np.all(fc.phi_predicted >= 0.0)


# ----------------------------------------------------------------------
# lyapunov_forecast
# ----------------------------------------------------------------------


def test_lyapunov_forecast_decay_history() -> None:
    history = [0.8, 0.4, 0.2, 0.1]  # exponential decay
    fc = lyapunov_forecast(history, horizon=3)
    assert fc.method == "lyapunov"
    assert fc.lyapunov_drift() < 0.0


def test_lyapunov_forecast_falls_back_when_history_short() -> None:
    fc = lyapunov_forecast([0.5], horizon=4)
    assert fc.horizon_steps == 4
    # falls back to linear which is flat for single-value history
    assert fc.phi_at(0) == pytest.approx(0.5)


# ----------------------------------------------------------------------
# TrinityView + make_trinity_view
# ----------------------------------------------------------------------


def test_make_trinity_view_with_history() -> None:
    history = [0.5, 0.6, 0.7, 0.8]
    tv = make_trinity_view(history, present_phi=0.85, horizon=4)
    assert isinstance(tv, TrinityView)
    assert tv.past_summary["n"] == 4
    assert tv.past_summary["min"] == pytest.approx(0.5)
    assert tv.past_summary["max"] == pytest.approx(0.8)
    assert tv.present_phi == pytest.approx(0.85)
    assert tv.forecast.horizon_steps == 4


def test_make_trinity_view_empty_history() -> None:
    tv = make_trinity_view([], present_phi=0.5, horizon=4)
    assert tv.past_summary["n"] == 0
    assert tv.past_summary["min"] is None


def test_trinity_view_to_dict_round_keys() -> None:
    tv = make_trinity_view([0.5, 0.6], present_phi=0.7, horizon=3)
    d = tv.to_dict()
    assert "past" in d
    assert "present" in d
    assert "future" in d
    assert d["present"]["phi"] == pytest.approx(0.7)


def test_trinity_view_early_warning_fires_on_decay() -> None:
    history = [0.9, 0.8, 0.7, 0.6, 0.5]
    tv = make_trinity_view(history, present_phi=0.5, horizon=10)
    warning = tv.early_warning_step(threshold=0.4)
    assert warning is not None and warning >= 0


# ----------------------------------------------------------------------
# SubstrateMonitor
# ----------------------------------------------------------------------


def test_substrate_monitor_sample_returns_state() -> None:
    monitor = SubstrateMonitor()
    state = monitor.sample()
    assert isinstance(state, SubstrateState)
    assert 0.0 <= state.overall_psi <= 1.0
    assert len(state.channels) >= 1


def test_substrate_monitor_history_grows() -> None:
    monitor = SubstrateMonitor()
    for i in range(3):
        monitor.sample(timestamp=float(i))
    assert len(monitor.history) == 3


def test_substrate_monitor_history_bounded_by_max() -> None:
    monitor = SubstrateMonitor(max_history=2)
    for i in range(5):
        monitor.sample(timestamp=float(i))
    assert len(monitor.history) == 2
    assert monitor.history[-1].timestamp == 4.0


def test_substrate_monitor_drift_computed_after_two_samples() -> None:
    monitor = SubstrateMonitor()
    monitor.sample(timestamp=1.0)
    state2 = monitor.sample(timestamp=2.0)
    # By second sample, drift_rate should be computable (zero or non-zero)
    for ch in state2.channels.values():
        assert isinstance(ch.drift_rate, float)


def test_substrate_monitor_reset_clears_history() -> None:
    monitor = SubstrateMonitor()
    monitor.sample()
    monitor.sample()
    monitor.reset()
    assert monitor.history == []


# ----------------------------------------------------------------------
# CLI: carl forecast show
# ----------------------------------------------------------------------


def test_cli_forecast_show_with_inline_history() -> None:
    result = runner.invoke(
        forecast_app,
        ["show", "--history", "0.5,0.6,0.7,0.8", "--horizon", "4"],
    )
    assert result.exit_code == 0
    assert "PAST" in result.stdout
    assert "PRESENT" in result.stdout
    assert "FUTURE" in result.stdout


def test_cli_forecast_show_json_output() -> None:
    result = runner.invoke(
        forecast_app,
        ["show", "--history", "0.5,0.6,0.7,0.8", "--horizon", "3", "--json"],
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["past"]["n"] == 4
    assert "future" in payload
    assert payload["future"]["horizon_steps"] == 3
    assert "early_warning_step" in payload
    assert payload["threshold"] == 0.5


def test_cli_forecast_show_history_file(tmp_path: Path) -> None:
    history_file = tmp_path / "phi_hist.json"
    history_file.write_text(json.dumps([0.4, 0.45, 0.5, 0.55]))
    result = runner.invoke(
        forecast_app,
        ["show", "--history-file", str(history_file), "--horizon", "2", "--json"],
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["past"]["n"] == 4


def test_cli_forecast_show_no_history_uses_flat_default() -> None:
    result = runner.invoke(forecast_app, ["show", "--horizon", "3", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["past"]["n"] == 0


def test_cli_forecast_show_warning_fires_on_decay() -> None:
    result = runner.invoke(
        forecast_app,
        [
            "show",
            "--history",
            "0.9,0.8,0.7,0.6,0.5,0.4",
            "--horizon",
            "5",
            "--threshold",
            "0.3",
            "--json",
        ],
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    # decaying history with threshold=0.3 should fire eventually
    assert payload["early_warning_step"] is not None or payload["future"]["phi_predicted"][-1] < 0.3


def test_cli_forecast_show_lyapunov_method() -> None:
    result = runner.invoke(
        forecast_app,
        ["show", "--history", "0.8,0.4,0.2,0.1", "--method", "lyapunov", "--json"],
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["future"]["method"] == "lyapunov"


def test_cli_forecast_show_invalid_history_rejected() -> None:
    result = runner.invoke(forecast_app, ["show", "--history", "abc,def"])
    assert result.exit_code != 0


# ----------------------------------------------------------------------
# CLI: carl substrate show
# ----------------------------------------------------------------------


def test_cli_substrate_show_default() -> None:
    result = runner.invoke(substrate_app, ["show"])
    assert result.exit_code == 0
    assert "substrate state" in result.stdout.lower()
    assert "overall_psi" in result.stdout


def test_cli_substrate_show_json() -> None:
    result = runner.invoke(substrate_app, ["show", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert "psutil_available" in payload
    assert "samples" in payload
    assert len(payload["samples"]) == 1


def test_cli_substrate_show_multiple_samples() -> None:
    result = runner.invoke(substrate_app, ["show", "--samples", "3", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert len(payload["samples"]) == 3


# ----------------------------------------------------------------------
# Wiring: sub-apps registered on root
# ----------------------------------------------------------------------


def test_root_app_has_forecast_subcommand() -> None:
    """`carl forecast --help` should resolve via root app routing."""
    result = runner.invoke(root_app, ["forecast", "--help"])
    assert result.exit_code == 0
    assert "show" in result.stdout.lower()


def test_root_app_has_substrate_subcommand() -> None:
    result = runner.invoke(root_app, ["substrate", "--help"])
    assert result.exit_code == 0
    assert "show" in result.stdout.lower()


def test_root_app_forecast_show_executes() -> None:
    result = runner.invoke(
        root_app,
        ["forecast", "show", "--history", "0.5,0.6,0.7", "--horizon", "2", "--json"],
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["future"]["horizon_steps"] == 2


# ----------------------------------------------------------------------
# Substrate channel construction direct (sanity)
# ----------------------------------------------------------------------


def test_direct_substrate_channel_in_studio_roundtrip() -> None:
    """Carl-core types are usable from carl-studio surface."""
    ch = SubstrateChannel(name="test", health=0.9, drift_rate=0.05, drift_acceleration=-0.01)
    assert ch.leading_indicator is True
    state = SubstrateState(timestamp=0.0, channels={"test": ch}, overall_psi=0.9)
    assert "test" in state.critical_channels()
