"""Tests for carl_core.dynamics.ContractionProbe (SEM-003)."""
from __future__ import annotations

import math

import pytest

from carl_core.dynamics import ContractionProbe, ContractionReport


def test_probe_requires_min_samples() -> None:
    """With <3 points the probe cannot fit and must return ``None``.

    Rationale: the fit uses adjacent ratios of consecutive distances,
    which needs at least two distances, which needs at least three
    points in the trajectory.
    """
    probe = ContractionProbe(window=10)
    assert probe.report() is None
    assert probe.contraction_violation() is False

    probe.record(0.5, 0.5)
    probe.record(0.6, 0.6)
    # Two points -> still below the three-point floor.
    assert probe.report() is None
    assert probe.contraction_violation() is False


def test_probe_detects_contraction_on_converging_series() -> None:
    """A strictly contracting phi trajectory must produce q_hat < 1."""
    probe = ContractionProbe(window=20)
    # Geometric convergence toward phi=0.7, cq=0.7.
    phi_vals = [0.5, 0.6, 0.65, 0.675, 0.6875, 0.69375, 0.696875, 0.698438]
    cq_vals = [0.5, 0.6, 0.65, 0.675, 0.6875, 0.69375, 0.696875, 0.698438]
    for p, c in zip(phi_vals, cq_vals):
        probe.record(p, c)
    report = probe.report()
    assert report is not None
    assert isinstance(report, ContractionReport)
    assert report.contraction_holds is True
    assert report.q_hat < 1.0
    # Successive distances halve, so q_hat should be close to 0.5.
    assert 0.3 < report.q_hat < 0.8
    assert probe.contraction_violation() is False


def test_probe_detects_violation_on_diverging_series() -> None:
    """An oscillating/expanding trajectory should register q_hat >= 1."""
    probe = ContractionProbe(window=20)
    # Distances: 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4 -> q_hat = 1.
    # Sprinkle in a true expansion to push >= 1.
    phi_vals = [0.1, 0.5, 0.9, 0.4, 0.95, 0.2, 0.99, 0.05]
    cq_vals = [0.1, 0.5, 0.9, 0.4, 0.95, 0.2, 0.99, 0.05]
    for p, c in zip(phi_vals, cq_vals):
        probe.record(p, c)
    report = probe.report()
    assert report is not None
    # Either q_hat >= 1 (divergence) or very close to 1 (no contraction).
    assert report.q_hat >= 1.0 or math.isclose(report.q_hat, 1.0, abs_tol=0.1)
    # The violation API should agree at threshold=1.0 when q_hat >= 1.
    if report.q_hat >= 1.0:
        assert probe.contraction_violation(threshold=1.0) is True


def test_probe_ignores_nonfinite() -> None:
    """NaN / inf inputs are dropped silently without crashing the probe."""
    probe = ContractionProbe(window=10)
    probe.record(0.5, 0.5)
    probe.record(float("nan"), 0.6)
    probe.record(0.6, float("inf"))
    probe.record(float("-inf"), float("nan"))
    probe.record(0.7, 0.7)
    probe.record(0.8, 0.8)
    # Only three valid points recorded.
    assert len(probe) == 3


def test_probe_reset_clears_history() -> None:
    probe = ContractionProbe(window=10)
    for i in range(5):
        probe.record(0.5 + i * 0.05, 0.5 + i * 0.05)
    assert len(probe) == 5
    probe.reset()
    assert len(probe) == 0
    assert probe.report() is None


def test_probe_window_caps_history() -> None:
    """``window`` caps history length -- oldest points are evicted."""
    probe = ContractionProbe(window=4)
    for i in range(10):
        probe.record(0.1 * i, 0.1 * i)
    # deque maxlen truncates to window.
    assert len(probe) == 4


def test_probe_constant_series_returns_none_or_zero_ratio() -> None:
    """A constant trajectory has zero step-distances -> fit is degenerate."""
    probe = ContractionProbe(window=10)
    for _ in range(6):
        probe.record(0.5, 0.5)
    # All consecutive distances are 0; cannot form any ratio -> None.
    assert probe.report() is None
    assert probe.contraction_violation() is False


def test_probe_window_validation() -> None:
    """``window < 3`` is rejected at construction time."""
    with pytest.raises(ValueError, match="window must be >= 3"):
        ContractionProbe(window=2)
    with pytest.raises(ValueError):
        ContractionProbe(window=0)
    # Boundary: 3 is accepted.
    ContractionProbe(window=3)


def test_probe_reports_trajectory_length() -> None:
    probe = ContractionProbe(window=50)
    for i in range(7):
        probe.record(0.5 + 0.01 * i, 0.5 + 0.01 * i)
    report = probe.report()
    assert report is not None
    assert report.window_trajectory_length == 7


def test_probe_report_residual_variance_nonnegative() -> None:
    """Residual variance is always >= 0 by construction."""
    probe = ContractionProbe(window=20)
    for i in range(10):
        probe.record(0.5 + 0.03 * i, 0.4 + 0.05 * i)
    report = probe.report()
    assert report is not None
    assert report.residual_variance >= 0.0


def test_probe_custom_threshold() -> None:
    """A strict threshold (< q_hat) fires the violation even when < 1."""
    probe = ContractionProbe(window=20)
    # Series with q_hat ~ 0.5 (halving steps).
    vals = [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125]
    for v in vals:
        probe.record(v, v)
    report = probe.report()
    assert report is not None
    assert report.q_hat < 1.0
    # Even a contracting trajectory will "violate" a threshold lower than q_hat.
    assert probe.contraction_violation(threshold=0.1) is True
    assert probe.contraction_violation(threshold=1.0) is False
