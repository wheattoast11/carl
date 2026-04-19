"""SEM-001: EvalGate must require coherence AND primary metric.

Today's CARL package is named for coherence but the legacy eval gate only
checks a correctness metric. These tests enforce the reward/eval
isomorphism: the gate now reads the same Phi/discontinuity field that
the training rewards optimize for, and it writes back a human-readable
``gate_reason`` explaining which dimension failed.
"""

from __future__ import annotations

import pytest

from carl_core.constants import SIGMA
from carl_studio.eval.runner import EvalConfig, EvalGate, EvalReport


def _make_config(
    *,
    threshold: float = 0.5,
    phi_floor: float | None = None,
    disc_min: float = 0.3,
    disc_max: float = 0.7,
    require: bool = True,
) -> EvalConfig:
    kwargs: dict[str, float | bool] = {
        "checkpoint": "org/model",
        "dataset": "org/data",
        "phase": "1",
        "threshold": threshold,
        "discontinuity_min": disc_min,
        "discontinuity_max": disc_max,
        "require_coherence_gate": require,
    }
    if phi_floor is not None:
        kwargs["coherence_phi_floor"] = phi_floor
    return EvalConfig(**kwargs)  # type: ignore[arg-type]


def _make_report(
    *,
    primary_value: float = 0.7,
    phi_mean: float | None = 0.3,
    disc: float | None = 0.5,
    cloud: float | None = 0.5,
    threshold: float = 0.5,
) -> EvalReport:
    coherence: dict[str, float] | None
    if phi_mean is None and disc is None and cloud is None:
        coherence = None
    else:
        coherence = {}
        if phi_mean is not None:
            coherence["phi_mean"] = phi_mean
        if disc is not None:
            coherence["discontinuity_score"] = disc
        if cloud is not None:
            coherence["cloud_quality_mean"] = cloud
    return EvalReport(
        checkpoint="org/model",
        phase="1",
        n_samples=20,
        metrics={"chain_completion_rate": primary_value},
        primary_metric="chain_completion_rate",
        primary_value=primary_value,
        threshold=threshold,
        passed=False,  # recomputed by gate.check
        coherence=coherence,
    )


def test_gate_passes_with_primary_and_coherence():
    """Pass when primary metric >= threshold AND coherence is in band."""
    cfg = _make_config(threshold=0.5, phi_floor=SIGMA, disc_min=0.3, disc_max=0.7)
    gate = EvalGate(threshold=cfg.threshold, phase="1", config=cfg)
    report = _make_report(primary_value=0.75, phi_mean=0.40, disc=0.5)
    assert gate.check(report) is True
    assert report.gate_reason is not None
    assert report.gate_reason.startswith("PASS")


def test_gate_fails_when_coherence_below_floor():
    """Primary metric alone is no longer enough — low phi must fail."""
    cfg = _make_config(threshold=0.5, phi_floor=SIGMA)
    gate = EvalGate(threshold=cfg.threshold, phase="1", config=cfg)
    report = _make_report(
        primary_value=0.90,  # correctness would pass legacy gate
        phi_mean=0.10,        # below SIGMA=3/16=0.1875 — incoherent output
        disc=0.5,
    )
    assert gate.check(report) is False
    assert report.gate_reason is not None
    assert "FAIL" in report.gate_reason
    assert "coherence=False" in report.gate_reason


def test_gate_fails_when_discontinuity_below_min():
    """Frozen model: no phase transitions happening — not learning."""
    cfg = _make_config(disc_min=0.3, disc_max=0.7)
    gate = EvalGate(threshold=cfg.threshold, phase="1", config=cfg)
    report = _make_report(primary_value=0.75, phi_mean=0.40, disc=0.20)
    assert gate.check(report) is False
    assert report.gate_reason is not None
    assert "disc=0.200" in report.gate_reason
    assert "FAIL" in report.gate_reason


def test_gate_fails_when_discontinuity_above_max():
    """Thrashing model: too many phase transitions — unstable outputs."""
    cfg = _make_config(disc_min=0.3, disc_max=0.7)
    gate = EvalGate(threshold=cfg.threshold, phase="1", config=cfg)
    report = _make_report(primary_value=0.75, phi_mean=0.40, disc=0.85)
    assert gate.check(report) is False
    assert report.gate_reason is not None
    assert "disc=0.850" in report.gate_reason


def test_gate_opt_out_skips_coherence_check():
    """require_coherence_gate=False restores legacy correctness-only gate."""
    cfg = _make_config(require=False)
    gate = EvalGate(threshold=cfg.threshold, phase="1", config=cfg)
    # Phi far below the floor — but the gate is opted out, so primary wins.
    report = _make_report(primary_value=0.75, phi_mean=0.01, disc=0.5)
    assert gate.check(report) is True
    assert report.gate_reason == "PASS"


def test_gate_reason_logs_which_dimension_failed():
    """The reason string names every failing dimension explicitly."""
    cfg = _make_config(
        threshold=0.5, phi_floor=SIGMA, disc_min=0.3, disc_max=0.7
    )
    gate = EvalGate(threshold=cfg.threshold, phase="1", config=cfg)

    # Primary fails, coherence passes.
    report_primary_fail = _make_report(
        primary_value=0.20, phi_mean=0.40, disc=0.5
    )
    assert gate.check(report_primary_fail) is False
    assert report_primary_fail.gate_reason is not None
    assert "primary=False" in report_primary_fail.gate_reason
    assert "coherence=True" in report_primary_fail.gate_reason

    # Primary passes, coherence fails on phi.
    report_phi_fail = _make_report(primary_value=0.80, phi_mean=0.05, disc=0.5)
    assert gate.check(report_phi_fail) is False
    assert report_phi_fail.gate_reason is not None
    assert "primary=True" in report_phi_fail.gate_reason
    assert "coherence=False" in report_phi_fail.gate_reason

    # Both fail — both dimensions recorded.
    report_both = _make_report(primary_value=0.10, phi_mean=0.01, disc=0.95)
    assert gate.check(report_both) is False
    assert report_both.gate_reason is not None
    assert "primary=False" in report_both.gate_reason
    assert "coherence=False" in report_both.gate_reason


def test_gate_fails_closed_when_coherence_is_none():
    """Missing coherence field is not a free pass when the gate is required."""
    cfg = _make_config()
    gate = EvalGate(threshold=cfg.threshold, phase="1", config=cfg)
    report = _make_report(
        primary_value=0.90, phi_mean=None, disc=None, cloud=None
    )
    assert report.coherence is None
    assert gate.check(report) is False
    assert report.gate_reason is not None
    assert "MISSING" in report.gate_reason


def test_gate_legacy_construction_without_config_stays_primary_only():
    """Callers that still construct EvalGate without a config keep old behavior."""
    gate = EvalGate(threshold=0.5, phase="1")
    report = _make_report(primary_value=0.80, phi_mean=0.01, disc=0.9)
    # No config -> gate is legacy -> only primary metric matters.
    assert gate.check(report) is True
    assert report.gate_reason == "PASS"


def test_config_rejects_inverted_discontinuity_band():
    """disc_min > disc_max is a configuration error — reject at validation."""
    with pytest.raises(Exception):
        EvalConfig(
            checkpoint="x",
            dataset="y",
            discontinuity_min=0.8,
            discontinuity_max=0.2,
        )


def test_config_defaults_match_spec():
    """Default phi_floor = SIGMA; default band = [0.3, 0.7]; gate required."""
    cfg = EvalConfig(checkpoint="x", dataset="y")
    assert cfg.coherence_phi_floor == pytest.approx(SIGMA)
    assert cfg.discontinuity_min == pytest.approx(0.3)
    assert cfg.discontinuity_max == pytest.approx(0.7)
    assert cfg.require_coherence_gate is True
