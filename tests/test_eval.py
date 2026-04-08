"""Tests for carl_studio.eval -- config validation, gate logic, report structure.

Does NOT test model loading or GPU inference.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from carl_studio.eval.runner import (
    EvalConfig,
    EvalGate,
    EvalReport,
    _detect_phase,
    _compute_phase1_metrics,
    _compute_phase2_metrics,
    _compute_phase2prime_metrics,
    _PRIMARY_METRIC,
)


# ---------------------------------------------------------------------------
# EvalConfig validation
# ---------------------------------------------------------------------------

class TestEvalConfig:
    def test_defaults(self):
        cfg = EvalConfig(checkpoint="org/model", dataset="org/data")
        assert cfg.dataset_split == "test"
        assert cfg.phase == "auto"
        assert cfg.threshold == 0.5
        assert cfg.max_samples is None
        assert cfg.batch_size == 1
        assert cfg.device == "auto"

    def test_invalid_phase(self):
        with pytest.raises(ValidationError, match="phase"):
            EvalConfig(checkpoint="x", dataset="y", phase="3")

    def test_valid_phases(self):
        for p in ("auto", "1", "2", "2prime"):
            cfg = EvalConfig(checkpoint="x", dataset="y", phase=p)
            assert cfg.phase == p

    def test_threshold_bounds(self):
        EvalConfig(checkpoint="x", dataset="y", threshold=0.0)
        EvalConfig(checkpoint="x", dataset="y", threshold=1.0)
        with pytest.raises(ValidationError):
            EvalConfig(checkpoint="x", dataset="y", threshold=-0.1)
        with pytest.raises(ValidationError):
            EvalConfig(checkpoint="x", dataset="y", threshold=1.1)

    def test_max_samples_must_be_positive(self):
        with pytest.raises(ValidationError):
            EvalConfig(checkpoint="x", dataset="y", max_samples=0)


# ---------------------------------------------------------------------------
# Phase detection
# ---------------------------------------------------------------------------

class TestPhaseDetection:
    def test_phase2_vision(self):
        assert _detect_phase("wheattoast11/OmniCoder-9B-Zero-Phase2") == "2"

    def test_phase2prime(self):
        assert _detect_phase("org/model-Phase2Prime-v1") == "2prime"

    def test_phase1_tool_call(self):
        assert _detect_phase("org/model-Phase1-sft") == "1"

    def test_phase1_grpo(self):
        assert _detect_phase("org/model-grpo-step500") == "1"

    def test_vision_keyword(self):
        assert _detect_phase("org/vision-grounding-v2") == "2"

    def test_env_grpo_keyword(self):
        assert _detect_phase("org/env-grpo-checkpoint") == "2prime"

    def test_unknown_defaults_to_1(self):
        assert _detect_phase("totally-unknown-model-name") == "1"


# ---------------------------------------------------------------------------
# EvalReport
# ---------------------------------------------------------------------------

class TestEvalReport:
    def _make_report(self, primary_value: float = 0.7, threshold: float = 0.5) -> EvalReport:
        return EvalReport(
            checkpoint="org/model",
            phase="1",
            n_samples=100,
            metrics={"chain_completion_rate": primary_value, "format_validity": 0.9},
            primary_metric="chain_completion_rate",
            primary_value=primary_value,
            threshold=threshold,
            passed=primary_value >= threshold,
        )

    def test_pass(self):
        report = self._make_report(primary_value=0.8, threshold=0.5)
        assert report.passed is True

    def test_fail(self):
        report = self._make_report(primary_value=0.3, threshold=0.5)
        assert report.passed is False

    def test_exact_threshold(self):
        report = self._make_report(primary_value=0.5, threshold=0.5)
        assert report.passed is True

    def test_coherence_none_by_default(self):
        report = self._make_report()
        assert report.coherence is None

    def test_coherence_dict(self):
        report = EvalReport(
            checkpoint="x", phase="1", n_samples=10,
            metrics={}, primary_metric="m", primary_value=0.5,
            threshold=0.5, passed=True,
            coherence={"phi_mean": 0.8, "cloud_quality_mean": 0.6, "discontinuity_score": 0.02},
        )
        assert report.coherence["phi_mean"] == 0.8


# ---------------------------------------------------------------------------
# EvalGate
# ---------------------------------------------------------------------------

class TestEvalGate:
    def test_pass(self):
        gate = EvalGate(threshold=0.5)
        report = EvalReport(
            checkpoint="x", phase="1", n_samples=1,
            metrics={}, primary_metric="m", primary_value=0.6,
            threshold=0.5, passed=True,
        )
        assert gate.check(report) is True

    def test_fail(self):
        gate = EvalGate(threshold=0.8)
        report = EvalReport(
            checkpoint="x", phase="1", n_samples=1,
            metrics={}, primary_metric="m", primary_value=0.5,
            threshold=0.8, passed=False,
        )
        assert gate.check(report) is False

    def test_exact_threshold(self):
        gate = EvalGate(threshold=0.5)
        report = EvalReport(
            checkpoint="x", phase="1", n_samples=1,
            metrics={}, primary_metric="m", primary_value=0.5,
            threshold=0.5, passed=True,
        )
        assert gate.check(report) is True

    def test_invalid_threshold(self):
        with pytest.raises(ValueError):
            EvalGate(threshold=-0.1)
        with pytest.raises(ValueError):
            EvalGate(threshold=1.5)

    def test_different_thresholds(self):
        """Gate threshold is independent of report threshold."""
        gate = EvalGate(threshold=0.9)
        report = EvalReport(
            checkpoint="x", phase="1", n_samples=1,
            metrics={}, primary_metric="m", primary_value=0.7,
            threshold=0.5, passed=True,  # report says pass at 0.5
        )
        # But gate at 0.9 says fail
        assert gate.check(report) is False


# ---------------------------------------------------------------------------
# Phase-specific metric computation (no model needed)
# ---------------------------------------------------------------------------

class TestPhase1Metrics:
    def test_perfect_format(self):
        completions = ['{"name": "read_file", "arguments": {"path": "/x"}}']
        samples = [{"expected_tools": ["read_file"], "chain_length": 1}]
        m = _compute_phase1_metrics(completions, samples)
        assert m["format_validity"] == 1.0
        assert m["tool_selection_accuracy"] == 1.0
        assert m["chain_completion_rate"] == 1.0

    def test_no_tool_call(self):
        completions = ["just some text with no json"]
        samples = [{"expected_tools": ["read_file"], "chain_length": 1}]
        m = _compute_phase1_metrics(completions, samples)
        assert m["format_validity"] == 0.0
        assert m["tool_selection_accuracy"] == 0.0
        assert m["chain_completion_rate"] == 0.0


class TestPhase2Metrics:
    def test_perfect_click(self):
        completions = ["(100, 100)"]
        samples = [{"bbox": [50, 50, 150, 150]}]
        m = _compute_phase2_metrics(completions, samples)
        assert m["format_compliance"] == 1.0
        assert m["click_accuracy"] == 1.0
        assert m["mean_precision"] > 0.9

    def test_no_bbox(self):
        completions = ["(100, 100)"]
        samples = [{}]  # no bbox
        m = _compute_phase2_metrics(completions, samples)
        assert m["format_compliance"] == 1.0
        assert m["click_accuracy"] == 0.0


class TestPhase2PrimeMetrics:
    def test_completed(self):
        completions = ['{"name": "execute_code", "arguments": {"code": "print(1)"}}']
        samples = [{"completed": True, "carl_score": 0.8}]
        m = _compute_phase2prime_metrics(completions, samples)
        assert m["task_completion_rate"] == 1.0
        assert m["mean_carl_score"] == 0.8

    def test_failed(self):
        completions = ["no tool call"]
        samples = [{"completed": False, "carl_score": 0.1}]
        m = _compute_phase2prime_metrics(completions, samples)
        assert m["task_completion_rate"] == 0.0
        assert m["mean_carl_score"] == 0.1

    def test_heuristic_fallback(self):
        """When no completed/success field, uses tool-call heuristic."""
        completions = ['{"name": "write_file", "arguments": {"path": "/x", "content": "y"}}']
        samples = [{"carl_score": 0.5}]  # no completed field
        m = _compute_phase2prime_metrics(completions, samples)
        assert m["task_completion_rate"] == 1.0


# ---------------------------------------------------------------------------
# Primary metric mapping
# ---------------------------------------------------------------------------

class TestPrimaryMetric:
    def test_phase1(self):
        assert _PRIMARY_METRIC["1"] == "chain_completion_rate"

    def test_phase2(self):
        assert _PRIMARY_METRIC["2"] == "click_accuracy"

    def test_phase2prime(self):
        assert _PRIMARY_METRIC["2prime"] == "task_completion_rate"
