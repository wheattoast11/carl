"""Tests for carl_studio.bench -- ProbeResult grading, CTI scoring, BenchConfig validation."""

import pytest
from carl_studio.bench.probes import (
    GRADE_MAP,
    ProbeResult,
    CoherenceUnderPressureProbe,
    _grade_to_score,
)
from carl_studio.bench.suite import (
    BenchConfig,
    BenchReport,
    BenchSuite,
    CTI,
    _WEIGHTS,
    _null_probe,
    compute_verdict,
)


# ---------------------------------------------------------------------------
# ProbeResult
# ---------------------------------------------------------------------------


class TestProbeResult:
    def test_construction(self):
        r = ProbeResult(
            probe_name="test", grade="A", score=1.0, value=10.0, detail="ok"
        )
        assert r.grade == "A"
        assert r.score == 1.0

    def test_all_grades_mapped(self):
        for grade, score in GRADE_MAP.items():
            assert _grade_to_score(grade) == score

    def test_unknown_grade_returns_zero(self):
        assert _grade_to_score("Z") == 0.0


# ---------------------------------------------------------------------------
# BenchConfig
# ---------------------------------------------------------------------------


class TestBenchConfig:
    def test_valid_config(self):
        c = BenchConfig(model="test/model")
        assert c.suite == "all"
        assert c.device == "auto"
        assert c.compare_model is None

    def test_all_suite_values(self):
        for s in ("all", "transition", "stability", "pressure", "adaptation"):
            c = BenchConfig(model="m", suite=s)
            assert c.suite == s

    def test_invalid_suite(self):
        with pytest.raises(ValueError, match="suite must be one of"):
            BenchConfig(model="m", suite="invalid")

    def test_empty_model_rejected(self):
        with pytest.raises(ValueError, match="non-empty"):
            BenchConfig(model="")

    def test_whitespace_model_rejected(self):
        with pytest.raises(ValueError, match="non-empty"):
            BenchConfig(model="   ")

    def test_model_stripped(self):
        c = BenchConfig(model="  test/model  ")
        assert c.model == "test/model"

    def test_compare_model_accepted(self):
        c = BenchConfig(model="a", compare_model="b")
        assert c.compare_model == "b"


# ---------------------------------------------------------------------------
# CTI scoring
# ---------------------------------------------------------------------------


class TestCTI:
    @staticmethod
    def _make_result(name: str, grade: str) -> ProbeResult:
        return ProbeResult(
            probe_name=name,
            grade=grade,
            score=_grade_to_score(grade),
            value=0.0,
            detail="test",
        )

    def test_all_A_is_excellent(self):
        cti = CTI(
            transition=self._make_result("phase_transition", "A"),
            stability=self._make_result("phi_stability", "A"),
            pressure=self._make_result("coherence_pressure", "A"),
            adaptation=self._make_result("adaptation_rate", "A"),
            score=1.0,
            verdict="excellent",
        )
        assert cti.score == 1.0
        assert cti.verdict == "excellent"

    def test_all_F_is_poor(self):
        cti = CTI(
            transition=self._make_result("phase_transition", "F"),
            stability=self._make_result("phi_stability", "F"),
            pressure=self._make_result("coherence_pressure", "F"),
            adaptation=self._make_result("adaptation_rate", "F"),
            score=0.0,
            verdict="poor",
        )
        assert cti.score == 0.0
        assert cti.verdict == "poor"

    def test_weighted_score_calculation(self):
        # A=1.0, B=0.75, C=0.5, D=0.25
        # 0.30*1.0 + 0.25*0.75 + 0.25*0.5 + 0.20*0.25 = 0.30 + 0.1875 + 0.125 + 0.05 = 0.6625
        expected = 0.30 * 1.0 + 0.25 * 0.75 + 0.25 * 0.5 + 0.20 * 0.25
        cti = CTI(
            transition=self._make_result("phase_transition", "A"),
            stability=self._make_result("phi_stability", "B"),
            pressure=self._make_result("coherence_pressure", "C"),
            adaptation=self._make_result("adaptation_rate", "D"),
            score=round(expected, 4),
            verdict=compute_verdict(expected),
        )
        assert abs(cti.score - expected) < 1e-4
        assert cti.verdict == "good"

    def test_summary_contains_all_probes(self):
        cti = CTI(
            transition=self._make_result("phase_transition", "A"),
            stability=self._make_result("phi_stability", "B"),
            pressure=self._make_result("coherence_pressure", "C"),
            adaptation=self._make_result("adaptation_rate", "D"),
            score=0.66,
            verdict="good",
        )
        s = cti.summary()
        assert "phase_transition=A" in s
        assert "phi_stability=B" in s
        assert "coherence_pressure=C" in s
        assert "adaptation_rate=D" in s
        assert "good" in s


# ---------------------------------------------------------------------------
# Verdict boundaries
# ---------------------------------------------------------------------------


class TestVerdict:
    def test_excellent(self):
        assert compute_verdict(0.85) == "excellent"
        assert compute_verdict(1.0) == "excellent"

    def test_good(self):
        assert compute_verdict(0.65) == "good"
        assert compute_verdict(0.84) == "good"

    def test_marginal(self):
        assert compute_verdict(0.45) == "marginal"
        assert compute_verdict(0.64) == "marginal"

    def test_poor(self):
        assert compute_verdict(0.0) == "poor"
        assert compute_verdict(0.44) == "poor"


# ---------------------------------------------------------------------------
# Weights sum to 1
# ---------------------------------------------------------------------------


class TestWeights:
    def test_weights_sum_to_one(self):
        assert abs(sum(_WEIGHTS.values()) - 1.0) < 1e-10

    def test_all_probes_have_weight(self):
        for key in ("transition", "stability", "pressure", "adaptation"):
            assert key in _WEIGHTS


# ---------------------------------------------------------------------------
# Null probe
# ---------------------------------------------------------------------------


class TestNullProbe:
    def test_null_probe_returns_F(self):
        r = _null_probe("test_probe")
        assert r.grade == "F"
        assert r.score == 0.0
        assert "skipped" in r.detail.lower()


# ---------------------------------------------------------------------------
# Pressure profile classifier (CPU-only, no model needed)
# ---------------------------------------------------------------------------


class TestPressureProfile:
    def test_plateau(self):
        profile, _ = CoherenceUnderPressureProbe._classify_profile(
            [0.8, 0.79, 0.81, 0.78, 0.80]
        )
        assert profile == "plateau"

    def test_collapse(self):
        profile, _ = CoherenceUnderPressureProbe._classify_profile(
            [0.8, 0.6, 0.3, 0.1, 0.05]
        )
        assert profile == "collapse"

    def test_cliff(self):
        profile, _ = CoherenceUnderPressureProbe._classify_profile(
            [0.9, 0.88, 0.85, 0.4, 0.45]
        )
        assert profile in ("cliff", "chaotic")

    def test_empty(self):
        profile, val = CoherenceUnderPressureProbe._classify_profile([])
        assert profile == "collapse"
        assert val == 1.0

    def test_all_zero(self):
        profile, _ = CoherenceUnderPressureProbe._classify_profile([0, 0, 0, 0, 0])
        assert profile == "collapse"


# ---------------------------------------------------------------------------
# BenchReport
# ---------------------------------------------------------------------------


class TestBenchReport:
    @staticmethod
    def _make_cti(score: float, verdict: str) -> CTI:
        r = ProbeResult(
            probe_name="t", grade="B", score=0.75, value=0.0, detail="test"
        )
        return CTI(
            transition=r,
            stability=r,
            pressure=r,
            adaptation=r,
            score=score,
            verdict=verdict,
        )

    def test_report_without_comparison(self):
        report = BenchReport(primary=self._make_cti(0.8, "good"))
        assert report.comparison is None
        assert report.delta is None
        assert "Primary" in report.summary()

    def test_report_with_comparison(self):
        report = BenchReport(
            primary=self._make_cti(0.8, "good"),
            comparison=self._make_cti(0.5, "marginal"),
            delta=0.3,
        )
        assert report.delta == 0.3
        s = report.summary()
        assert "Comparison" in s
        assert "Delta" in s


# ---------------------------------------------------------------------------
# BenchSuite graceful degradation (no model load)
# ---------------------------------------------------------------------------


class TestBenchSuiteGraceful:
    def test_run_returns_cti_on_model_failure(self):
        config = BenchConfig(model="nonexistent/model-that-does-not-exist")
        suite = BenchSuite(config)
        cti = suite.run()
        assert isinstance(cti, CTI)
        assert cti.verdict == "poor"
        assert cti.score == 0.0
        assert "failed" in cti.transition.detail.lower() or "failed" in cti.transition.detail

    def test_probes_without_model_return_F(self):
        from carl_studio.bench.probes import (
            AdaptationRateProbe,
            PhaseTransitionProbe,
            PhiStabilityProbe,
        )

        for ProbeClass in (
            PhaseTransitionProbe,
            PhiStabilityProbe,
            CoherenceUnderPressureProbe,
            AdaptationRateProbe,
        ):
            r = ProbeClass().run(model=None, tokenizer=None)
            assert r.grade == "F"
            assert r.score == 0.0
