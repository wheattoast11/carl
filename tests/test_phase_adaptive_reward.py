"""Tests for PhaseAdaptiveCARLReward — SEM-010.

Exercises the phase classifier, weight shifts, fall-back behavior,
and exception resilience. Builds minimal fake trace objects rather
than depending on real model logits.
"""
from __future__ import annotations

import numpy as np
import pytest

from carl_studio.training.rewards.composite import (
    CARLReward,
    PhaseAdaptiveCARLReward,
)
from carl_studio.training.rewards.multiscale import reset_clamp_counts


# ---------------------------------------------------------------------------
# Fixtures + fake trace
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _fresh_state():
    reset_clamp_counts()
    yield
    reset_clamp_counts()


class _FakeTrace:
    """Minimal CoherenceTrace stand-in.

    Implements only what PhaseAdaptiveCARLReward and CARLReward.score_from_trace
    touch: the Kuramoto-R read path plus the carl_reward + derived metric
    properties used by score_from_trace. All scalars are fixed-value so tests
    assert on the class's behavior, not on the underlying math.
    """

    def __init__(
        self,
        R: float,
        multiscale: float = 0.5,
        cloud: float = 0.5,
        discontinuity: float = 0.5,
        raise_on_R: bool = False,
    ) -> None:
        self._R = R
        self._raise_on_R = raise_on_R
        self.multiscale_coherence = multiscale
        self.cloud_quality = cloud
        self.discontinuity_score = discontinuity

    def kuramoto_R(self) -> float:
        if self._raise_on_R:
            raise RuntimeError("forced kuramoto_R failure")
        return self._R

    def carl_reward(
        self,
        w_coherence: float = 0.5,
        w_cloud: float = 0.3,
        w_discontinuity: float = 0.2,
    ) -> float:
        return (
            w_coherence * self.multiscale_coherence
            + w_cloud * self.cloud_quality
            + w_discontinuity * self.discontinuity_score
        )


# ---------------------------------------------------------------------------
# Default / fall-back behavior
# ---------------------------------------------------------------------------


class TestFallback:
    def test_par_defaults_to_static_when_no_traces(self):
        par = PhaseAdaptiveCARLReward()
        # No score_*() has been called yet -- cache is empty.
        assert par._last_traces == []
        # current_weights should equal the static CARLReward defaults.
        assert par.current_weights == (0.5, 0.3, 0.2)
        assert par.weight_multiscale == 0.5
        assert par.weight_cloud == 0.3
        assert par.weight_discontinuity == 0.2
        # Calling _apply_phase_weights with no traces must not change them.
        par._apply_phase_weights()
        assert par.current_weights == (0.5, 0.3, 0.2)

    def test_par_retains_custom_default_weights(self):
        par = PhaseAdaptiveCARLReward(
            weight_multiscale=0.7,
            weight_cloud=0.2,
            weight_discontinuity=0.1,
        )
        assert par.current_weights == (0.7, 0.2, 0.1)
        # Fallback path must not clobber user-supplied defaults.
        par._apply_phase_weights()
        assert par.current_weights == (0.7, 0.2, 0.1)

    def test_par_is_carlreward_subclass(self):
        par = PhaseAdaptiveCARLReward()
        assert isinstance(par, CARLReward)


# ---------------------------------------------------------------------------
# Phase classification from Kuramoto-R
# ---------------------------------------------------------------------------


class TestPhaseClassification:
    def test_par_classifies_phase_from_R(self):
        """Direct unit test of _phase_weights_from_R across the full R spectrum."""
        par = PhaseAdaptiveCARLReward()
        # Boundary cases + representative midpoints.
        assert par._phase_weights_from_R(0.00) == PhaseAdaptiveCARLReward._PROFILE_GASEOUS
        assert par._phase_weights_from_R(0.10) == PhaseAdaptiveCARLReward._PROFILE_GASEOUS
        assert par._phase_weights_from_R(0.29) == PhaseAdaptiveCARLReward._PROFILE_GASEOUS
        # 0.30 is the gaseous-max threshold; inclusive of liquid.
        assert par._phase_weights_from_R(0.30) == PhaseAdaptiveCARLReward._PROFILE_LIQUID
        assert par._phase_weights_from_R(0.50) == PhaseAdaptiveCARLReward._PROFILE_LIQUID
        assert par._phase_weights_from_R(0.69) == PhaseAdaptiveCARLReward._PROFILE_LIQUID
        # 0.70 is the liquid-max threshold; inclusive of crystalline.
        assert par._phase_weights_from_R(0.70) == PhaseAdaptiveCARLReward._PROFILE_CRYSTALLINE
        assert par._phase_weights_from_R(0.85) == PhaseAdaptiveCARLReward._PROFILE_CRYSTALLINE
        assert par._phase_weights_from_R(1.00) == PhaseAdaptiveCARLReward._PROFILE_CRYSTALLINE

    def test_par_gaseous_phase(self):
        """R=0.1 -> gaseous profile (discontinuity dominates)."""
        par = PhaseAdaptiveCARLReward()
        par._last_traces = [_FakeTrace(R=0.1)]
        par._apply_phase_weights()
        assert par.current_weights == (0.20, 0.30, 0.50)
        assert par.weight_multiscale == 0.20
        assert par.weight_cloud == 0.30
        assert par.weight_discontinuity == 0.50
        assert par.current_phase == "gaseous"

    def test_par_liquid_phase(self):
        """R=0.5 -> liquid profile (balanced)."""
        par = PhaseAdaptiveCARLReward()
        par._last_traces = [_FakeTrace(R=0.5)]
        par._apply_phase_weights()
        assert par.current_weights == (0.40, 0.30, 0.30)
        assert par.weight_multiscale == 0.40
        assert par.weight_cloud == 0.30
        assert par.weight_discontinuity == 0.30
        assert par.current_phase == "liquid"

    def test_par_crystalline_phase(self):
        """R=0.85 -> crystalline profile (multiscale dominates)."""
        par = PhaseAdaptiveCARLReward()
        par._last_traces = [_FakeTrace(R=0.85)]
        par._apply_phase_weights()
        assert par.current_weights == (0.60, 0.30, 0.10)
        assert par.weight_multiscale == 0.60
        assert par.weight_cloud == 0.30
        assert par.weight_discontinuity == 0.10
        assert par.current_phase == "crystalline"

    def test_par_all_profiles_sum_to_one(self):
        """Invariant: every phase profile's weights sum to exactly 1.0."""
        for profile in (
            PhaseAdaptiveCARLReward._PROFILE_GASEOUS,
            PhaseAdaptiveCARLReward._PROFILE_LIQUID,
            PhaseAdaptiveCARLReward._PROFILE_CRYSTALLINE,
        ):
            assert sum(profile) == pytest.approx(1.0)

    def test_par_current_R_averages_across_batch(self):
        """Multi-trace cache should produce mean R."""
        par = PhaseAdaptiveCARLReward()
        par._last_traces = [_FakeTrace(R=0.1), _FakeTrace(R=0.9)]
        assert par._current_R() == pytest.approx(0.5)
        par._apply_phase_weights()
        # Mean R=0.5 -> liquid
        assert par.current_phase == "liquid"


# ---------------------------------------------------------------------------
# current_phase property
# ---------------------------------------------------------------------------


class TestCurrentPhaseProperty:
    def test_par_current_phase_property(self):
        par = PhaseAdaptiveCARLReward()
        # No traces -> "unknown"
        assert par.current_phase == "unknown"

        par._last_traces = [_FakeTrace(R=0.05)]
        assert par.current_phase == "gaseous"

        par._last_traces = [_FakeTrace(R=0.45)]
        assert par.current_phase == "liquid"

        par._last_traces = [_FakeTrace(R=0.95)]
        assert par.current_phase == "crystalline"

    def test_par_current_phase_matches_applied_weights(self):
        """After _apply_phase_weights, current_phase should match the chosen profile."""
        par = PhaseAdaptiveCARLReward()
        for R, expected_phase, expected_profile in (
            (0.15, "gaseous", PhaseAdaptiveCARLReward._PROFILE_GASEOUS),
            (0.55, "liquid", PhaseAdaptiveCARLReward._PROFILE_LIQUID),
            (0.80, "crystalline", PhaseAdaptiveCARLReward._PROFILE_CRYSTALLINE),
        ):
            par._last_traces = [_FakeTrace(R=R)]
            par._apply_phase_weights()
            assert par.current_phase == expected_phase
            assert par.current_weights == expected_profile


# ---------------------------------------------------------------------------
# Exception resilience
# ---------------------------------------------------------------------------


class TestExceptionResilience:
    def test_par_handles_trace_exception_gracefully(self):
        """If kuramoto_R() raises, weights must not change and no crash propagates."""
        par = PhaseAdaptiveCARLReward(
            weight_multiscale=0.5,
            weight_cloud=0.3,
            weight_discontinuity=0.2,
        )
        par._last_traces = [_FakeTrace(R=0.1, raise_on_R=True)]
        # _current_R must return None, not raise.
        assert par._current_R() is None
        # _apply_phase_weights must be a no-op.
        par._apply_phase_weights()
        assert par.current_weights == (0.5, 0.3, 0.2)
        # current_phase must also surface "unknown".
        assert par.current_phase == "unknown"

    def test_par_handles_none_entries_in_cache(self):
        """None entries in _last_traces must be skipped, not crash."""
        par = PhaseAdaptiveCARLReward()
        par._last_traces = [None, _FakeTrace(R=0.8), None]  # type: ignore[list-item]
        # Should average over the single valid trace.
        assert par._current_R() == pytest.approx(0.8)
        par._apply_phase_weights()
        assert par.current_weights == PhaseAdaptiveCARLReward._PROFILE_CRYSTALLINE

    def test_par_handles_all_none_entries(self):
        """If all cache entries are None, fall back to static weights."""
        par = PhaseAdaptiveCARLReward()
        par._last_traces = [None, None]  # type: ignore[list-item]
        assert par._current_R() is None
        par._apply_phase_weights()
        assert par.current_weights == (0.5, 0.3, 0.2)

    def test_par_handles_non_finite_R(self):
        """NaN / inf Kuramoto-R values must be filtered out."""
        par = PhaseAdaptiveCARLReward()
        par._last_traces = [
            _FakeTrace(R=float("nan")),
            _FakeTrace(R=float("inf")),
            _FakeTrace(R=0.85),
        ]
        # Only the finite value survives.
        assert par._current_R() == pytest.approx(0.85)


# ---------------------------------------------------------------------------
# score_from_trace integration: weights shift BEFORE scoring, cache AFTER
# ---------------------------------------------------------------------------


class TestScoreFromTraceIntegration:
    def test_score_from_trace_caches_current_trace(self):
        par = PhaseAdaptiveCARLReward()
        trace = _FakeTrace(R=0.85, multiscale=0.7, cloud=0.6, discontinuity=0.2)
        par.score_from_trace(trace)  # type: ignore[arg-type]
        # Cache now holds this trace.
        assert par._last_traces == [trace]
        assert par.current_phase == "crystalline"

    def test_score_from_trace_shifts_weights_on_subsequent_call(self):
        """First call uses static weights (no cache); second call uses phase weights."""
        par = PhaseAdaptiveCARLReward()

        # First trace: R=0.1. At score time, cache is empty -> static weights (0.5, 0.3, 0.2).
        trace_gas = _FakeTrace(
            R=0.1, multiscale=0.8, cloud=0.5, discontinuity=0.9
        )
        par.score_from_trace(trace_gas)  # type: ignore[arg-type]
        assert par.current_weights == (0.5, 0.3, 0.2)  # unchanged

        # Second call: cache now holds trace_gas (R=0.1); weights shift to gaseous.
        trace_two = _FakeTrace(
            R=0.1, multiscale=0.8, cloud=0.5, discontinuity=0.9
        )
        par.score_from_trace(trace_two)  # type: ignore[arg-type]
        assert par.current_weights == PhaseAdaptiveCARLReward._PROFILE_GASEOUS

    def test_compute_batch_scoring(self):
        """compute() scores a batch and caches all traces."""
        par = PhaseAdaptiveCARLReward()
        traces = [
            _FakeTrace(R=0.85, multiscale=1.0, cloud=1.0, discontinuity=1.0),
            _FakeTrace(R=0.85, multiscale=0.5, cloud=0.5, discontinuity=0.5),
        ]
        scores = par.compute(traces)  # type: ignore[arg-type]
        assert len(scores) == 2
        assert all(isinstance(s, float) for s in scores)
        # All traces cached.
        assert par._last_traces == traces
        # Next compute() will see R=0.85 -> crystalline.
        assert par.current_phase == "crystalline"

    def test_compute_is_empty_safe(self):
        par = PhaseAdaptiveCARLReward()
        assert par.compute([]) == []
        assert par._last_traces == []


# ---------------------------------------------------------------------------
# score() integration (logits path)
# ---------------------------------------------------------------------------


class TestScoreLogitsIntegration:
    def test_score_logits_caches_trace(self):
        """score() builds a CoherenceTrace and caches it."""
        par = PhaseAdaptiveCARLReward()
        T, V = 8, 16
        logits = np.random.default_rng(42).standard_normal((T, V))
        token_ids = np.random.default_rng(43).integers(0, V, size=T)
        score_val, components = par.score(logits, token_ids)
        assert isinstance(score_val, float)
        assert "multiscale" in components
        assert "cloud_quality" in components
        assert "discontinuity" in components
        # Exactly one trace cached.
        assert len(par._last_traces) == 1


# ---------------------------------------------------------------------------
# Exported from package
# ---------------------------------------------------------------------------


def test_exported_from_rewards_package():
    """PhaseAdaptiveCARLReward is re-exported from the rewards package root."""
    from carl_studio.training.rewards import PhaseAdaptiveCARLReward as Exported

    assert Exported is PhaseAdaptiveCARLReward
    import carl_studio.training.rewards as rewards_mod

    assert "PhaseAdaptiveCARLReward" in rewards_mod.__all__
