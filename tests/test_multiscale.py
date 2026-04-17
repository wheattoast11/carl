"""Tests for multiscale coherence reward — WS-T3 (clamping) + WS-T4 (shape guards)."""
from __future__ import annotations

import logging

import numpy as np
import pytest

from carl_core.errors import ValidationError

from carl_studio.training.rewards.multiscale import (
    REWARD_CLAMP_MAX,
    REWARD_CLAMP_MIN,
    _clamp_reward,
    _normalize_logits_2d,
    _reset_warn_cache,
    clamp_counts,
    multiscale_coherence_reward,
    reset_clamp_counts,
)


@pytest.fixture(autouse=True)
def _fresh_state():
    _reset_warn_cache()
    reset_clamp_counts()
    yield
    _reset_warn_cache()
    reset_clamp_counts()


# ---------------------------------------------------------------------------
# _clamp_reward — WS-T3 core helper
# ---------------------------------------------------------------------------


class TestClampReward:
    def test_finite_passthrough(self):
        assert _clamp_reward(0.5) == 0.5
        assert _clamp_reward(0.0) == 0.0
        assert _clamp_reward(-0.5) == -0.5

    def test_nan_becomes_zero(self):
        assert _clamp_reward(float("nan")) == 0.0

    def test_positive_inf_becomes_zero(self):
        assert _clamp_reward(float("inf")) == 0.0

    def test_negative_inf_becomes_zero(self):
        assert _clamp_reward(float("-inf")) == 0.0

    def test_above_cap_clips(self):
        assert _clamp_reward(1e9) == REWARD_CLAMP_MAX
        assert _clamp_reward(101.0) == 100.0

    def test_below_floor_clips(self):
        assert _clamp_reward(-1e9) == REWARD_CLAMP_MIN
        assert _clamp_reward(-101.0) == -100.0

    def test_counter_increments_nonfinite(self):
        _clamp_reward(float("nan"))
        counts = clamp_counts()
        assert counts["nonfinite"] == 1
        assert counts["total"] == 1

    def test_counter_increments_overflow(self):
        _clamp_reward(1e6)
        _clamp_reward(-1e6)
        counts = clamp_counts()
        assert counts["overflow"] == 2
        assert counts["total"] == 2

    def test_reset_clamp_counts(self):
        _clamp_reward(float("nan"))
        reset_clamp_counts()
        assert clamp_counts() == {"nonfinite": 0, "overflow": 0, "total": 0}

    def test_non_numeric_coerces_to_zero(self):
        # A string accidentally passed through is treated as non-finite.
        assert _clamp_reward("not-a-number") == 0.0  # type: ignore[arg-type]
        assert clamp_counts()["nonfinite"] == 1


# ---------------------------------------------------------------------------
# _normalize_logits_2d — WS-T4 shape guard
# ---------------------------------------------------------------------------


class TestNormalizeLogits:
    def test_2d_passthrough(self):
        arr = np.zeros((4, 8))
        assert _normalize_logits_2d(arr) is arr

    def test_3d_collapsed_to_last_batch(self):
        arr = np.arange(24).reshape(2, 3, 4).astype(float)
        out = _normalize_logits_2d(arr)
        assert out.shape == (3, 4)
        # Last batch row, not the first
        assert np.array_equal(out, arr[-1])

    def test_1d_rejected(self):
        with pytest.raises(ValidationError) as exc:
            _normalize_logits_2d(np.zeros(5))
        assert exc.value.code == "carl.logits_shape"
        assert exc.value.context["shape"] == [5]

    def test_4d_rejected(self):
        with pytest.raises(ValidationError):
            _normalize_logits_2d(np.zeros((2, 2, 2, 2)))


# ---------------------------------------------------------------------------
# multiscale_coherence_reward — end-to-end
# ---------------------------------------------------------------------------


class TestMultiscaleReward:
    def _logits(self, T: int = 8, V: int = 16, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return rng.standard_normal((T, V))

    def test_basic_returns_in_range(self):
        logits = self._logits()
        token_ids = np.zeros(logits.shape[0], dtype=int)
        score = multiscale_coherence_reward(logits, token_ids)
        assert 0.0 <= score <= 1.0

    def test_empty_T_returns_zero(self):
        logits = np.zeros((0, 16))
        score = multiscale_coherence_reward(logits, np.zeros(0, dtype=int))
        assert score == 0.0

    def test_empty_V_returns_zero(self):
        logits = np.zeros((4, 0))
        score = multiscale_coherence_reward(logits, np.zeros(4, dtype=int))
        assert score == 0.0

    def test_all_scales_fail_warns_once(self, caplog):
        caplog.set_level(logging.WARNING, logger="carl_studio.training.rewards.multiscale")
        logits = np.zeros((0, 4))
        # First call -> one warning
        multiscale_coherence_reward(logits, np.zeros(0, dtype=int))
        first_count = sum(
            1 for r in caplog.records if "empty logits" in r.getMessage()
        )
        # Second call -> NO additional warning (dedupe)
        multiscale_coherence_reward(logits, np.zeros(0, dtype=int))
        second_count = sum(
            1 for r in caplog.records if "empty logits" in r.getMessage()
        )
        assert first_count == 1
        assert second_count == 1

    def test_3d_logits_accepted(self):
        arr = np.random.default_rng(0).standard_normal((2, 8, 16))
        score = multiscale_coherence_reward(arr, np.zeros(8, dtype=int))
        assert 0.0 <= score <= 1.0

    def test_1d_logits_raise_validation(self):
        with pytest.raises(ValidationError):
            multiscale_coherence_reward(np.zeros(5), np.zeros(5, dtype=int))

    def test_result_is_clamped(self):
        logits = self._logits(T=16)
        score = multiscale_coherence_reward(logits, np.zeros(16, dtype=int))
        # A valid multiscale score is always in [0, 1], safely inside clamp range.
        assert -100.0 <= score <= 100.0

    def test_nonfinite_phi_survives(self, monkeypatch):
        # Force compute_phi to return non-finite Phi; multiscale must survive.
        import carl_studio.training.rewards.multiscale as ms

        def _bad_phi(arr):  # noqa: ANN001
            phi = np.full(arr.shape[0], np.nan)
            probs = np.zeros_like(arr)
            entropy = np.zeros(arr.shape[0])
            return phi, probs, entropy

        monkeypatch.setattr(ms, "compute_phi", _bad_phi)
        score = multiscale_coherence_reward(self._logits(T=8), np.zeros(8, dtype=int))
        assert score == 0.0 or (0.0 <= score <= 1.0)
