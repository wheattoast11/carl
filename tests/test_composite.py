"""Tests for CARLReward composite — WS-T3 (clamping) + WS-T4 (shape)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from carl_core.errors import ValidationError

from carl_studio.training.rewards.composite import (
    CARLReward,
    _ensure_2d_logits,
)
from carl_studio.training.rewards.multiscale import reset_clamp_counts


@pytest.fixture(autouse=True)
def _fresh_state():
    reset_clamp_counts()
    yield
    reset_clamp_counts()


def _make_logits(T: int = 8, V: int = 16, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((T, V))


# ---------------------------------------------------------------------------
# _ensure_2d_logits
# ---------------------------------------------------------------------------


class TestShapeGuard:
    def test_2d_passthrough(self):
        arr = _make_logits()
        assert _ensure_2d_logits(arr).shape == arr.shape

    def test_3d_collapsed(self):
        arr = np.random.default_rng(1).standard_normal((3, 8, 16))
        out = _ensure_2d_logits(arr)
        assert out.shape == (8, 16)

    def test_1d_rejected(self):
        with pytest.raises(ValidationError) as exc:
            _ensure_2d_logits(np.zeros(5))
        assert exc.value.code == "carl.logits_shape"

    def test_empty_2d_rejected(self):
        with pytest.raises(ValidationError):
            _ensure_2d_logits(np.zeros((0, 16)))
        with pytest.raises(ValidationError):
            _ensure_2d_logits(np.zeros((8, 0)))


# ---------------------------------------------------------------------------
# CARLReward.score()
# ---------------------------------------------------------------------------


class TestCARLRewardScore:
    def test_basic_score(self):
        carl = CARLReward()
        logits = _make_logits(T=16)
        token_ids = np.zeros(16, dtype=int)
        score, components = carl.score(logits, token_ids)
        assert isinstance(score, float)
        assert math.isfinite(score)
        # All three keys present
        assert set(components.keys()) == {"multiscale", "cloud_quality", "discontinuity"}
        for v in components.values():
            assert math.isfinite(v)
            assert -100.0 <= v <= 100.0

    def test_score_3d_logits(self):
        carl = CARLReward()
        logits = np.random.default_rng(0).standard_normal((2, 16, 32))
        token_ids = np.zeros(16, dtype=int)
        score, components = carl.score(logits, token_ids)
        assert math.isfinite(score)

    def test_score_1d_logits_raises(self):
        carl = CARLReward()
        with pytest.raises(ValidationError):
            carl.score(np.zeros(8), np.zeros(8, dtype=int))

    def test_score_empty_rejected(self):
        carl = CARLReward()
        with pytest.raises(ValidationError):
            carl.score(np.zeros((0, 16)), np.zeros(0, dtype=int))


# ---------------------------------------------------------------------------
# score_from_trace clamping (WS-T3)
# ---------------------------------------------------------------------------


class TestScoreFromTraceClamping:
    class _FakeTrace:
        """Minimal stand-in for CoherenceTrace with tunable components."""

        def __init__(
            self,
            composite: float,
            multiscale: float = 0.5,
            cloud: float = 0.5,
            disc: float = 0.5,
        ) -> None:
            self._composite = composite
            self.multiscale_coherence = multiscale
            self.cloud_quality = cloud
            self.discontinuity_score = disc

        def carl_reward(self, **_kwargs):  # noqa: ANN003
            return self._composite

    def test_nan_composite_becomes_zero(self):
        carl = CARLReward()
        trace = self._FakeTrace(composite=float("nan"))
        score, _ = carl.score_from_trace(trace)  # type: ignore[arg-type]
        assert score == 0.0

    def test_positive_inf_composite_becomes_zero(self):
        carl = CARLReward()
        trace = self._FakeTrace(composite=float("inf"))
        score, _ = carl.score_from_trace(trace)  # type: ignore[arg-type]
        assert score == 0.0

    def test_negative_inf_composite_becomes_zero(self):
        carl = CARLReward()
        trace = self._FakeTrace(composite=float("-inf"))
        score, _ = carl.score_from_trace(trace)  # type: ignore[arg-type]
        assert score == 0.0

    def test_runaway_positive_clips_to_100(self):
        carl = CARLReward()
        trace = self._FakeTrace(composite=1e9)
        score, _ = carl.score_from_trace(trace)  # type: ignore[arg-type]
        assert score == 100.0

    def test_runaway_negative_clips_to_minus_100(self):
        carl = CARLReward()
        trace = self._FakeTrace(composite=-1e9)
        score, _ = carl.score_from_trace(trace)  # type: ignore[arg-type]
        assert score == -100.0

    def test_components_clamped_too(self):
        carl = CARLReward()
        trace = self._FakeTrace(
            composite=0.5,
            multiscale=float("nan"),
            cloud=float("inf"),
            disc=-1e9,
        )
        _, components = carl.score_from_trace(trace)  # type: ignore[arg-type]
        assert components["multiscale"] == 0.0
        assert components["cloud_quality"] == 0.0
        assert components["discontinuity"] == -100.0
