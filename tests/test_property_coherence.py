"""Property tests for carl_core.math.compute_phi.

Invariants:
- For any finite 2D logit matrix [T, V] with T <= 64, V in [2, 1024]:
  phi is finite, in [0, 1], and deterministic across calls.
"""
from __future__ import annotations

import math

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from carl_core.math import compute_phi


def _finite_logits_strategy(
    max_t: int = 16, max_v: int = 128,
) -> st.SearchStrategy[np.ndarray]:
    """Draw a [T, V] float32 logit matrix with finite entries."""

    @st.composite
    def _strategy(draw: st.DrawFn) -> np.ndarray:
        t = draw(st.integers(min_value=1, max_value=max_t))
        v = draw(st.integers(min_value=2, max_value=max_v))
        flat = draw(
            st.lists(
                st.floats(
                    min_value=-30.0,
                    max_value=30.0,
                    allow_nan=False,
                    allow_infinity=False,
                    width=32,
                ),
                min_size=t * v,
                max_size=t * v,
            )
        )
        return np.array(flat, dtype=np.float32).reshape(t, v)

    return _strategy()


@given(logits=_finite_logits_strategy())
@settings(max_examples=75, deadline=None)
def test_phi_is_finite(logits: np.ndarray) -> None:
    """phi has no NaN/inf for any finite input."""
    phi, probs, entropy = compute_phi(logits)
    assert phi.shape == (logits.shape[0],)
    assert np.isfinite(phi).all()
    assert np.isfinite(probs).all()
    assert np.isfinite(entropy).all()


@given(logits=_finite_logits_strategy())
@settings(max_examples=75, deadline=None)
def test_phi_in_unit_range(logits: np.ndarray) -> None:
    """phi values lie in [0, 1] (allowing a tiny numerical epsilon)."""
    phi, _, _ = compute_phi(logits)
    # The ``+ 1e-10`` smoothing inside log_probs can push phi slightly above 1
    # for near-uniform distributions; accept a small tolerance.
    epsilon = 1e-6
    assert phi.min() >= -epsilon
    assert phi.max() <= 1.0 + epsilon


@given(logits=_finite_logits_strategy())
@settings(max_examples=75, deadline=None)
def test_phi_deterministic(logits: np.ndarray) -> None:
    """Same input -> same output across runs (pure function)."""
    phi1, p1, e1 = compute_phi(logits)
    phi2, p2, e2 = compute_phi(logits)
    assert np.array_equal(phi1, phi2)
    assert np.array_equal(p1, p2)
    assert np.array_equal(e1, e2)


@given(
    t=st.integers(min_value=1, max_value=8),
    v=st.integers(min_value=2, max_value=128),
)
@settings(max_examples=50, deadline=None)
def test_uniform_logits_phi_near_zero(t: int, v: int) -> None:
    """For uniform logits, entropy = log(V) so phi approximates 0."""
    logits = np.zeros((t, v), dtype=np.float32)
    phi, _, _ = compute_phi(logits)
    # 1e-10 smoothing introduces a small bias but phi should be << 1.
    expected_max = 1e-3
    assert (phi < expected_max).all()


@given(
    t=st.integers(min_value=1, max_value=8),
    v=st.integers(min_value=2, max_value=128),
)
@settings(max_examples=50, deadline=None)
def test_peaked_logits_phi_near_one(t: int, v: int) -> None:
    """Logits spiked on one token: entropy -> 0 and phi -> 1."""
    logits = np.full((t, v), fill_value=-30.0, dtype=np.float32)
    logits[:, 0] = 30.0
    phi, _, _ = compute_phi(logits)
    # All rows should be close to 1.
    assert phi.min() > 1.0 - 1e-2
    assert math.isfinite(float(phi.max()))
