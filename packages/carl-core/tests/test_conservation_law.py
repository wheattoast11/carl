"""Conservation-law smoke: integral of (1 - phi_t) over T_STAR(d) (SEM-008).

Per Desai (2026), in a conservation-law regime the accumulated
distance-from-crystallization should relate to SIGMA * T_STAR(d). This
test is falsifiable -- if the observed trajectory diverges from theory
by more than an order of magnitude, either the theory needs a correction
or the synthetic trajectory isn't representative.

We do NOT run a real training loop here (too slow for unit tests).
Instead we synthesize a plausible phi(t) trajectory by constructing
CoherenceTrace objects from logits at progressively lower temperatures
(a textbook crystallization schedule).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from carl_core.coherence_trace import CoherenceTrace
from carl_core.constants import KAPPA, SIGMA, T_STAR


def _synthesize_trajectory(d: int, steps: int, seed: int = 42) -> list[float]:
    """Simulate phi_mean(t) over ``steps`` training steps at vocab size ``d``.

    Uses an exponential temperature decay schedule. Coherence (phi) rises
    as temperature falls -- the canonical "cooling into order" trajectory
    that the conservation law should govern.
    """
    rng = np.random.default_rng(seed)
    phi_series: list[float] = []
    T_tokens = 16  # tokens per generated completion
    for t in range(steps):
        # Temperature decays -> distribution sharpens -> phi rises.
        temperature = max(0.1, 2.0 * math.exp(-t / max(steps / 3, 1.0)))
        logits = rng.normal(scale=temperature, size=(T_tokens, d)).astype(
            np.float64
        )
        # Select argmax as the "chosen" token. Shape: [T_tokens].
        token_ids = np.argmax(logits, axis=-1).astype(np.int64)
        trace = CoherenceTrace.from_logits(logits, token_ids, step=t)
        phi_series.append(trace.phi_mean)
    return phi_series


@pytest.mark.slow
def test_conservation_law_matches_theory_within_tolerance() -> None:
    """Order-of-magnitude check on the conservation-law prediction.

    Marked ``slow`` because logits sampling at vocab=512 over ~100 steps
    takes a few seconds. Run with ``pytest -m slow``.
    """
    d = 512  # token vocab proxy
    horizon = T_STAR(d)  # kappa * d
    # Scale down to keep the test tractable while preserving shape.
    steps = max(horizon // 100, 50)

    phi_series = _synthesize_trajectory(d, steps)

    # Accumulated distance-from-crystallization over the trajectory.
    integral = sum(1.0 - p for p in phi_series)
    # Theory predicts SIGMA * steps in the reduced-horizon regime.
    predicted = SIGMA * steps

    # Loose tolerance -- we check order of magnitude on synthetic data.
    lower_bound = 0.1 * predicted
    upper_bound = 10.0 * predicted
    assert lower_bound <= integral <= upper_bound, (
        f"observed sum(1-phi) = {integral:.3f} diverged from theory "
        f"SIGMA*steps = {predicted:.3f} "
        f"(tolerance: [{lower_bound:.3f}, {upper_bound:.3f}])"
    )


def test_kappa_sigma_constants_have_documented_values() -> None:
    """KAPPA=64/3, SIGMA=3/16, SIGMA*KAPPA=4 (Desai 2026)."""
    assert KAPPA == 64 / 3
    assert SIGMA == 3 / 16
    # SIGMA * KAPPA = (3/16) * (64/3) = 64/16 = 4 exactly.
    assert abs(SIGMA * KAPPA - 4.0) < 1e-9


def test_t_star_integer_horizon() -> None:
    """T_STAR(d) = floor(kappa * d) and is integer-typed."""
    assert isinstance(T_STAR(512), int)
    assert T_STAR(512) == int(KAPPA * 512)
    # Spot-check a few more: T_STAR should scale linearly with d.
    assert T_STAR(1024) == int(KAPPA * 1024)
    assert T_STAR(1) == int(KAPPA * 1)  # = 21
