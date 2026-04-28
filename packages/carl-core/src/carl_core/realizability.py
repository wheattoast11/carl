"""realizability_gap — KL divergence between forecast and observed coherence (v0.19 §4.5).

By Kleene realizability + Curry-Howard: a forecast is a *witness* to a
future state; its realizability is the *constructive proof* that the
forecast is sound. R(τ) → 0 is the proof-completion event. This IS the
realizability operator.

Decision derived constructively in v0.19 design doc §10 (the procedure of
deciding emulates the trinity it implements).

See docs/v19_anticipatory_coherence_design.md sections 2.3 / 4.5 / 10.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from carl_core.errors import ValidationError
from carl_core.forecast import CoherenceForecast


def _kl_divergence(
    p: NDArray[np.float64],
    q: NDArray[np.float64],
    *,
    epsilon: float = 1e-12,
) -> float:
    """KL(p || q) over normalized non-negative arrays. epsilon-stable."""
    p_safe = np.maximum(np.asarray(p, dtype=np.float64), epsilon)
    q_safe = np.maximum(np.asarray(q, dtype=np.float64), epsilon)
    p_norm = p_safe / p_safe.sum()
    q_norm = q_safe / q_safe.sum()
    return float(np.sum(p_norm * np.log(p_norm / q_norm)))


def realizability_gap(
    forecast: CoherenceForecast,
    observed_phi: NDArray[np.float64] | Sequence[float],
    *,
    horizon: int | None = None,
    epsilon: float = 1e-12,
) -> float:
    """KL divergence between forecasted and observed coherence distributions.

    R(τ) → 0 iff prediction is exact (constructive Tarski). Vanishing gap
    is the proof-completion event for the forecast as a constructive
    witness to its observed outcome.

    Args:
        forecast: The CoherenceForecast under test.
        observed_phi: Realized phi values, same shape as forecast.phi_predicted
            (or longer; will be sliced to match if horizon is None).
        horizon: If provided, compare only at this single step. Otherwise,
            compare full arrays.
        epsilon: Numerical floor for log domain.

    Returns:
        Non-negative real (KL divergence).

    Raises:
        ValidationError(carl.realizability.horizon_mismatch) on shape mismatch.
        ValidationError(carl.realizability.divergence) if KL is non-finite.
    """
    obs = np.asarray(observed_phi, dtype=np.float64)
    if horizon is not None:
        if horizon < 0 or horizon >= forecast.horizon_steps:
            raise ValidationError(
                f"horizon {horizon} out of range [0, {forecast.horizon_steps})",
                code="carl.realizability.horizon_mismatch",
            )
        if obs.ndim == 0 or obs.size <= horizon:
            raise ValidationError(
                f"observed_phi too short for horizon {horizon}: size={obs.size}",
                code="carl.realizability.horizon_mismatch",
            )
        f_arr = np.array([forecast.phi_at(horizon)], dtype=np.float64)
        o_arr = np.array([float(obs[horizon])], dtype=np.float64)
    else:
        if obs.shape != forecast.phi_predicted.shape:
            raise ValidationError(
                f"observed_phi shape {obs.shape} != forecast.phi_predicted "
                f"{forecast.phi_predicted.shape}",
                code="carl.realizability.horizon_mismatch",
            )
        f_arr = forecast.phi_predicted
        o_arr = obs

    gap = _kl_divergence(o_arr, f_arr, epsilon=epsilon)
    if not np.isfinite(gap):
        raise ValidationError(
            f"realizability gap non-finite: {gap}",
            code="carl.realizability.divergence",
        )
    return gap


def realizability_chain(
    forecasts: Sequence[CoherenceForecast],
    observations: Sequence[NDArray[np.float64] | Sequence[float]],
    horizons: Sequence[int | None],
) -> float:
    """Composed realizability gap over a sequence.

    Sum of per-step realizability gaps. By the data processing inequality
    (Cover & Thomas Thm 2.8.1): R(τ_1) + R(τ_2) ≥ R(τ_1 + τ_2). The
    chain is an upper bound on the joint realizability cost.

    Args:
        forecasts: Sequence of CoherenceForecast objects.
        observations: Realized phi arrays, one per forecast.
        horizons: Optional horizon for each forecast (None ⇒ full-array).

    Returns:
        Sum of per-element realizability_gap calls. Empty input returns 0.0.
    """
    if not (len(forecasts) == len(observations) == len(horizons)):
        raise ValidationError(
            f"length mismatch: forecasts={len(forecasts)}, "
            f"observations={len(observations)}, horizons={len(horizons)}",
            code="carl.realizability.horizon_mismatch",
        )
    total = 0.0
    for fc, obs, h in zip(forecasts, observations, horizons):
        total += realizability_gap(fc, obs, horizon=h)
    return total


__all__ = ["realizability_gap", "realizability_chain"]
