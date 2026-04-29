"""Analytic forecaster + trinity view for v0.19 anticipatory coherence (FREE tier).

Pure-numpy linear/lyapunov extrapolation of phi history. No torch, no
learned models — those live in the [anticipatory] PAID extra.

Trinity view composes past (resonant integration), present (compute_phi),
and future (CoherenceForecast) for CLI / observability surfaces.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from carl_core import CoherenceForecast


def linear_forecast(
    phi_history: list[float] | NDArray[np.float64],
    horizon: int,
    *,
    scale_band: float = 1.0,
    base_confidence: float = 0.7,
) -> CoherenceForecast:
    """Linear extrapolation of phi history to produce a CoherenceForecast.

    Fits a linear regression to ``phi_history`` (positions 0..N-1) and
    projects N..N+horizon-1. Confidence decays linearly with horizon
    (per-step decay = (1 - base_confidence) / horizon).

    History of length 0 returns a flat 0.5-phi forecast at low confidence.
    """
    phi = np.asarray(phi_history, dtype=np.float64)
    if horizon <= 0:
        raise ValueError("horizon must be positive")

    if phi.size == 0:
        phi_pred = np.full(horizon, 0.5)
        intercept = 0.5
        slope = 0.0
    elif phi.size == 1:
        phi_pred = np.full(horizon, float(phi[0]))
        intercept = float(phi[0])
        slope = 0.0
    else:
        t = np.arange(phi.size, dtype=np.float64)
        slope, intercept = np.polyfit(t, phi, 1)
        future_t = np.arange(phi.size, phi.size + horizon, dtype=np.float64)
        phi_pred = np.clip(intercept + slope * future_t, 0.0, 1.0)

    # confidence decays linearly with how far forward we project
    conf = np.clip(
        base_confidence - np.arange(horizon, dtype=np.float64) * (base_confidence / max(horizon, 1)),
        0.05,
        1.0,
    )
    return CoherenceForecast(
        horizon_steps=horizon,
        phi_predicted=phi_pred,
        subspace_prior=np.zeros((horizon, 1)),  # FREE: no prior over geometry
        substrate_health=np.full(horizon, _slope_to_health(slope)),
        confidence=conf,
        scale_band=scale_band,
        method="linear",
    )


def lyapunov_forecast(
    phi_history: list[float] | NDArray[np.float64],
    horizon: int,
    *,
    scale_band: float = 1.0,
    base_confidence: float = 0.6,
) -> CoherenceForecast:
    """Exponential extrapolation: assume log(phi) grows linearly.

    Captures Lyapunov-style coherence drift more faithfully than linear
    when history shows multiplicative (exponential) decay/growth.
    """
    phi = np.asarray(phi_history, dtype=np.float64)
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    safe_phi = np.maximum(phi, 1e-9)

    if safe_phi.size < 2:
        return linear_forecast(
            phi_history, horizon, scale_band=scale_band, base_confidence=base_confidence
        )

    log_phi = np.log(safe_phi)
    t = np.arange(safe_phi.size, dtype=np.float64)
    slope, intercept = np.polyfit(t, log_phi, 1)
    future_t = np.arange(safe_phi.size, safe_phi.size + horizon, dtype=np.float64)
    phi_pred = np.clip(np.exp(intercept + slope * future_t), 0.0, 1.0)

    conf = np.clip(
        base_confidence - np.arange(horizon, dtype=np.float64) * (base_confidence / max(horizon, 1)),
        0.05,
        1.0,
    )
    return CoherenceForecast(
        horizon_steps=horizon,
        phi_predicted=phi_pred,
        subspace_prior=np.zeros((horizon, 1)),
        substrate_health=np.full(horizon, _slope_to_health(float(slope))),
        confidence=conf,
        scale_band=scale_band,
        method="lyapunov",
    )


def _slope_to_health(slope: float) -> float:
    """Map slope ∈ [-∞, ∞] -> health ∈ [0, 1] via sigmoid."""
    # Sigmoid: positive slope -> health > 0.5, negative -> < 0.5
    return float(1.0 / (1.0 + np.exp(-10.0 * slope)))


@dataclass(frozen=True, eq=False)
class TrinityView:
    """Past + present + future coherence summary for display."""

    past_summary: dict[str, Any]
    present_phi: float
    forecast: CoherenceForecast

    def to_dict(self) -> dict[str, Any]:
        return {
            "past": self.past_summary,
            "present": {"phi": self.present_phi},
            "future": self.forecast.to_dict(),
        }

    def early_warning_step(self, threshold: float = 0.5) -> int | None:
        return self.forecast.early_warning(threshold=threshold)


def make_trinity_view(
    phi_history: list[float] | NDArray[np.float64],
    present_phi: float,
    horizon: int,
    *,
    method: str = "linear",
) -> TrinityView:
    """Compose past + present + future into a single display object.

    Past is summarized as min/mean/max over the history window. Present
    is the supplied scalar. Future is computed via the chosen method.
    """
    arr = np.asarray(phi_history, dtype=np.float64)
    if arr.size == 0:
        past_summary: dict[str, Any] = {"min": None, "mean": None, "max": None, "n": 0}
    else:
        past_summary = {
            "min": float(arr.min()),
            "mean": float(arr.mean()),
            "max": float(arr.max()),
            "n": int(arr.size),
        }
    if method == "lyapunov":
        fc = lyapunov_forecast(phi_history, horizon)
    else:
        fc = linear_forecast(phi_history, horizon)
    return TrinityView(
        past_summary=past_summary, present_phi=float(present_phi), forecast=fc
    )


__all__ = [
    "linear_forecast",
    "lyapunov_forecast",
    "TrinityView",
    "make_trinity_view",
]
