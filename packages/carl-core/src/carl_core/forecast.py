"""CoherenceForecast — future-tense coherence prediction (v0.19 §4.1).

Closes the temporal coherence trinity. Prospective dual to compute_phi
(present) and compose_resonants (past). Pure math; numpy only; no torch,
no learned models.

See docs/v19_anticipatory_coherence_design.md sections 2.2 / 4.1.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from carl_core.errors import ValidationError

ForecastMethod = Literal["linear", "lyapunov", "learned"]
_VALID_METHODS: tuple[str, ...] = ("linear", "lyapunov", "learned")


@dataclass(frozen=True, eq=False)
class CoherenceForecast:
    """Future-tense coherence prediction over a horizon.

    Attributes:
        horizon_steps: Number of future steps predicted (must be > 0).
        phi_predicted: Anticipated coherence at each step. Shape: (horizon_steps,).
        subspace_prior: Anticipated subspace geometry. Shape: (horizon_steps, dim).
        substrate_health: Anticipated substrate alignment Ψ. Shape: (horizon_steps,).
        confidence: Per-step forecast confidence in [0, 1]. Shape: (horizon_steps,).
        scale_band: Which φ-scale band this forecast operates at.
        method: Generation method ("linear", "lyapunov", "learned").
    """

    horizon_steps: int
    phi_predicted: NDArray[np.float64]
    subspace_prior: NDArray[np.float64]
    substrate_health: NDArray[np.float64]
    confidence: NDArray[np.float64]
    scale_band: float
    method: str

    def __post_init__(self) -> None:
        if self.horizon_steps <= 0:
            raise ValidationError(
                f"horizon_steps must be positive int, got {self.horizon_steps}",
                code="carl.forecast.horizon_invalid",
            )
        if self.method not in _VALID_METHODS:
            raise ValidationError(
                f"method must be one of {_VALID_METHODS}, got {self.method!r}",
                code="carl.forecast.method_unknown",
            )
        for name in ("phi_predicted", "substrate_health", "confidence"):
            arr = getattr(self, name)
            if arr.shape != (self.horizon_steps,):
                raise ValidationError(
                    f"{name} shape {arr.shape} != (horizon_steps={self.horizon_steps},)",
                    code="carl.forecast.horizon_invalid",
                )
        if self.subspace_prior.ndim != 2 or self.subspace_prior.shape[0] != self.horizon_steps:
            raise ValidationError(
                f"subspace_prior shape {self.subspace_prior.shape} must be "
                f"(horizon_steps={self.horizon_steps}, dim)",
                code="carl.forecast.horizon_invalid",
            )
        if np.any(self.confidence < 0.0) or np.any(self.confidence > 1.0):
            raise ValidationError(
                "confidence values must lie in [0, 1]",
                code="carl.forecast.uncertainty_overflow",
            )

    def early_warning(self, threshold: float = 0.5) -> int | None:
        """Step index where phi first crosses below threshold; None if never."""
        if not (0.0 <= threshold <= 1.0):
            raise ValidationError(
                f"threshold must lie in [0, 1], got {threshold}",
                code="carl.forecast.uncertainty_overflow",
            )
        below = np.where(self.phi_predicted < threshold)[0]
        return int(below[0]) if below.size > 0 else None

    def lyapunov_drift(self) -> float:
        """Finite-time Lyapunov estimate: slope of log(phi) over horizon.

        Positive ⇒ growing coherence. Negative ⇒ decaying. Zero ⇒ constant.
        Returns 0.0 for single-step or all-zero series (degenerate).
        """
        if self.horizon_steps < 2:
            return 0.0
        phi = self.phi_predicted
        if not np.any(phi > 1e-12):
            return 0.0
        clipped = np.maximum(phi, 1e-12)
        log_phi = np.log(clipped)
        # Linear regression slope of log(phi) vs step index
        t = np.arange(self.horizon_steps, dtype=float)
        slope = float(np.polyfit(t, log_phi, 1)[0])
        if not math.isfinite(slope):
            return 0.0
        return slope

    def confidence_interval(
        self, alpha: float = 0.05
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """(lo, hi) prediction-interval bounds at significance level alpha.

        Smaller alpha (e.g. 0.05 = 95% CI) yields wider intervals than
        larger alpha (e.g. 0.32 = 68% CI). alpha must lie in (0, 1) exclusive.
        """
        if not (0.0 < alpha < 1.0):
            raise ValidationError(
                f"alpha must lie strictly in (0, 1), got {alpha}",
                code="carl.forecast.uncertainty_overflow",
            )
        # Width factor monotonically decreasing in alpha
        z = math.sqrt(-2.0 * math.log(alpha / 2.0))
        stderr = 1.0 - self.confidence
        half_width = z * stderr
        lo = self.phi_predicted - half_width
        hi = self.phi_predicted + half_width
        return lo, hi

    def phi_at(self, step: int) -> float:
        """phi predicted at a given horizon step. Raises if out of range."""
        if step < 0 or step >= self.horizon_steps:
            raise ValidationError(
                f"step {step} out of range [0, {self.horizon_steps})",
                code="carl.forecast.horizon_invalid",
            )
        return float(self.phi_predicted[step])

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-safe dict."""
        return {
            "horizon_steps": self.horizon_steps,
            "phi_predicted": self.phi_predicted.tolist(),
            "subspace_prior": self.subspace_prior.tolist(),
            "substrate_health": self.substrate_health.tolist(),
            "confidence": self.confidence.tolist(),
            "scale_band": self.scale_band,
            "method": self.method,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CoherenceForecast:
        """Reconstruct from to_dict() output."""
        return cls(
            horizon_steps=int(payload["horizon_steps"]),
            phi_predicted=np.asarray(payload["phi_predicted"], dtype=np.float64),
            subspace_prior=np.asarray(payload["subspace_prior"], dtype=np.float64),
            substrate_health=np.asarray(payload["substrate_health"], dtype=np.float64),
            confidence=np.asarray(payload["confidence"], dtype=np.float64),
            scale_band=float(payload["scale_band"]),
            method=str(payload["method"]),
        )


__all__ = ["CoherenceForecast", "ForecastMethod"]
