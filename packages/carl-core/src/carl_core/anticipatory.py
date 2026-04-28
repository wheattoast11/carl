"""AnticipatoryGate — gate firing on anticipated, not just observed, decoherence (v0.19 §4.3).

Lives in carl-core (not carl-studio.gating) because the anticipatory
trinity is foundational: studio-side gates compose with this primitive,
not the reverse. The carl_studio.gating.BaseGate composes AnticipatoryGate
via its predicate API.

See docs/v19_anticipatory_coherence_design.md sections 4.3 / 10.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from carl_core.errors import ValidationError
from carl_core.forecast import CoherenceForecast


@dataclass(frozen=True, eq=False)
class GateResult:
    """Outcome of an anticipatory gate evaluation."""

    allowed: bool
    warning_step: int | None
    reason: str


@dataclass(frozen=True, eq=False)
class AnticipatoryPredicate:
    """Predicate that checks a forecast for anticipated decoherence.

    Two failure modes:
      1. phi crosses threshold within horizon (early warning)
      2. lyapunov drift below -drift_threshold (decay too steep)

    Either fires ⇒ predicate denies.
    """

    forecast: CoherenceForecast
    threshold: float = 0.5
    drift_threshold: float = 0.1
    name: str = "anticipatory"

    def check(self) -> tuple[bool, str]:
        """Return (allowed, reason)."""
        warn_step = self.forecast.early_warning(threshold=self.threshold)
        if warn_step is not None:
            return False, f"anticipatory warning at step {warn_step}"
        drift = self.forecast.lyapunov_drift()
        if drift < -self.drift_threshold:
            return False, f"anticipatory drift {drift:.4f} below -{self.drift_threshold}"
        return True, "anticipatory ok"


@dataclass(frozen=True, eq=False)
class AnticipatoryGate:
    """Gate that fires on anticipated decoherence.

    Composes with carl_studio.gating.BaseGate via the predicate API.
    """

    horizon: int
    threshold: float
    drift_threshold: float

    def __post_init__(self) -> None:
        if self.horizon <= 0:
            raise ValidationError(
                f"horizon must be positive int, got {self.horizon}",
                code="carl.gate.anticipatory_threshold_invalid",
            )
        if not (0.0 <= self.threshold <= 1.0):
            raise ValidationError(
                f"threshold must lie in [0, 1], got {self.threshold}",
                code="carl.gate.anticipatory_threshold_invalid",
            )
        if self.drift_threshold < 0.0:
            raise ValidationError(
                f"drift_threshold must be non-negative, got {self.drift_threshold}",
                code="carl.gate.anticipatory_threshold_invalid",
            )

    def evaluate(
        self,
        state: dict[str, Any],
        *,
        forecast_fn: Callable[[dict[str, Any]], CoherenceForecast | None],
    ) -> GateResult:
        """Run the gate. Errors in forecast_fn surface as forecast_unavailable."""
        try:
            fc = forecast_fn(state)
        except Exception as exc:
            raise ValidationError(
                "forecast_fn raised; forecast unavailable",
                code="carl.gate.forecast_unavailable",
                cause=exc,
            ) from exc
        if fc is None:
            raise ValidationError(
                "forecast_fn returned None; forecast unavailable",
                code="carl.gate.forecast_unavailable",
            )
        if fc.horizon_steps != self.horizon:
            raise ValidationError(
                f"forecast horizon {fc.horizon_steps} != gate horizon {self.horizon}",
                code="carl.gate.forecast_unavailable",
            )
        pred = AnticipatoryPredicate(
            forecast=fc,
            threshold=self.threshold,
            drift_threshold=self.drift_threshold,
        )
        allowed, reason = pred.check()
        warn_step = (
            fc.early_warning(threshold=self.threshold) if not allowed else None
        )
        return GateResult(allowed=allowed, warning_step=warn_step, reason=reason)


__all__ = ["AnticipatoryGate", "AnticipatoryPredicate", "GateResult"]
