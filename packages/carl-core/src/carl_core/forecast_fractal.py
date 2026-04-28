"""FractalForecast — multi-scale CoherenceForecast over phi-spaced bands (v0.19 §4.4).

KAM theorem (Arnold 1963, Moser 1962): the most-Diophantine number is the
most stable winding ratio under perturbation. φ has continued-fraction
expansion [1;1,1,1,...], minimizing the constants in the Diophantine
condition. This makes φ-scaled bands uniquely optimal for anti-resonant
multi-scale decomposition.

K=4 bands span φ^3 ≈ 4.236× dynamic range — half a decade — empirically
sufficient for token / action / episode / curriculum scales. The current
MAX_DEPTH=4 in EMLTree and compose_resonants is *derivable* under this
principle, not arbitrary.

See docs/v19_anticipatory_coherence_design.md sections 2.5 / 4.4 / 9.
"""
from __future__ import annotations

from typing import Any, Callable

from carl_core.constants import PHI
from carl_core.errors import ValidationError
from carl_core.forecast import CoherenceForecast

# Soft cap on scale-band count. φ^10 ≈ 122.99 — beyond this, integer
# horizon scaling overflows step-budget on practical training loops.
_MAX_BANDS: int = 10


def phi_scale_bands(num_bands: int = 4) -> tuple[float, ...]:
    """Return num_bands phi-scaled multipliers: (1, φ, φ², …, φ^{n-1})."""
    if num_bands <= 0:
        raise ValidationError(
            f"num_bands must be positive, got {num_bands}",
            code="carl.fractal.band_count_invalid",
        )
    if num_bands > _MAX_BANDS:
        raise ValidationError(
            f"num_bands {num_bands} > max {_MAX_BANDS}; risks scale overflow",
            code="carl.fractal.scale_overflow",
        )
    return tuple(PHI**k for k in range(num_bands))


BaseForecastFn = Callable[..., CoherenceForecast]


class FractalForecast:
    """Multi-scale CoherenceForecast over φ-spaced bands.

    Wraps a base forecast function and invokes it once per band with a
    horizon scaled by the band's φ-multiplier. Aggregate warning fires
    when ANY band fires (band-agnostic decision).
    """

    def __init__(
        self,
        base_forecast_fn: BaseForecastFn,
        num_bands: int = 4,
    ) -> None:
        self.base_forecast_fn = base_forecast_fn
        self.bands: tuple[float, ...] = phi_scale_bands(num_bands)

    def forecast(
        self,
        state: dict[str, Any] | Any,
        base_horizon: int,
    ) -> list[CoherenceForecast]:
        """Produce one CoherenceForecast per band, scaled by φ^k."""
        if base_horizon <= 0:
            raise ValidationError(
                f"base_horizon must be positive, got {base_horizon}",
                code="carl.fractal.band_count_invalid",
            )
        out: list[CoherenceForecast] = []
        for scale in self.bands:
            scaled_horizon = max(1, int(base_horizon * scale))
            fc = self.base_forecast_fn(
                state, horizon=scaled_horizon, scale_band=scale
            )
            out.append(fc)
        return out

    def aggregate_warning(
        self,
        forecasts: list[CoherenceForecast],
        threshold: float = 0.5,
    ) -> dict[str, int | None]:
        """Per-band early-warning steps. Key format: 'band_{i}_phi^{i}'."""
        return {
            f"band_{i}_phi^{i}": fc.early_warning(threshold=threshold)
            for i, fc in enumerate(forecasts)
        }

    def any_band_fires(
        self,
        forecasts: list[CoherenceForecast],
        threshold: float = 0.5,
    ) -> bool:
        """True iff any band's early_warning fires."""
        return any(
            fc.early_warning(threshold=threshold) is not None for fc in forecasts
        )


__all__ = ["FractalForecast", "phi_scale_bands"]
