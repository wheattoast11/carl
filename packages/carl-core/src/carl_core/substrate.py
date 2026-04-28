"""SubstrateChannel and SubstrateState — multi-channel substrate health (v0.19 §4.2).

Substrate-cognition isomorphism: same algebra applies to compute substrate
(memory/gpu/fd), cognitive substrate (attention/working_set), and
conversational substrate (session/grounding/persona).

Leading-indicator algebra (Scheffer et al. 2009): drift_rate > 0 AND
drift_acceleration < 0 ⇒ imminent collapse. This is the only
substrate-agnostic early warning mathematically guaranteed to fire before
catastrophic decoherence on smooth dynamical systems.

See docs/v19_anticipatory_coherence_design.md sections 2.4 / 2.6 / 4.2.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from carl_core.errors import ValidationError


@dataclass(frozen=True, eq=False)
class SubstrateChannel:
    """One channel of substrate health.

    Attributes:
        name: Channel identifier (e.g., "memory", "compute", "tokenizer",
            "context", "attention"). Must be non-empty.
        health: Current channel health in [0, 1].
        drift_rate: dΨ/dt — first derivative of channel alignment.
        drift_acceleration: d²Ψ/dt² — second derivative.
    """

    name: str
    health: float
    drift_rate: float
    drift_acceleration: float

    def __post_init__(self) -> None:
        if not self.name:
            raise ValidationError(
                "channel name must be non-empty",
                code="carl.substrate.channel_unknown",
            )
        if not (0.0 <= self.health <= 1.0):
            raise ValidationError(
                f"health must lie in [0, 1], got {self.health}",
                code="carl.substrate.channel_unknown",
            )

    @property
    def leading_indicator(self) -> bool:
        """True iff drift_rate > 0 AND drift_acceleration < 0.

        The Scheffer leading-indicator condition: alignment is currently
        rising but the second derivative is negative — collapse anticipated.
        """
        return self.drift_rate > 0.0 and self.drift_acceleration < 0.0


def _empty_channels() -> dict[str, SubstrateChannel]:
    return {}


@dataclass(frozen=True, eq=False)
class SubstrateState:
    """Multi-channel substrate snapshot at a timestamp."""

    timestamp: float
    channels: dict[str, SubstrateChannel] = field(default_factory=_empty_channels)
    overall_psi: float = 0.0

    def __post_init__(self) -> None:
        if not (0.0 <= self.overall_psi <= 1.0):
            raise ValidationError(
                f"overall_psi must lie in [0, 1], got {self.overall_psi}",
                code="carl.substrate.channel_unknown",
            )

    def critical_channels(self) -> list[str]:
        """Names of channels whose leading indicator has fired."""
        return [name for name, ch in self.channels.items() if ch.leading_indicator]

    def lyapunov_exponent_estimate(self, history: list[SubstrateState]) -> float:
        """Estimate finite-time Lyapunov exponent of overall_psi.

        Uses linear regression slope of log(psi) vs timestamp across history
        and self. Decaying psi ⇒ negative exponent; growing ⇒ positive;
        constant ⇒ zero.

        Requires len(history) >= 2 (so combined ≥ 3 points). Smaller
        histories raise carl.substrate.history_too_short.
        """
        if len(history) < 2:
            raise ValidationError(
                f"need >= 2 history points, got {len(history)}",
                code="carl.substrate.history_too_short",
            )
        times = np.array([s.timestamp for s in history] + [self.timestamp], dtype=float)
        psis = np.array(
            [s.overall_psi for s in history] + [self.overall_psi], dtype=float
        )
        psis = np.maximum(psis, 1e-12)
        log_psis = np.log(psis)
        try:
            slope = float(np.polyfit(times, log_psis, 1)[0])
        except (np.linalg.LinAlgError, ValueError) as exc:
            raise ValidationError(
                "lyapunov regression failed",
                code="carl.substrate.smoothing_failed",
                cause=exc,
            ) from exc
        return slope

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-safe dict."""
        return {
            "timestamp": self.timestamp,
            "overall_psi": self.overall_psi,
            "channels": {
                name: {
                    "name": ch.name,
                    "health": ch.health,
                    "drift_rate": ch.drift_rate,
                    "drift_acceleration": ch.drift_acceleration,
                    "leading_indicator": ch.leading_indicator,
                }
                for name, ch in self.channels.items()
            },
        }


__all__ = ["SubstrateChannel", "SubstrateState"]
