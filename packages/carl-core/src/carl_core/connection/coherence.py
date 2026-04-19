"""Cross-channel coherence observable.

ChannelCoherence is a small numeric summary that any channel (training /
env / MCP / A2A / x402) can publish per-transaction. It is the shared
observable that makes the Connection primitive's isomorphism claim
falsifiable: if channel A and channel B are structurally analogous,
their ChannelCoherence fields should track each other up to a stated map.
``channel_coherence_diff`` computes the pointwise distance so tests /
monitors can detect divergence.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class ChannelCoherence:
    """Per-transaction coherence summary for a Connection.

    - phi_mean:      field-strength proxy (0..1; higher = more structured)
    - cloud_quality: dispersion-quality proxy (0..1; higher = smoother)
    - success_rate:  rolling fraction of successful transactions (0..1)
    - latency_ms:    median latency of recent transactions (milliseconds)
    """

    phi_mean: float = 0.0
    cloud_quality: float = 0.0
    success_rate: float = 0.0
    latency_ms: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "phi_mean": self.phi_mean,
            "cloud_quality": self.cloud_quality,
            "success_rate": self.success_rate,
            "latency_ms": self.latency_ms,
        }

    @classmethod
    def empty(cls) -> ChannelCoherence:
        return cls()

    @classmethod
    def from_mapping(cls, data: dict[str, float]) -> ChannelCoherence:
        return cls(
            phi_mean=float(data.get("phi_mean", 0.0)),
            cloud_quality=float(data.get("cloud_quality", 0.0)),
            success_rate=float(data.get("success_rate", 0.0)),
            latency_ms=float(data.get("latency_ms", 0.0)),
        )


def channel_coherence_diff(
    a: ChannelCoherence, b: ChannelCoherence
) -> dict[str, float]:
    """Pointwise absolute difference between two ChannelCoherence observations.

    Returns a dict of the four field deltas. Non-finite values contribute
    ``float('inf')`` so downstream alarms fire rather than silently skipping.
    """
    out: dict[str, float] = {}
    for field_name in ("phi_mean", "cloud_quality", "success_rate", "latency_ms"):
        av = getattr(a, field_name)
        bv = getattr(b, field_name)
        if not (isfinite(av) and isfinite(bv)):
            out[field_name] = float("inf")
        else:
            out[field_name] = abs(av - bv)
    return out


def channel_coherence_distance(
    a: ChannelCoherence, b: ChannelCoherence
) -> float:
    """Euclidean distance on (phi_mean, cloud_quality, success_rate).

    ``latency_ms`` is excluded because it lives on a different scale (ms,
    unbounded) from the 0..1 coherence proxies. Monitors that care about
    latency divergence should read ``channel_coherence_diff`` directly.
    """
    diffs = channel_coherence_diff(a, b)
    s = (
        diffs["phi_mean"] ** 2
        + diffs["cloud_quality"] ** 2
        + diffs["success_rate"] ** 2
    )
    return s ** 0.5


__all__ = [
    "ChannelCoherence",
    "channel_coherence_diff",
    "channel_coherence_distance",
]
