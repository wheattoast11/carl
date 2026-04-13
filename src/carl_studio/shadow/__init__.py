"""Shadow environment — weight-fork training with coherence-gated promotion."""
from __future__ import annotations

from shadow.weight_fork import WeightForker, ShadowCycle, PromotionStatus
from shadow.traffic_log import TrafficLogger, TrafficEntry
from shadow.shadow_manager import ShadowManager, GateResult

__all__ = [
    "WeightForker",
    "ShadowCycle",
    "PromotionStatus",
    "TrafficLogger",
    "TrafficEntry",
    "ShadowManager",
    "GateResult",
]
