"""Shadow environment manager.

Manages the full lifecycle: fork -> replay traffic -> gate -> promote/reject.

The shadow is a weight fork (LoRA copy), not an infrastructure duplicate.
No extra GPU needed -- replay runs sequentially. Inspired by Epic/Cerner
healthcare IT shadow environments that ran 15-30 min behind production
with live data.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from carl_studio.shadow.traffic_log import TrafficEntry, TrafficLogger
from carl_studio.shadow.weight_fork import ShadowCycle, WeightForker


@dataclass
class GateResult:
    """Result of coherence gate evaluation on shadow adapter."""

    passed: bool
    metrics: dict
    reason: str


class ShadowManager:
    """Manages the full shadow environment lifecycle.

    Lifecycle: fork -> replay traffic -> gate -> promote or reject.

    The shadow is a weight fork (LoRA copy), not an infrastructure
    duplicate. No extra GPU needed -- replay runs sequentially.
    """

    def __init__(
        self,
        prod_adapter_path: str,
        shadow_dir: str = ".carl/shadow",
        traffic_db: str = ".carl/traffic.db",
    ):
        if not prod_adapter_path:
            raise ValueError("prod_adapter_path must not be empty")
        self.prod_adapter_path = prod_adapter_path
        self.forker = WeightForker(shadow_dir=shadow_dir)
        self.traffic = TrafficLogger(db_path=traffic_db)
        self._current_cycle: ShadowCycle | None = None

    def start_cycle(self) -> ShadowCycle:
        """Start a new shadow cycle by forking production weights.

        Returns:
            A new ShadowCycle with PENDING status.
        """
        cycle = self.forker.fork(self.prod_adapter_path)
        self._current_cycle = cycle
        return cycle

    def get_replay_traffic(self, window_minutes: int = 30) -> list[TrafficEntry]:
        """Get traffic from the last N minutes for replay.

        Args:
            window_minutes: How many minutes of traffic to replay.

        Returns:
            List of TrafficEntry objects in chronological order.
        """
        if window_minutes <= 0:
            return []
        end = datetime.now(timezone.utc)
        start = end - timedelta(minutes=window_minutes)
        return self.traffic.window(start, end)

    def gate(
        self,
        cycle: ShadowCycle,
        task_completion: float,
        format_compliance: float,
        phi_mean: float,
    ) -> GateResult:
        """Run coherence gate on shadow cycle metrics.

        Gate criteria (from CARL gate config):
        - task_completion >= 0.80
        - format_compliance >= 0.95
        - phi_mean > 0 (model is not degenerate)

        Args:
            cycle: The shadow cycle to evaluate.
            task_completion: Fraction of tasks completed successfully.
            format_compliance: Fraction of outputs with valid format.
            phi_mean: Mean coherence measure (must be positive).

        Returns:
            GateResult with pass/fail, metrics, and reason string.
        """
        passed = (
            task_completion >= 0.80
            and format_compliance >= 0.95
            and phi_mean > 0
        )

        metrics = {
            "task_completion": task_completion,
            "format_compliance": format_compliance,
            "phi_mean": phi_mean,
        }

        reasons = []
        if task_completion < 0.80:
            reasons.append(f"task_completion {task_completion:.2f} < 0.80")
        if format_compliance < 0.95:
            reasons.append(f"format_compliance {format_compliance:.2f} < 0.95")
        if phi_mean <= 0:
            reasons.append(f"phi_mean {phi_mean:.4f} <= 0")

        reason = "PASS" if passed else "FAIL: " + "; ".join(reasons)

        result = GateResult(passed=passed, metrics=metrics, reason=reason)
        cycle.gate_passed = passed
        cycle.metrics = metrics

        return result

    def promote(self, cycle: ShadowCycle) -> bool:
        """Promote shadow to production if gate passed.

        Args:
            cycle: The shadow cycle to promote.

        Returns:
            True if promotion succeeded.
        """
        return self.forker.promote(cycle)

    def reject(self, cycle: ShadowCycle) -> None:
        """Reject and clean up shadow cycle.

        Args:
            cycle: The shadow cycle to reject.
        """
        self.forker.reject(cycle)

    @property
    def current_cycle(self) -> ShadowCycle | None:
        """The currently active shadow cycle, or None."""
        return self._current_cycle

    def close(self) -> None:
        """Release resources."""
        self.traffic.close()
