"""Weight forking for shadow LoRA adapters.

Production serves with LoRA_prod. The shadow creates LoRA_shadow (a copy),
trains independently via SLOT/TTT micro-updates on replayed traffic, then
promotes to prod when the coherence gate passes.
"""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path


class PromotionStatus(Enum):
    PENDING = "pending"
    PROMOTED = "promoted"
    REJECTED = "rejected"
    EXPIRED = "expired"


@dataclass
class ShadowCycle:
    """One shadow training cycle."""

    cycle_id: int
    shadow_adapter: str  # path to LoRA_shadow directory
    prod_adapter: str  # path to LoRA_prod directory
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None
    traffic_start: datetime | None = None
    traffic_end: datetime | None = None
    gate_passed: bool | None = None
    promotion_status: PromotionStatus = PromotionStatus.PENDING
    metrics: dict = field(default_factory=dict)


class WeightForker:
    """Creates and manages LoRA weight forks for shadow training.

    The shadow is a COPY of LoRA_prod that trains independently
    on replayed traffic. Promotion = atomic directory swap.
    """

    def __init__(self, shadow_dir: str = ".carl/shadow"):
        self.shadow_dir = Path(shadow_dir)
        self.shadow_dir.mkdir(parents=True, exist_ok=True)
        self._cycle_counter = 0

    def fork(self, prod_adapter_path: str) -> ShadowCycle:
        """Create LoRA_shadow as a copy of LoRA_prod.

        Args:
            prod_adapter_path: Path to the production LoRA adapter directory.

        Returns:
            ShadowCycle with paths set, status PENDING.

        Raises:
            ValueError: If prod_adapter_path is empty.
        """
        if not prod_adapter_path:
            raise ValueError("prod_adapter_path must not be empty")

        self._cycle_counter += 1
        shadow_path = str(self.shadow_dir / f"shadow_cycle_{self._cycle_counter}")

        # Copy adapter files
        if os.path.isdir(prod_adapter_path):
            shutil.copytree(prod_adapter_path, shadow_path, dirs_exist_ok=True)
        else:
            # For testing or when adapter is a HF hub ID: create empty directory
            os.makedirs(shadow_path, exist_ok=True)

        return ShadowCycle(
            cycle_id=self._cycle_counter,
            shadow_adapter=shadow_path,
            prod_adapter=prod_adapter_path,
        )

    def promote(self, cycle: ShadowCycle) -> bool:
        """Promote shadow adapter to production.

        Atomic swap: backup prod, move shadow -> prod.
        Rolls back on failure.

        Args:
            cycle: The shadow cycle to promote.

        Returns:
            True if promotion succeeded, False otherwise.
        """
        if cycle.promotion_status != PromotionStatus.PENDING:
            return False
        if not cycle.gate_passed:
            cycle.promotion_status = PromotionStatus.REJECTED
            return False

        prod_path = Path(cycle.prod_adapter)
        shadow_path = Path(cycle.shadow_adapter)
        backup_path = prod_path.parent / f"{prod_path.name}_backup_{cycle.cycle_id}"

        if not shadow_path.exists():
            cycle.promotion_status = PromotionStatus.REJECTED
            return False

        try:
            # Atomic-ish swap: backup prod, move shadow to prod
            if prod_path.exists():
                shutil.move(str(prod_path), str(backup_path))
            shutil.move(str(shadow_path), str(prod_path))
            cycle.promotion_status = PromotionStatus.PROMOTED
            cycle.completed_at = datetime.now(timezone.utc)
            return True
        except OSError:
            # Restore backup on failure
            if backup_path.exists() and not prod_path.exists():
                shutil.move(str(backup_path), str(prod_path))
            cycle.promotion_status = PromotionStatus.REJECTED
            return False

    def reject(self, cycle: ShadowCycle) -> None:
        """Reject and clean up a shadow cycle.

        Args:
            cycle: The shadow cycle to reject.
        """
        cycle.promotion_status = PromotionStatus.REJECTED
        cycle.completed_at = datetime.now(timezone.utc)
        shadow_path = Path(cycle.shadow_adapter)
        if shadow_path.exists():
            shutil.rmtree(shadow_path)
