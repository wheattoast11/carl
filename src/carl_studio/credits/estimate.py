"""Credit cost estimation for training jobs.

1 credit = 1 A100-minute at standard rate.
Hardware multipliers scale from 0.3x (L4) to 2.5x (H200).
"""

from __future__ import annotations

import math
from typing import Final

from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Credit rates: credits per GPU-minute for each hardware flavor
# ---------------------------------------------------------------------------

CREDIT_RATES: Final[dict[str, float]] = {
    "l4x1": 0.3,
    "l4x4": 1.2,
    "l40sx1": 0.7,
    "l40sx4": 2.8,
    "l40sx8": 5.6,
    "a10g-large": 0.8,
    "a10g-largex2": 1.6,
    "a10g-largex4": 3.2,
    "a100-large": 1.0,
    "a100-largex2": 2.0,
    "a100-largex4": 4.0,
    "a100-largex8": 8.0,
    "h200": 2.5,
    "local": 0.0,
}

# ---------------------------------------------------------------------------
# Bundle pricing for Stripe checkout
# ---------------------------------------------------------------------------

BUNDLES: Final[dict[str, dict[str, int | float]]] = {
    "starter": {"credits": 100, "price_usd": 8},
    "explorer": {"credits": 500, "price_usd": 35},
    "researcher": {"credits": 2000, "price_usd": 120},
}

# ---------------------------------------------------------------------------
# Per-step time estimates (seconds) by training method
# ---------------------------------------------------------------------------

METHOD_STEP_SECONDS: Final[dict[str, float]] = {
    "sft": 5.0,
    "grpo-text": 15.0,
    "grpo-env": 20.0,
    "grpo-vision": 25.0,
}

# ---------------------------------------------------------------------------
# Included monthly credits by subscription plan
# ---------------------------------------------------------------------------

INCLUDED_CREDITS: Final[dict[str, int]] = {
    "monthly": 200,
    "annual": 300,
}

# Reference price per credit (for USD estimates)
_CREDIT_USD: Final[float] = 0.08  # Weighted avg across bundles


class CreditEstimate(BaseModel):
    """Estimated credit cost for a job."""

    hardware: str
    rate_per_min: float
    estimated_minutes: float
    estimated_credits: int
    """ceil(rate * minutes) -- raw cost without buffer."""
    estimated_usd: float
    """Reference USD cost at weighted-average bundle price."""
    buffer_pct: float = 0.20
    """Safety buffer percentage on estimate (default 20%)."""
    total_with_buffer: int
    """What gets pre-deducted: ceil(estimated_credits * (1 + buffer_pct))."""


def estimate_job_cost(
    hardware: str,
    max_steps: int,
    per_step_seconds: float = 20.0,
) -> CreditEstimate:
    """Estimate credit cost for a training job.

    Args:
        hardware: Compute target name (e.g. "a100-large").
        max_steps: Maximum training steps.
        per_step_seconds: Average seconds per step. Defaults to 20.0 (env GRPO on A100).
            Use METHOD_STEP_SECONDS for method-specific defaults.

    Returns:
        CreditEstimate with breakdown and buffered total.
    """
    if max_steps <= 0:
        raise ValueError(f"max_steps must be positive, got {max_steps}")
    if per_step_seconds <= 0:
        raise ValueError(f"per_step_seconds must be positive, got {per_step_seconds}")

    rate = CREDIT_RATES.get(hardware, 1.0)
    minutes = (max_steps * per_step_seconds) / 60.0
    raw_credits = math.ceil(rate * minutes)
    buffer = 0.20
    total = math.ceil(raw_credits * (1 + buffer))

    return CreditEstimate(
        hardware=hardware,
        rate_per_min=rate,
        estimated_minutes=round(minutes, 1),
        estimated_credits=raw_credits,
        estimated_usd=round(raw_credits * _CREDIT_USD, 2),
        buffer_pct=buffer,
        total_with_buffer=total,
    )


def best_bundle(credits_needed: int) -> str | None:
    """Recommend the cheapest bundle that covers the credit need.

    Returns bundle name or None if no single bundle covers it.
    """
    for name in ("starter", "explorer", "researcher"):
        bundle = BUNDLES[name]
        if int(bundle["credits"]) >= credits_needed:
            return name
    return None
