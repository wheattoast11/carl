"""CARL freemium tier system.

Two tiers: FREE and PAID.

Tier philosophy:
  FREE  — Full CARL loop: observe, train, eval, bench, align, learn, push,
          bundle. BYOK compute. SQLite persistence. All public CARL rewards.
          Gate on autonomy, not capability. Users train for free.
  PAID  — The discovery engine: autonomous pipeline (--send-it), auto-gating,
          scheduled runs, resonance rewards, experiment management, carl.camp
          dashboard, cloud sync, MCP server, multi-tenant/RBAC.

Subscription check priority:
  1. SQLite auth cache (~/.carl/carl.db) — sub-ms, offline-safe
  2. Supabase RPC (carl.camp) — ~100ms, if cache expired
  3. Auto-elevation from local credentials — ANTHROPIC_API_KEY or HF auth implies PAID
  4. 48h grace period if network down
  5. Default to FREE
"""

from __future__ import annotations

import functools
import os
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable


class Tier(str, Enum):
    """CARL pricing tier."""

    FREE = "free"
    PAID = "paid"

    # Backwards compatibility aliases
    PRO = "paid"
    ENTERPRISE = "paid"

    def __ge__(self, other: object) -> bool:
        if not isinstance(other, Tier):
            return NotImplemented
        return _TIER_RANK[self] >= _TIER_RANK[other]

    def __gt__(self, other: object) -> bool:
        if not isinstance(other, Tier):
            return NotImplemented
        return _TIER_RANK[self] > _TIER_RANK[other]

    def __le__(self, other: object) -> bool:
        if not isinstance(other, Tier):
            return NotImplemented
        return _TIER_RANK[self] <= _TIER_RANK[other]

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Tier):
            return NotImplemented
        return _TIER_RANK[self] < _TIER_RANK[other]


_TIER_RANK: dict[Tier, int] = {
    Tier.FREE: 0,
    Tier.PAID: 1,
}


# ---------------------------------------------------------------------------
# Feature registry
# ---------------------------------------------------------------------------

# Maps feature name -> minimum tier required
#
# Philosophy: the FULL CARL loop is free. What costs money are the
# autonomous integrations that turn CARL into a superagent.
FEATURE_TIERS: dict[str, Tier] = {
    # ---------------------------------------------------------------
    # FREE — Full researcher toolkit. Train, observe, eval, BYOK.
    # This is how we win adoption. No gates on core functionality.
    # ---------------------------------------------------------------
    "observe": Tier.FREE,
    "observe.basic": Tier.FREE,
    "observe.live": Tier.FREE,  # Real-time Textual TUI dashboard
    "observe.diagnose": Tier.FREE,  # BYOK Claude-powered crystal analysis
    "eval": Tier.FREE,
    "eval.phase1": Tier.FREE,
    "eval.phase2": Tier.FREE,
    "eval.phase2prime": Tier.FREE,
    "eval.remote": Tier.FREE,
    "train": Tier.FREE,
    "train.grpo": Tier.FREE,
    "train.sft": Tier.FREE,
    "bench": Tier.FREE,
    "bench.basic": Tier.FREE,
    "project": Tier.FREE,
    "setup": Tier.FREE,
    "config": Tier.FREE,
    "status": Tier.FREE,
    "logs": Tier.FREE,
    "stop": Tier.FREE,
    "push": Tier.FREE,
    "bundle": Tier.FREE,
    "align": Tier.FREE,
    "learn": Tier.FREE,
    "checkpoint": Tier.FREE,
    "compute": Tier.FREE,
    # ---------------------------------------------------------------
    # PAID ($29/mo) — Autonomy + Discovery + Resonance + Platform.
    # Everything that runs without human in the loop.
    # ---------------------------------------------------------------
    "train.send_it": Tier.PAID,  # Autonomous SFT→eval→GRPO→eval→push pipeline
    "train.auto_gate": Tier.PAID,  # Automatic eval gating between stages
    "train.auto_cascade": Tier.PAID,  # Autonomous cascade stage transitions
    "train.scheduled": Tier.PAID,  # Scheduled recurring training runs
    "bench.cti_report": Tier.PAID,  # Full CTI (CARL Trainability Index) report
    "bench.probes": Tier.PAID,  # Advanced probes (pressure, adaptation)
    "observe.claude_stream": Tier.PAID,  # Streaming Claude observations during training
    "eval.auto_schedule": Tier.PAID,  # Automatic eval scheduling on checkpoints
    "mcp": Tier.PAID,
    "mcp.serve": Tier.PAID,
    "environments.custom": Tier.PAID,
    "compute.multi_backend": Tier.PAID,
    "orchestration": Tier.PAID,
    "orchestration.multi_run": Tier.PAID,
    "train.pipeline.multi": Tier.PAID,  # Multi-model pipeline orchestration
    "rbac": Tier.PAID,
    "audit_trail": Tier.PAID,
    "experiment": Tier.PAID,  # Discovery engine
    "experiment.auto_judge": Tier.PAID,  # Automated experiment judgment
    "sync.cloud": Tier.PAID,  # carl.camp cloud sync
    "dashboard": Tier.PAID,  # carl.camp web dashboard
    "marketplace.publish": Tier.PAID,  # Publish to carl.camp marketplace
}


def feature_tier(feature: str) -> Tier:
    """Get the minimum tier required for a feature.

    Falls back to FREE for unknown features (permissive by default).
    """
    return FEATURE_TIERS.get(feature, Tier.FREE)


def tier_allows(tier: Tier, feature: str) -> bool:
    """Check if a tier can access a feature."""
    required = feature_tier(feature)
    return tier >= required


# ---------------------------------------------------------------------------
# Auto-elevation: infer tier from available credentials
# ---------------------------------------------------------------------------


def detect_effective_tier(configured_tier: Tier) -> Tier:
    """Elevate tier based on subscription status or credentials.

    Priority:
    1. Supabase subscription (cached in SQLite, checked via carl.camp)
    2. Auto-elevation from env vars (ANTHROPIC_API_KEY → PAID)
    3. Configured tier from settings

    Never downgrades the configured tier.
    """
    effective = configured_tier

    # Check Supabase subscription via local cache
    try:
        from carl_studio.db import LocalDB

        db = LocalDB()
        cached_tier = db.get_auth("tier")
        if cached_tier == "paid":
            return Tier.PAID
    except Exception:
        pass

    # Auto-elevate if user has local credentials for advanced workflows.
    has_anthropic = bool(os.environ.get("ANTHROPIC_API_KEY"))
    has_hf = bool(_detect_hf_token())
    if (has_anthropic or has_hf) and effective < Tier.PAID:
        effective = Tier.PAID

    return effective


def _detect_hf_token() -> str | None:
    """Detect HF token from env or huggingface_hub credentials."""
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    try:
        from huggingface_hub import get_token

        return get_token()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Tier gate decorator
# ---------------------------------------------------------------------------

_UPGRADE_URLS: dict[Tier, str] = {
    Tier.PAID: "https://carl.camp/pricing",
}


class TierGateError(Exception):
    """Raised when a feature requires a higher tier."""

    def __init__(self, feature: str, required: Tier, current: Tier) -> None:
        self.feature = feature
        self.required = required
        self.current = current
        url = _UPGRADE_URLS.get(required, "https://terminals.tech/pricing")
        super().__init__(
            f"'{feature}' requires CARL {required.value.title()}. "
            f"Current tier: {current.value.title()}. "
            f"Upgrade at {url}"
        )


def tier_gate(required_tier: Tier, feature: str | None = None) -> Callable:
    """Decorator that gates a function behind a tier requirement.

    Usage::

        @tier_gate(Tier.PRO)
        def observe_live():
            ...

        @tier_gate(Tier.ENTERPRISE, feature="mcp.serve")
        def mcp_serve():
            ...
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            from carl_studio.settings import CARLSettings

            settings = CARLSettings.load()
            effective = detect_effective_tier(settings.tier)
            feat_name = feature or func.__name__

            # Check both: the explicit required_tier AND the feature registry.
            # The decorator's required_tier is the authoritative minimum;
            # the feature registry may also impose a minimum via tier_allows.
            if effective < required_tier or not tier_allows(effective, feat_name):
                raise TierGateError(feat_name, required_tier, effective)

            return func(*args, **kwargs)

        # Preserve metadata for Typer introspection
        wrapper.__tier_required__ = required_tier
        wrapper.__tier_feature__ = feature
        return wrapper

    return decorator


def check_tier(feature: str) -> tuple[bool, Tier, Tier]:
    """Check if current effective tier allows a feature.

    Returns (allowed, effective_tier, required_tier).
    Use this for inline checks where the decorator pattern doesn't fit.
    """
    from carl_studio.settings import CARLSettings

    settings = CARLSettings.load()
    effective = detect_effective_tier(settings.tier)
    required = feature_tier(feature)
    return tier_allows(effective, feature), effective, required


def tier_message(feature: str) -> str | None:
    """Get an upgrade message if the feature is gated, or None if allowed.

    Friendly message for CLI output -- not an exception.
    """
    allowed, effective, required = check_tier(feature)
    if allowed:
        return None
    url = _UPGRADE_URLS.get(required, "https://terminals.tech/pricing")
    return f"This feature requires CARL {required.value.title()}. Upgrade at {url}"
