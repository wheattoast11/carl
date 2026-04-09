"""CARL freemium tier system.

Three tiers: FREE, PRO, ENTERPRISE.

Tier philosophy:
  FREE  — Everything a researcher needs: observe, train, eval, BYOK compute.
          The full CARL loop works without paying. This is how we win adoption.
  PRO   — Autonomous integrations: Claude-powered diagnosis, auto-gated
          send-it pipeline, live TUI dashboards, managed eval scheduling.
          These are the features that make CARL a superagent.
  ENTERPRISE — Multi-tenant orchestration, MCP server for agent integration,
               custom environments, RBAC, audit trail, SLA guarantees.

Auto-elevation: if a user has the right credentials, they get the capabilities
without hitting a paywall. Having ANTHROPIC_API_KEY implies Pro (enables
Claude-powered features). CARL_ENTERPRISE=1 + credentials → Enterprise.
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
    PRO = "pro"
    ENTERPRISE = "enterprise"

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
    Tier.PRO: 1,
    Tier.ENTERPRISE: 2,
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
    # PRO — Autonomous superagent features. Claude-powered diagnosis,
    # auto-gated pipelines, live dashboards, managed orchestration.
    # These are what make CARL more than a training script.
    # ---------------------------------------------------------------
    "observe.live": Tier.PRO,           # Real-time Textual TUI dashboard
    "observe.diagnose": Tier.PRO,       # Claude-powered crystal analysis
    "train.send_it": Tier.PRO,          # Autonomous SFT→eval→GRPO→eval→push pipeline
    "train.auto_gate": Tier.PRO,        # Automatic eval gating between stages
    "train.auto_cascade": Tier.PRO,     # Autonomous cascade stage transitions
    "train.scheduled": Tier.PRO,        # Scheduled recurring training runs
    "bench.cti_report": Tier.PRO,       # Full CTI (CARL Trainability Index) report
    "bench.probes": Tier.PRO,           # Advanced probes (pressure, adaptation)
    "observe.claude_stream": Tier.PRO,  # Streaming Claude observations during training
    "eval.auto_schedule": Tier.PRO,     # Automatic eval scheduling on checkpoints
    # ---------------------------------------------------------------
    # ENTERPRISE — Multi-tenant, agent integration, custom envs, SLA.
    # ---------------------------------------------------------------
    "mcp": Tier.ENTERPRISE,
    "mcp.serve": Tier.ENTERPRISE,
    "environments.custom": Tier.ENTERPRISE,
    "compute.multi_backend": Tier.ENTERPRISE,
    "orchestration": Tier.ENTERPRISE,
    "orchestration.multi_run": Tier.ENTERPRISE,
    "train.pipeline.multi": Tier.ENTERPRISE,  # Multi-model pipeline orchestration
    "rbac": Tier.ENTERPRISE,
    "audit_trail": Tier.ENTERPRISE,
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
    """Elevate tier based on available credentials.

    Rules:
    - ANTHROPIC_API_KEY present -> at least PRO
    - HF_TOKEN present -> at least PRO (needed for training)
    - Both + CARL_ENTERPRISE=1 -> ENTERPRISE

    Never downgrades the configured tier.
    """
    effective = configured_tier

    has_anthropic = bool(os.environ.get("ANTHROPIC_API_KEY"))
    has_hf = bool(_detect_hf_token())
    has_enterprise_flag = os.environ.get("CARL_ENTERPRISE", "").lower() in ("1", "true", "yes")

    # Auto-elevate to PRO if user has training-capable credentials
    if has_anthropic or has_hf:
        if effective < Tier.PRO:
            effective = Tier.PRO

    # Enterprise requires explicit opt-in plus credentials
    if has_enterprise_flag and has_anthropic and has_hf:
        effective = Tier.ENTERPRISE

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
    Tier.PRO: "https://terminals.tech/pricing",
    Tier.ENTERPRISE: "https://terminals.tech/enterprise",
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

            if not tier_allows(effective, feat_name):
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
    return (
        f"This feature requires CARL {required.value.title()}. "
        f"Upgrade at {url}"
    )
