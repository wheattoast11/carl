"""CARL pricing tier primitives.

Pure, dependency-free tier model shared across the CARL stack. Contains:

* :class:`Tier` — the ordered FREE / PAID enum (with PRO, ENTERPRISE aliases).
* :data:`FEATURE_TIERS` — feature name -> minimum required tier.
* :func:`feature_tier` / :func:`tier_allows` — pure lookup helpers.
* :class:`TierGateError` — raised when a feature is gated above the current tier.

The effective-tier resolution, settings-backed gate decorator, and HF auth
helpers live in ``carl_studio.tier`` because they depend on the carl-studio
settings / db modules. Keeping the primitives here lets downstream packages
(carl-agent, carl-training, carl-marketplace, ...) consume the tier model
without pulling carl-studio as a dependency.

Tier philosophy
---------------

FREE  — Full CARL loop: observe, train, eval, bench, align, learn, push,
        bundle. BYOK compute. SQLite persistence. All public CARL rewards.
        Gate on autonomy, not capability. Users train for free.
PAID  — The discovery engine: autonomous pipeline (--send-it), auto-gating,
        scheduled runs, resonance rewards, experiment management, carl.camp
        dashboard, cloud sync, MCP server, multi-tenant/RBAC.
"""

from __future__ import annotations

from enum import Enum

__all__ = [
    "Tier",
    "FEATURE_TIERS",
    "TierGateError",
    "feature_tier",
    "tier_allows",
]


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
# Gate error
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
