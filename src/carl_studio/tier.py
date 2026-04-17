"""CARL freemium tier system — carl-studio facade.

The pure primitives (:class:`Tier`, :data:`FEATURE_TIERS`,
:class:`TierGateError`, :func:`feature_tier`, :func:`tier_allows`) live in
:mod:`carl_core.tier` so downstream packages can depend on them without
pulling carl-studio.

This module keeps the carl-studio-specific helpers that need settings / db /
HF auth: :func:`detect_effective_tier`, :func:`tier_gate`,
:func:`check_tier`, :func:`tier_message`, and :func:`_detect_hf_token`.

Subscription check priority:
  1. SQLite auth cache (~/.carl/carl.db) — sub-ms, offline-safe
  2. Fresh carl.camp subscription truth, when explicitly refreshed by the user
  3. Configured preference from settings
  4. Default to FREE
"""

from __future__ import annotations

import functools
import os
from typing import TYPE_CHECKING

from carl_core.tier import (
    FEATURE_TIERS,
    Tier,
    TierGateError,
    feature_tier,
    tier_allows,
)
from carl_core.tier import _TIER_RANK as _TIER_RANK  # pyright: ignore[reportPrivateUsage]
from carl_core.tier import _UPGRADE_URLS  # pyright: ignore[reportPrivateUsage]

if TYPE_CHECKING:
    from collections.abc import Callable


__all__ = [
    # Re-exported primitives (carl_core.tier)
    "Tier",
    "FEATURE_TIERS",
    "TierGateError",
    "feature_tier",
    "tier_allows",
    # carl-studio-specific helpers
    "detect_effective_tier",
    "tier_gate",
    "check_tier",
    "tier_message",
]


# ---------------------------------------------------------------------------
# Effective tier: subscription truth beats local preference
# ---------------------------------------------------------------------------


def detect_effective_tier(configured_tier: Tier) -> Tier:
    """Resolve the current tier from cached subscription truth and local preference.

    Local provider credentials unlock provider integrations, not managed CARL Paid
    platform surfaces. Paid access comes from a cached or freshly refreshed
    carl.camp account state.

    Never downgrades an explicitly configured paid tier.
    """
    try:
        from carl_studio.db import LocalDB

        db = LocalDB()
        cached_tier = db.get_auth("tier")
        if cached_tier == Tier.PAID.value:
            return Tier.PAID
    except Exception:
        pass

    return configured_tier


def _detect_hf_token() -> str | None:
    """Detect HF token from huggingface_hub credentials, then env fallback."""
    try:
        from huggingface_hub import get_token

        token = get_token()
        if token:
            return token
    except Exception:
        pass
    return os.environ.get("HF_TOKEN")


# ---------------------------------------------------------------------------
# Tier gate decorator
# ---------------------------------------------------------------------------


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
