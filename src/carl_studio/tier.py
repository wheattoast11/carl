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

from carl_studio.gating import GATE_TIER_INSUFFICIENT, emit_gate_event

if TYPE_CHECKING:
    from collections.abc import Callable

    from carl_core.interaction import InteractionChain


__all__ = [
    # Re-exported primitives (carl_core.tier)
    "Tier",
    "FEATURE_TIERS",
    "TierGateError",
    "feature_tier",
    "tier_allows",
    # carl-studio-specific helpers
    "TierPredicate",
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


class TierPredicate:
    """A :class:`~carl_studio.gating.GatingPredicate` for a tier requirement.

    Wraps the tier-comparison + feature-registry check used by
    :func:`tier_gate` and :func:`check_tier` in the shared gate-predicate
    shape. Resolves the effective tier from
    :class:`~carl_studio.settings.CARLSettings` lazily inside
    :meth:`check` so the predicate itself stays cheap to construct.
    """

    __slots__ = ("_required", "_feature", "_effective_override")

    def __init__(
        self,
        required_tier: Tier,
        feature: str | None = None,
        *,
        effective: Tier | None = None,
    ) -> None:
        self._required = required_tier
        self._feature = feature
        # Test hook: when supplied, skip the CARLSettings round-trip.
        self._effective_override = effective

    @property
    def name(self) -> str:
        return f"tier:{self._feature or self._required.value}"

    @property
    def required(self) -> Tier:
        """Minimum tier required by this predicate."""
        return self._required

    @property
    def feature(self) -> str | None:
        """Feature name (for registry lookup), if any."""
        return self._feature

    def _effective(self) -> Tier:
        if self._effective_override is not None:
            return self._effective_override
        # Lazy import: CARLSettings is heavy and we don't want to pull it
        # into ``import carl_studio.tier`` at module load time.
        from carl_studio.settings import CARLSettings

        return detect_effective_tier(CARLSettings.load().tier)

    def check(self) -> tuple[bool, str]:
        """Return ``(allowed, reason)`` for the tier requirement."""
        effective = self._effective()
        feat_name = self._feature or ""
        feature_ok = not feat_name or tier_allows(effective, feat_name)
        rank_ok = effective >= self._required
        if rank_ok and feature_ok:
            return True, f"tier '{effective.value}' >= required '{self._required.value}'"
        url = _UPGRADE_URLS.get(self._required, "https://terminals.tech/pricing")
        return (
            False,
            f"'{feat_name or self._required.value}' requires CARL "
            f"{self._required.value.title()}. Current tier: "
            f"{effective.value.title()}. Upgrade at {url}",
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

    On denial the raised :class:`TierGateError` additionally carries a
    ``context`` attribute with a ``gate_code`` entry
    (``carl.gate.tier_insufficient``) so operators can filter any gate
    denial — consent or tier — on the shared ``carl.gate.*`` namespace
    without collapsing the exception taxonomy. The legacy attributes
    (``feature`` / ``required`` / ``current``) remain unchanged.
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            from carl_studio.settings import CARLSettings

            settings = CARLSettings.load()
            effective = detect_effective_tier(settings.tier)
            feat_name = feature or func.__name__

            predicate = TierPredicate(
                required_tier, feat_name, effective=effective
            )
            # ``_gate_chain`` can be threaded in via kwargs for structured
            # logging without changing the wrapped function's signature.
            # Pop so we don't leak it into ``func`` (which would TypeError
            # on an unexpected kwarg).
            chain: InteractionChain | None = kwargs.pop("_gate_chain", None)
            allowed, reason = predicate.check()
            emit_gate_event(
                predicate_name=predicate.name,
                allowed=allowed,
                reason=reason,
                chain=chain,
            )

            if not allowed:
                err = TierGateError(feat_name, required_tier, effective)
                _attach_gate_context(
                    err,
                    feature=feat_name,
                    required=required_tier,
                    current=effective,
                )
                raise err

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
    # Route through the shared predicate so the tier-check accounting is
    # identical across the decorator and inline-check paths. We don't
    # raise from here (check_tier is for friendly UI), so no chain is
    # wired — callers that want a gate event should use ``tier_gate``.
    predicate = TierPredicate(required, feature, effective=effective)
    allowed, _reason = predicate.check()
    return allowed, effective, required


def _attach_gate_context(
    err: TierGateError,
    *,
    feature: str,
    required: Tier,
    current: Tier,
) -> None:
    """Attach a ``context`` dict (with ``gate_code``) to a TierGateError.

    :class:`TierGateError` lives in ``carl_core.tier`` as a plain
    ``Exception`` for primitive-layer portability; we layer the shared
    ``carl.gate.*`` attribution at the carl-studio boundary instead of
    mutating the primitive.
    """
    err.context = {  # pyright: ignore[reportAttributeAccessIssue]
        "feature": feature,
        "required": required.value,
        "current": current.value,
        "gate_code": GATE_TIER_INSUFFICIENT,
    }


def tier_message(feature: str) -> str | None:
    """Get an upgrade message if the feature is gated, or None if allowed.

    Friendly message for CLI output -- not an exception.
    """
    allowed, effective, required = check_tier(feature)
    if allowed:
        return None
    url = _UPGRADE_URLS.get(required, "https://terminals.tech/pricing")
    return f"This feature requires CARL {required.value.title()}. Upgrade at {url}"
