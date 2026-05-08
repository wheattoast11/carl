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

import contextlib
import functools
import os
from typing import TYPE_CHECKING, Any

from carl_core.errors import CARLError
from carl_core.tier import (
    FEATURE_TIERS,
    Tier,
    TierGateError,
    feature_tier,
    tier_allows,
)
from carl_core.tier import _TIER_RANK as _TIER_RANK  # pyright: ignore[reportPrivateUsage]
from carl_core.tier import _UPGRADE_URLS  # pyright: ignore[reportPrivateUsage]

from carl_studio.gating import GATE_TIER_INSUFFICIENT, BaseGate, emit_gate_event

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
    "TierPredicate",
    "detect_effective_tier",
    "tier_gate",
    "check_tier",
    "tier_message",
    # Private-runtime plug-point (v0.8 · S3)
    "register_tier_resolver",
    "clear_tier_resolver",
    "get_tier_resolver",
]


# ---------------------------------------------------------------------------
# Pluggable tier resolver (v0.8 · S3)
# ---------------------------------------------------------------------------
#
# Private runtimes sometimes need an alternate tier-source semantics
# (wallet-balance-based, JWT-embedded-claim, org-plan lookup). Rather than
# force a monkey-patch on ``detect_effective_tier``, we expose a single
# module-level resolver slot. When registered, the resolver is called
# from :meth:`TierPredicate._effective` with the predicate's feature name
# (or ``None`` when no feature was bound), and its return value is used as
# the effective tier.
#
# The registry is a process-local plug-point — intentionally not persisted
# to ``LocalDB``. Callers install it at process start (e.g., during
# private-runtime bootstrap) and never ship it across processes.

_TIER_RESOLVER: Callable[[str | None], Tier] | None = None


def register_tier_resolver(fn: Callable[[str | None], Tier]) -> None:
    """Register a custom tier resolver. Replaces the default detect_effective_tier() path."""
    global _TIER_RESOLVER  # noqa: PLW0603
    _TIER_RESOLVER = fn  # pyright: ignore[reportConstantRedefinition]


def clear_tier_resolver() -> None:
    """Restore the default tier resolver (detect_effective_tier)."""
    global _TIER_RESOLVER  # noqa: PLW0603
    _TIER_RESOLVER = None  # pyright: ignore[reportConstantRedefinition]


def get_tier_resolver() -> Callable[[str | None], Tier] | None:
    """Return the currently registered custom resolver, or None if default is active."""
    return _TIER_RESOLVER


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
        # Pluggable tier resolver (v0.8 · S3). When a private-runtime has
        # registered an alternate tier source (wallet-balance, JWT claim,
        # etc.), it takes precedence over the default settings round-trip.
        # Resolver errors are wrapped in a CARLError with a stable code so
        # operators can filter on ``carl.tier.resolver_error``.
        resolver = _TIER_RESOLVER
        if resolver is not None:
            try:
                return resolver(self._feature)
            except Exception as exc:
                raise CARLError(
                    "tier resolver raised",
                    code="carl.tier.resolver_error",
                    context={"inner": str(exc)},
                    cause=exc,
                ) from exc
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


def tier_gate(
    required_tier: Tier,
    feature: str | None = None,
    *,
    verify_remote: bool = False,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator that gates a function behind a tier requirement.

    Usage::

        @tier_gate(Tier.PAID)
        def observe_live():
            ...

        @tier_gate(Tier.PAID, feature="mcp.serve")
        def mcp_serve():
            ...

        @tier_gate(Tier.PAID, feature="train.slime.managed", verify_remote=True)
        def submit_managed_run():
            ...

    On denial the raised :class:`TierGateError` additionally carries a
    ``context`` attribute with a ``gate_code`` entry
    (``carl.gate.tier_insufficient``) so operators can filter any gate
    denial — consent or tier — on the shared ``carl.gate.*`` namespace
    without collapsing the exception taxonomy. The legacy attributes
    (``feature`` / ``required`` / ``current``) remain unchanged.

    Remote verification (v0.10)
    ---------------------------

    ``verify_remote=False`` (default) preserves byte-for-byte the
    pre-v0.10 local-only behaviour — the existing :class:`BaseGate`
    runs and that's the entire decision.

    ``verify_remote=True`` adds a local-fast-path-then-async-verify
    ladder (per CLAUDE.md AP-1):

    1. The existing local :class:`BaseGate` runs first (sub-ms,
       offline-safe). It is the dominant decision and determines the
       common path.
    2. If local **allows**, schedule
       :func:`carl_studio.entitlements.fetch_remote_async` as a
       fire-and-forget background verify. The future is intentionally
       discarded — the studio caller never blocks on the wire and never
       sees its exceptions. A subsequent call after the cache flips
       (e.g. tier downgrade on the server) will surface as a normal
       local-deny on the next gate invocation.
    3. If local **denies**, consult
       :meth:`carl_studio.entitlements.EntitlementsClient.cache_get` for
       a matching ``EntitlementGrant``. When the cache is within the
       24-hour offline-grace window AND contains a grant whose ``key``
       equals ``feature``, override the deny: emit a
       ``tier_remote_grace`` event and admit the call. Otherwise the
       original :class:`TierGateError` is raised unchanged.

    This shape preserves the doctrine that ``carl train --dry-run``
    never blocks on the network — remote verify is async; only the
    local-deny-but-cache-says-grant path adds latency, and even that
    path hits a local file (the cache), not the wire.
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        feat_name = feature or func.__name__

        # Bind a per-call predicate factory that resolves the effective
        # tier lazily each invocation (matching pre-v0.8 semantics —
        # upgrades take effect without re-importing). When a custom
        # resolver is registered (v0.8 · S3) we defer to it by leaving
        # ``effective`` unset — ``TierPredicate._effective`` will consult
        # the resolver each check. Otherwise we pre-resolve via the
        # legacy settings path so the predicate object stays pure.
        def _make_predicate() -> TierPredicate:
            if _TIER_RESOLVER is not None:
                return TierPredicate(required_tier, feat_name)
            from carl_studio.settings import CARLSettings

            settings = CARLSettings.load()
            effective = detect_effective_tier(settings.tier)
            return TierPredicate(
                required_tier, feat_name, effective=effective
            )

        tier_engine = BaseGate[TierPredicate].for_predicate(
            predicate_factory=_make_predicate,
            error_type=TierGateError,
            gate_code=GATE_TIER_INSUFFICIENT,
        )
        local_wrapped = tier_engine()(func)

        if not verify_remote:
            # Default path — no behavioural change from pre-v0.10.
            local_wrapped.__tier_required__ = required_tier  # type: ignore[attr-defined]
            local_wrapped.__tier_feature__ = feature  # type: ignore[attr-defined]
            local_wrapped.__tier_verify_remote__ = False  # type: ignore[attr-defined]
            return local_wrapped

        # Remote-verify path. The local gate still runs first; we wrap
        # ``local_wrapped`` and only tap the cache / fire async on
        # specific outcomes.
        @functools.wraps(func)
        def remote_wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                result = local_wrapped(*args, **kwargs)
            except TierGateError:
                # Local denied. Consult the cache for an override.
                cached_grant = _try_cached_grant(feat_name)
                if cached_grant is not None:
                    _emit_tier_remote_grace(feat_name, cached_grant)
                    return func(*args, **kwargs)
                # No cached grant → original deny stands.
                raise
            else:
                # Local allowed. Fire background verify; never block.
                _schedule_async_verify(feat_name)
                return result

        remote_wrapper.__tier_required__ = required_tier  # type: ignore[attr-defined]
        remote_wrapper.__tier_feature__ = feature  # type: ignore[attr-defined]
        remote_wrapper.__tier_verify_remote__ = True  # type: ignore[attr-defined]
        return remote_wrapper

    return decorator


# ---------------------------------------------------------------------------
# Remote-verify helpers (v0.10)
# ---------------------------------------------------------------------------
#
# These three helpers compose the remote-verify ladder. They are kept
# module-private (``_``-prefixed) but are designed to be patched in
# tests via the standard ``monkeypatch.setattr`` seam at
# ``carl_studio.tier._<helper>`` so test-local fakes don't have to
# rebuild the entire decorator wrapper.


def _schedule_async_verify(feature: str) -> None:
    """Fire :func:`fetch_remote_async` as background verification.

    Resolves the bearer JWT via the studio's standard chain
    (``CARL_CAMP_TOKEN`` env → ``~/.carl/camp_token`` → LocalDB jwt),
    then schedules a fire-and-forget fetch. Errors are swallowed: the
    fast-path caller must never block, and the future captures any
    network / signature exceptions for inspection by callers that opt
    into ``future.add_done_callback``.

    No-ops when the entitlements module is unimportable (e.g., when the
    ``[constitutional]`` extra isn't installed) or when no JWT is
    available — both are normal local-only configurations.
    """
    try:
        from carl_studio.entitlements import fetch_remote_async
    except ImportError:
        return

    jwt = _resolve_bearer_token_for_verify()
    if not jwt:
        return

    with contextlib.suppress(Exception):
        # Future is intentionally discarded — fire and forget. The
        # background executor in entitlements.py captures the future's
        # exception so it never propagates to the fast-path caller.
        fetch_remote_async(jwt)


def _try_cached_grant(feature: str) -> Any:
    """Return the cached ``Entitlements`` iff a matching grant exists in grace.

    Returns the full :class:`carl_studio.entitlements.Entitlements`
    object on a successful cache hit so the caller can stamp
    ``key_id`` / ``cached_at`` on the gate event. Returns ``None`` for
    every cache miss / mismatch / outside-grace condition; never
    raises.

    The cache is consulted only — no network call. This is the local
    file at ``~/.carl/entitlements_cache.json`` populated by an
    earlier successful :func:`fetch_remote_async`.
    """
    try:
        from carl_studio.entitlements import default_client
    except ImportError:
        return None

    try:
        client = default_client()
        cached = client.cache_get()
    except Exception:
        return None

    if cached is None:
        return None
    if cached.tier != "PAID":
        return None
    if not any(g.key == feature for g in cached.entitlements):
        return None
    if not client.is_offline_grace_valid(cached):
        return None
    return cached


def _emit_tier_remote_grace(feature: str, ent: Any) -> None:
    """Emit a structured ``tier_remote_grace`` gate event, best-effort.

    The current decorator wrapper has no direct handle on the active
    :class:`InteractionChain` (the local gate consumed the
    ``_gate_chain`` kwarg before raising), so :func:`emit_gate_event`
    no-ops here today. The call shape is preserved because (a) tests
    can patch this helper to assert the grace path was taken without
    threading a chain, and (b) when a future iteration plumbs the
    chain reference through ``BaseGate.for_predicate`` we'll fill it
    in here without changing the call sites. Failures are swallowed —
    a logging hiccup must never convert an admit into a deny.
    """
    with contextlib.suppress(Exception):
        cached_at = getattr(ent, "cached_at", None)
        key_id = getattr(ent, "key_id", None)
        cached_at_iso = (
            cached_at.isoformat() if cached_at is not None else "unknown"
        )
        emit_gate_event(
            predicate_name=f"tier_remote_grace:{feature}",
            allowed=True,
            reason=(
                f"local denied PAID '{feature}' but cached entitlements "
                f"({key_id}, cached_at={cached_at_iso}) within 24h "
                f"offline-grace contain a matching grant"
            ),
            gate_code=GATE_TIER_INSUFFICIENT,
            chain=None,
        )


def _resolve_bearer_token_for_verify() -> str | None:
    """Studio bearer-token resolution chain (v0.10).

    Mirrors the helper at ``a2a._cli._resolve_bearer_token`` and
    ``cli/resonant._resolve_bearer_token``: the public CLAUDE.md-doc'd
    order is ``CARL_CAMP_TOKEN`` env → ``~/.carl/camp_token`` legacy
    file → ``LocalDB.get_auth("jwt")``. Returns ``None`` when nothing
    is found — the verify call simply no-ops on missing auth.

    Kept module-private so tests can patch
    ``carl_studio.tier._resolve_bearer_token_for_verify`` without
    intruding on the two existing CLI-side helpers.
    """
    explicit = os.environ.get("CARL_CAMP_TOKEN")
    if explicit:
        return explicit

    from pathlib import Path

    token_path = Path.home() / ".carl" / "camp_token"
    if token_path.is_file():
        try:
            text = token_path.read_text().strip()
        except OSError:
            text = ""
        if text:
            return text

    try:
        from carl_studio.db import LocalDB

        return LocalDB().get_auth("jwt")
    except Exception:
        return None


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


def tier_message(feature: str) -> str | None:
    """Get an upgrade message if the feature is gated, or None if allowed.

    Friendly message for CLI output -- not an exception.
    """
    allowed, effective, required = check_tier(feature)
    if allowed:
        return None
    url = _UPGRADE_URLS.get(required, "https://terminals.tech/pricing")
    return f"This feature requires CARL {required.value.title()}. Upgrade at {url}"
