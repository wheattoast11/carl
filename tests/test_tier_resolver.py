"""Tests for the pluggable tier resolver (v0.8 · S3).

Private runtimes can install a ``_TIER_RESOLVER`` via
:func:`carl_studio.tier.register_tier_resolver` to source the effective
tier from a custom place (wallet balance, JWT claim, org-plan lookup)
without monkey-patching :func:`carl_studio.tier.detect_effective_tier`.

Contract pins:

* The default path (no resolver registered) behaves exactly as before.
* A registered resolver replaces the default ``detect_effective_tier``
  lookup inside :meth:`TierPredicate._effective` and the ``tier_gate``
  decorator's predicate construction.
* The resolver receives the predicate's feature name (``str | None``).
* Resolver exceptions surface as
  :class:`~carl_core.errors.CARLError` with
  ``code == "carl.tier.resolver_error"``.
* :func:`clear_tier_resolver` restores the default path.
* The resolver is called once per ``check()`` — no caching.
"""
from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import patch

import pytest

from carl_core.errors import CARLError
from carl_studio.tier import (
    Tier,
    TierGateError,
    TierPredicate,
    clear_tier_resolver,
    get_tier_resolver,
    register_tier_resolver,
    tier_gate,
)


@pytest.fixture(autouse=True)
def _reset_resolver() -> Iterator[None]:
    """Guarantee registry isolation between tests.

    The resolver slot is module-level; a test that forgets to clear
    would pollute every subsequent tier-related test. Autouse teardown
    guarantees the default path is always restored.
    """
    clear_tier_resolver()
    try:
        yield
    finally:
        clear_tier_resolver()


# ---------------------------------------------------------------------------
# Default path (no resolver) — regression guard
# ---------------------------------------------------------------------------


def test_default_resolver_path_unchanged() -> None:
    """Without a registered resolver, TierPredicate.check() behaves as before.

    The ``effective=`` override short-circuits the settings round-trip,
    so this exercises the same deny path v0.7.x shipped with — no
    resolver involvement whatsoever.
    """
    assert get_tier_resolver() is None
    predicate = TierPredicate(
        Tier.PAID, feature="mcp.serve", effective=Tier.FREE
    )
    allowed, reason = predicate.check()
    assert allowed is False
    assert "Paid" in reason


# ---------------------------------------------------------------------------
# Registered resolver overrides the default
# ---------------------------------------------------------------------------


def test_registered_resolver_overrides_default() -> None:
    """A registered resolver replaces detect_effective_tier() inside the gate.

    We force the default path to return FREE via patch; the resolver
    returns PAID; the gate must allow the PAID-gated call.
    """

    @tier_gate(Tier.PAID, feature="mcp.serve")
    def _guarded() -> str:
        return "ok"

    register_tier_resolver(lambda _feature: Tier.PAID)

    # If the resolver were ignored and the default path ran, this patch
    # would force FREE and the gate would raise. That it does not raise
    # is the contract we're pinning.
    with patch("carl_studio.tier.detect_effective_tier", return_value=Tier.FREE):
        assert _guarded() == "ok"


# ---------------------------------------------------------------------------
# Resolver receives the feature name
# ---------------------------------------------------------------------------


def test_resolver_receives_feature_name() -> None:
    """``@tier_gate(..., feature='x')`` must pass ``'x'`` to the resolver."""
    captured: list[str | None] = []

    def _resolver(feature: str | None) -> Tier:
        captured.append(feature)
        return Tier.PAID

    register_tier_resolver(_resolver)

    @tier_gate(Tier.PAID, feature="mcp.serve")
    def _guarded() -> str:
        return "ok"

    _guarded()
    assert captured == ["mcp.serve"]


def test_resolver_receives_function_name_when_feature_unset() -> None:
    """When ``feature`` is None, tier_gate defaults to the wrapped fn name.

    This matches the predicate construction in tier.py — ``feat_name =
    feature or func.__name__`` — so the resolver's ``str | None``
    argument is actually always a ``str`` through the decorator path.
    The ``None`` branch only applies when the predicate is constructed
    directly (see the direct-predicate test below).
    """
    captured: list[str | None] = []

    def _resolver(feature: str | None) -> Tier:
        captured.append(feature)
        return Tier.PAID

    register_tier_resolver(_resolver)

    @tier_gate(Tier.PAID)
    def my_fn() -> str:
        return "ok"

    my_fn()
    assert captured == ["my_fn"]


# ---------------------------------------------------------------------------
# Resolver exception wrapped in CARLError
# ---------------------------------------------------------------------------


def test_resolver_exception_wrapped_in_carl_error() -> None:
    """A resolver that raises must surface as CARLError('carl.tier.resolver_error').

    The inner exception's message is captured in ``context['inner']`` so
    operators have the root cause without losing the stable error code.
    """

    def _boom(_feature: str | None) -> Tier:
        raise RuntimeError("wallet backend down")

    register_tier_resolver(_boom)
    predicate = TierPredicate(Tier.PAID, feature="mcp.serve")
    with pytest.raises(CARLError) as excinfo:
        predicate.check()
    err = excinfo.value
    assert err.code == "carl.tier.resolver_error"
    assert "wallet backend down" in err.context.get("inner", "")
    # Cause chain preserved.
    assert isinstance(err.__cause__, RuntimeError)


def test_resolver_exception_surfaces_through_tier_gate() -> None:
    """tier_gate propagates the resolver's CARLError (it is not a TierGateError)."""

    def _boom(_feature: str | None) -> Tier:
        raise ValueError("jwt expired")

    register_tier_resolver(_boom)

    @tier_gate(Tier.PAID, feature="mcp.serve")
    def _guarded() -> str:
        return "ok"

    with pytest.raises(CARLError) as excinfo:
        _guarded()
    # Must NOT be squashed into a TierGateError — resolver failures are a
    # distinct operational class from tier-insufficient denials.
    assert not isinstance(excinfo.value, TierGateError)
    assert excinfo.value.code == "carl.tier.resolver_error"


# ---------------------------------------------------------------------------
# clear_tier_resolver restores the default
# ---------------------------------------------------------------------------


def test_clear_resolver_restores_default() -> None:
    register_tier_resolver(lambda _f: Tier.PAID)
    assert get_tier_resolver() is not None
    clear_tier_resolver()
    assert get_tier_resolver() is None

    # The default path is now back — patch detect_effective_tier to force
    # FREE and confirm the gate denies.
    @tier_gate(Tier.PAID, feature="mcp.serve")
    def _guarded() -> str:
        return "ok"

    with patch("carl_studio.tier.detect_effective_tier", return_value=Tier.FREE):
        with pytest.raises(TierGateError):
            _guarded()


# ---------------------------------------------------------------------------
# get_tier_resolver returns registered or None
# ---------------------------------------------------------------------------


def test_get_tier_resolver_returns_registered_or_none() -> None:
    assert get_tier_resolver() is None

    def _resolver(_feature: str | None) -> Tier:
        return Tier.PAID

    register_tier_resolver(_resolver)
    assert get_tier_resolver() is _resolver

    clear_tier_resolver()
    assert get_tier_resolver() is None


# ---------------------------------------------------------------------------
# Per-call resolution (no caching)
# ---------------------------------------------------------------------------


def test_resolver_called_for_each_check() -> None:
    """N calls to .check() must invoke the resolver N times.

    Private runtimes may derive tier from live state (wallet balance,
    session expiry); caching would defeat the point. The decorator's
    predicate_factory is invoked per call, and each predicate's
    ``_effective()`` invokes the resolver once.
    """
    counter = {"n": 0}

    def _resolver(_feature: str | None) -> Tier:
        counter["n"] += 1
        return Tier.PAID

    register_tier_resolver(_resolver)
    predicate = TierPredicate(Tier.PAID, feature="mcp.serve")
    for _ in range(5):
        predicate.check()
    assert counter["n"] == 5


def test_resolver_called_per_gate_invocation() -> None:
    """Parallel pin against the decorator path — each guarded call = 1 resolver call."""
    counter = {"n": 0}

    def _resolver(_feature: str | None) -> Tier:
        counter["n"] += 1
        return Tier.PAID

    register_tier_resolver(_resolver)

    @tier_gate(Tier.PAID, feature="mcp.serve")
    def _guarded() -> str:
        return "ok"

    for _ in range(3):
        _guarded()
    assert counter["n"] == 3
