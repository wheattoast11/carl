"""Tests for the shared gate primitives in ``carl_studio.gating``.

These tests verify the *shape unification* across the consent and tier
gates — both must emit identical structured-logging events and surface
the ``carl.gate.*`` namespace in their error ``context`` dict — without
collapsing the two public decorator APIs.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from carl_core.interaction import ActionType, InteractionChain
from carl_studio.consent import (
    ConsentError,
    ConsentManager,
    ConsentPredicate,
    consent_gate,
)
from carl_studio.gating import (
    GATE_CONSENT_DENIED,
    GATE_TIER_INSUFFICIENT,
    GatingPredicate,
    emit_gate_event,
)
from carl_studio.tier import Tier, TierGateError, TierPredicate, tier_gate


class FakeDB:
    """Minimal LocalDB stand-in used to exercise ConsentManager."""

    def __init__(self) -> None:
        self.config: dict[str, str] = {}

    def get_config(self, key: str, default: str | None = None) -> str | None:
        return self.config.get(key, default)

    def set_config(self, key: str, value: str) -> None:
        self.config[key] = value


# ---------------------------------------------------------------------------
# Protocol surface
# ---------------------------------------------------------------------------


def test_gating_predicate_is_runtime_checkable() -> None:
    """The Protocol is @runtime_checkable so we can isinstance() it."""

    class _Dummy:
        @property
        def name(self) -> str:
            return "dummy"

        def check(self) -> tuple[bool, str]:
            return True, "ok"

    assert isinstance(_Dummy(), GatingPredicate)


def test_consent_predicate_implements_gating_predicate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Undo the autouse "grant everything" fixture so we exercise the real
    # deny path. conftest patches ConsentManager.is_granted to ``True`` for
    # unrelated sync/x402/mcp tests; we want the un-mocked behaviour here.
    monkeypatch.setattr(ConsentManager, "is_granted", lambda self, key: False)
    db = FakeDB()
    mgr = ConsentManager(db=db)
    predicate = ConsentPredicate("telemetry", manager=mgr)
    assert isinstance(predicate, GatingPredicate)
    assert predicate.name == "consent:telemetry"
    allowed, reason = predicate.check()
    assert allowed is False
    assert "telemetry" in reason


def test_tier_predicate_implements_gating_predicate() -> None:
    predicate = TierPredicate(
        Tier.PAID, feature="mcp.serve", effective=Tier.FREE
    )
    assert isinstance(predicate, GatingPredicate)
    assert predicate.name == "tier:mcp.serve"
    allowed, reason = predicate.check()
    assert allowed is False
    assert "Paid" in reason


# ---------------------------------------------------------------------------
# Shared event shape
# ---------------------------------------------------------------------------


def _assert_gate_check_step(step: Any, *, predicate_name: str, allowed: bool) -> None:
    assert step.action is ActionType.GATE_CHECK
    assert step.name == predicate_name
    assert step.input == {"predicate": predicate_name, "allowed": allowed}
    assert "reason" in step.output
    assert step.success is allowed


def test_consent_gate_emits_shared_event_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """consent_gate with chain=<chain> must append exactly one GATE_CHECK step."""
    monkeypatch.setattr(ConsentManager, "is_granted", lambda self, key: False)
    chain = InteractionChain()
    db = FakeDB()
    mgr = ConsentManager(db=db)

    with pytest.raises(ConsentError):
        consent_gate("telemetry", manager=mgr, chain=chain)

    assert len(chain) == 1
    _assert_gate_check_step(
        chain.steps[0], predicate_name="consent:telemetry", allowed=False
    )


def test_tier_gate_emits_shared_event_shape() -> None:
    """tier_gate must append a GATE_CHECK step with identical shape."""
    chain = InteractionChain()

    @tier_gate(Tier.PAID, feature="mcp.serve")
    def _guarded() -> str:
        return "ok"

    # Force effective tier = FREE via detect_effective_tier patch (the
    # wrapper lazy-imports CARLSettings, so we can't patch it on
    # carl_studio.tier directly).
    with patch("carl_studio.tier.detect_effective_tier", return_value=Tier.FREE):
        with pytest.raises(TierGateError):
            # _gate_chain is a sentinel kwarg consumed by the tier_gate
            # wrapper before func(*args, **kwargs) — not part of the
            # decorated function's signature.
            _guarded(_gate_chain=chain)  # pyright: ignore[reportCallIssue]

    assert len(chain) == 1
    _assert_gate_check_step(
        chain.steps[0], predicate_name="tier:mcp.serve", allowed=False
    )


# ---------------------------------------------------------------------------
# Context / gate_code surfacing
# ---------------------------------------------------------------------------


def test_consent_error_has_gate_code_in_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ConsentManager, "is_granted", lambda self, key: False)
    db = FakeDB()
    mgr = ConsentManager(db=db)
    with pytest.raises(ConsentError) as excinfo:
        consent_gate("telemetry", manager=mgr)
    err = excinfo.value
    # Legacy code stays untouched (back-compat).
    assert err.code == "carl.consent.denied"
    # New shared gate_code is surfaced via context without breaking that.
    assert err.context.get("gate_code") == GATE_CONSENT_DENIED
    assert err.context.get("flag") == "telemetry"


def test_tier_gate_error_has_gate_code_in_context() -> None:
    @tier_gate(Tier.PAID, feature="mcp.serve")
    def _guarded() -> str:
        return "ok"

    with patch("carl_studio.tier.detect_effective_tier", return_value=Tier.FREE):
        with pytest.raises(TierGateError) as excinfo:
            _guarded()
    err = excinfo.value
    # The primitive attributes stay unchanged.
    assert err.feature == "mcp.serve"
    assert err.required is Tier.PAID
    assert err.current is Tier.FREE
    # And the shared gate_code is surfaced via err.context.
    ctx = getattr(err, "context", {})
    assert ctx.get("gate_code") == GATE_TIER_INSUFFICIENT
    assert ctx.get("feature") == "mcp.serve"


# ---------------------------------------------------------------------------
# Code-namespace invariants
# ---------------------------------------------------------------------------


def test_gate_codes_are_distinct_and_stable() -> None:
    assert GATE_CONSENT_DENIED != GATE_TIER_INSUFFICIENT
    # Both codes live under the shared carl.gate.* namespace.
    assert GATE_CONSENT_DENIED.startswith("carl.gate.")
    assert GATE_TIER_INSUFFICIENT.startswith("carl.gate.")
    # Exact current values (break deliberately if we ever rename them).
    assert GATE_CONSENT_DENIED == "carl.gate.consent_denied"
    assert GATE_TIER_INSUFFICIENT == "carl.gate.tier_insufficient"


def test_gate_codes_stable_across_imports() -> None:
    """Fresh imports must return the same string constants (no drift)."""
    import importlib

    import carl_studio.gating as gating_a

    reloaded = importlib.reload(gating_a)
    assert reloaded.GATE_CONSENT_DENIED == GATE_CONSENT_DENIED
    assert reloaded.GATE_TIER_INSUFFICIENT == GATE_TIER_INSUFFICIENT


# ---------------------------------------------------------------------------
# emit_gate_event
# ---------------------------------------------------------------------------


def test_emit_gate_event_noop_when_chain_none() -> None:
    """With chain=None the emitter is a silent no-op (no raises)."""
    emit_gate_event(
        predicate_name="consent:telemetry",
        allowed=False,
        reason="not granted",
        chain=None,
    )
    # No assertion beyond "does not raise" — that's the contract.


def test_emit_gate_event_appends_to_chain() -> None:
    """When a chain is supplied, exactly one GATE_CHECK step lands."""
    chain = InteractionChain()
    emit_gate_event(
        predicate_name="consent:telemetry",
        allowed=True,
        reason="granted",
        chain=chain,
    )
    assert len(chain) == 1
    step = chain.steps[0]
    assert step.action is ActionType.GATE_CHECK
    assert step.name == "consent:telemetry"
    assert step.success is True
    assert step.output == {"reason": "granted"}
