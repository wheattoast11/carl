"""v0.10 W9 — CoherenceGate predicate tests.

Peer-review finding P1-1 (2026-04-20): `BaseGate` gates on predicates
(consent, tier) but NEVER on coherence R. The IRE paper's "G"
(coherence-gated routing) was aspirational, not realized. This closes
the gap.

Contract pinned here:

1. CoherenceGatePredicate.check() returns (True, "no data") when chain
   is None or has no kuramoto_r on its steps.
2. Returns (True, reason) when sampled R >= min_R.
3. Returns (False, reason) when sampled R < min_R.
4. @coherence_gate(min_R=...) raises CoherenceError on deny with
   context carrying carl.gate.coherence_insufficient.
5. Decorator accepts _gate_chain kwarg and strips it before calling wrapped fn.
6. Invalid min_R (outside [0, 1]) raises ValueError at construction time.
7. Allow path emits a GATE_CHECK step on the chain with success=True.
8. Deny path emits a GATE_CHECK step with success=False + gate_code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from carl_core.interaction import ActionType, InteractionChain
from carl_studio.gating import (
    GATE_COHERENCE_INSUFFICIENT,
    CoherenceError,
    CoherenceGatePredicate,
    CoherenceSnapshot,
    coherence_gate,
    read_chain_coherence,
)


# ---------------------------------------------------------------------------
# Helpers: build a chain whose recent steps have explicit kuramoto_r values.
# ---------------------------------------------------------------------------


def _chain_with_R(values: list[float]) -> InteractionChain:
    chain = InteractionChain()
    for v in values:
        chain.record(
            ActionType.LLM_REPLY,
            "test.fixture",
            input={"seq": len(chain.steps)},
            output={"v": v},
            success=True,
            kuramoto_r=v,
        )
    return chain


# ---------------------------------------------------------------------------
# 1. read_chain_coherence
# ---------------------------------------------------------------------------


class TestReadChainCoherence:
    def test_none_chain_returns_no_data_default(self) -> None:
        snap = read_chain_coherence(None)
        assert snap == CoherenceSnapshot(R=1.0, window_size=0)
        assert snap.has_data is False

    def test_empty_chain_returns_no_data(self) -> None:
        snap = read_chain_coherence(InteractionChain())
        assert snap.has_data is False

    def test_chain_without_kuramoto_r_returns_no_data(self) -> None:
        chain = InteractionChain()
        chain.record(ActionType.CLI_CMD, "test", input={}, output={}, success=True)
        # No kuramoto_r populated → treated as no data
        snap = read_chain_coherence(chain)
        assert snap.has_data is False

    def test_mean_of_window(self) -> None:
        chain = _chain_with_R([0.1, 0.5, 0.9])
        snap = read_chain_coherence(chain, window=16)
        assert snap.window_size == 3
        assert snap.R == pytest.approx((0.1 + 0.5 + 0.9) / 3)

    def test_window_truncation_to_tail(self) -> None:
        chain = _chain_with_R([0.1, 0.1, 0.1, 0.9, 0.9])
        snap = read_chain_coherence(chain, window=2)
        assert snap.window_size == 2
        assert snap.R == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# 2. CoherenceGatePredicate
# ---------------------------------------------------------------------------


class TestCoherenceGatePredicate:
    def test_no_data_allows(self) -> None:
        pred = CoherenceGatePredicate(min_R=0.5, chain=None)
        allowed, reason = pred.check()
        assert allowed is True
        assert "no coherence data" in reason

    def test_above_threshold_allows(self) -> None:
        chain = _chain_with_R([0.8, 0.85, 0.9])
        pred = CoherenceGatePredicate(min_R=0.5, chain=chain)
        allowed, reason = pred.check()
        assert allowed is True
        assert "R=" in reason

    def test_below_threshold_denies(self) -> None:
        chain = _chain_with_R([0.1, 0.2, 0.3])
        pred = CoherenceGatePredicate(min_R=0.5, chain=chain)
        allowed, reason = pred.check()
        assert allowed is False
        assert "below required" in reason

    def test_invalid_min_R_raises(self) -> None:
        with pytest.raises(ValueError):
            CoherenceGatePredicate(min_R=1.5, chain=None)
        with pytest.raises(ValueError):
            CoherenceGatePredicate(min_R=-0.1, chain=None)

    def test_last_snapshot_populated_after_check(self) -> None:
        chain = _chain_with_R([0.5])
        pred = CoherenceGatePredicate(min_R=0.3, chain=chain)
        assert pred.last_snapshot is None
        pred.check()
        assert pred.last_snapshot is not None
        assert pred.last_snapshot.R == pytest.approx(0.5)

    def test_name_encodes_threshold(self) -> None:
        pred = CoherenceGatePredicate(min_R=0.5, chain=None)
        assert "coherence:" in pred.name
        assert "0.5" in pred.name


# ---------------------------------------------------------------------------
# 3. @coherence_gate decorator
# ---------------------------------------------------------------------------


class TestCoherenceGateDecorator:
    def test_allow_path_calls_wrapped_fn(self) -> None:
        chain = _chain_with_R([0.9])
        calls: list[Any] = []

        @coherence_gate(min_R=0.5, feature="test.allow")
        def fn(x: int) -> int:
            calls.append(x)
            return x * 2

        result = fn(7, _gate_chain=chain)
        assert result == 14
        assert calls == [7]

    def test_deny_path_raises_coherence_error(self) -> None:
        chain = _chain_with_R([0.1])

        @coherence_gate(min_R=0.5, feature="test.deny")
        def fn() -> None:
            pytest.fail("must not be called on deny")

        with pytest.raises(CoherenceError) as exc_info:
            fn(_gate_chain=chain)

        err = exc_info.value
        assert err.code == GATE_COHERENCE_INSUFFICIENT
        assert err.context["gate_code"] == GATE_COHERENCE_INSUFFICIENT
        assert err.context["required_R"] == 0.5
        assert err.current_R == pytest.approx(0.1)

    def test_no_data_allows_and_invokes(self) -> None:
        calls: list[Any] = []

        @coherence_gate(min_R=0.5)
        def fn(x: int) -> int:
            calls.append(x)
            return x

        result = fn(42)  # no chain passed
        assert result == 42
        assert calls == [42]

    def test_gate_chain_stripped_before_calling_fn(self) -> None:
        chain = _chain_with_R([0.9])
        captured: dict[str, Any] = {}

        @coherence_gate(min_R=0.5)
        def fn(**kwargs: Any) -> None:
            captured.update(kwargs)

        fn(foo="bar", _gate_chain=chain)
        assert "_gate_chain" not in captured
        assert captured == {"foo": "bar"}

    def test_functools_wraps_preserves_metadata(self) -> None:
        @coherence_gate(min_R=0.5)
        def my_special_name() -> None:
            """docstring lives on"""

        assert my_special_name.__name__ == "my_special_name"
        assert my_special_name.__doc__ == "docstring lives on"
        assert my_special_name.__wrapped__ is not None  # type: ignore[attr-defined]

    def test_feature_name_in_error_message(self) -> None:
        chain = _chain_with_R([0.1])

        @coherence_gate(min_R=0.5, feature="training.update")
        def fn() -> None:
            pass

        with pytest.raises(CoherenceError) as exc_info:
            fn(_gate_chain=chain)
        assert "training.update" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 4. GATE_CHECK event emission (allow + deny both emit)
# ---------------------------------------------------------------------------


class TestGateEventEmission:
    def test_allow_emits_gate_check_success(self) -> None:
        chain = _chain_with_R([0.9])

        @coherence_gate(min_R=0.5)
        def fn() -> None:
            pass

        fn(_gate_chain=chain)

        gate_steps = [s for s in chain.steps if s.action == ActionType.GATE_CHECK]
        assert len(gate_steps) == 1
        step = gate_steps[0]
        assert step.success is True

    def test_deny_emits_gate_check_failure_with_code(self) -> None:
        chain = _chain_with_R([0.1])

        @coherence_gate(min_R=0.5)
        def fn() -> None:
            pass

        with pytest.raises(CoherenceError):
            fn(_gate_chain=chain)

        gate_steps = [s for s in chain.steps if s.action == ActionType.GATE_CHECK]
        assert len(gate_steps) == 1
        step = gate_steps[0]
        assert step.success is False
        # gate_code carried in step output dict for filtering
        out = step.output or {}
        assert out.get("gate_code") == GATE_COHERENCE_INSUFFICIENT
