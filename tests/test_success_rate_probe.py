"""v0.11 Fano V7 — success_rate_probe full-stack integration test.

Verifies the complete loop:
1. Register default endogenous probe on an InteractionChain
2. Record steps with mixed success/failure outcomes
3. Probe populates kuramoto_r automatically via auto-attach
4. probe_call audit trail is written
5. @coherence_gate decides admission based on the rolling rate
6. BITC axiom 3 (endogenous measurability) holds — probe reads only
   the chain's own state

This is the production-ready realization of the IRE tuple's G + Φ
composition that Fano V7 flagged as aspirational at v0.10.0.
"""

from __future__ import annotations

from typing import Any

import pytest

from carl_core.interaction import ActionType, InteractionChain
from carl_core.presence import success_rate_probe


def _coherence_gate_and_error():
    """Lazy re-import so class identity matches the decorator's runtime class.

    conftest.py does module-reload trickery for transformers stubbing; that
    can desync the CoherenceError class identity between this test file
    (module-import-time capture) and the decorator body (runtime capture).
    Re-importing at call time ensures we always catch the LIVE class.
    """
    from carl_studio.gating import CoherenceError, coherence_gate

    return coherence_gate, CoherenceError


coherence_gate, CoherenceError = _coherence_gate_and_error()


class TestEndogenousProbeClosingTheLoop:
    def test_all_success_yields_high_R(self) -> None:
        chain = InteractionChain()
        chain.register_coherence_probe(success_rate_probe(chain))
        for i in range(6):
            chain.record(ActionType.TOOL_CALL, f"ok-{i}", success=True)
        # All steps but the last have entered the probe's window
        # (the one being recorded is not yet in chain.steps).
        # Check the last step's probe-populated kuramoto_r.
        last = chain.steps[-1]
        assert last.kuramoto_r == pytest.approx(1.0)
        assert last.probe_call is not None

    def test_alternating_failure_drives_R_down(self) -> None:
        chain = InteractionChain()
        chain.register_coherence_probe(success_rate_probe(chain))
        # Warmup failures
        for i in range(4):
            chain.record(ActionType.TOOL_CALL, f"fail-{i}", success=False)
        # New step now — probe sees 4 fails of same action type in window
        step = chain.record(ActionType.TOOL_CALL, "next", success=True)
        assert step.kuramoto_r == pytest.approx(0.0)

    def test_probe_filters_by_action_type(self) -> None:
        chain = InteractionChain()
        chain.register_coherence_probe(success_rate_probe(chain))
        # Mixed action types: failures on CLI_CMD, successes on TOOL_CALL
        for _ in range(3):
            chain.record(ActionType.CLI_CMD, "c", success=False)
        for _ in range(3):
            chain.record(ActionType.TOOL_CALL, "t", success=True)
        # New TOOL_CALL step — probe should see only prior TOOL_CALL
        # steps (all successful) → R = 1.0
        step = chain.record(ActionType.TOOL_CALL, "last", success=True)
        assert step.kuramoto_r == pytest.approx(1.0)

    def test_gate_allows_on_healthy_chain(self) -> None:
        chain = InteractionChain()
        chain.register_coherence_probe(success_rate_probe(chain))
        for _ in range(5):
            chain.record(ActionType.LLM_REPLY, "ok", success=True)

        @coherence_gate(min_R=0.6)
        def _work() -> str:
            return "done"

        assert _work(_gate_chain=chain) == "done"

    def test_gate_denies_on_failing_chain(self) -> None:
        chain = InteractionChain()
        chain.register_coherence_probe(success_rate_probe(chain))
        for _ in range(5):
            chain.record(ActionType.LLM_REPLY, "bad", success=False)

        coherence_gate, CoherenceError = _coherence_gate_and_error()

        @coherence_gate(min_R=0.6)
        def _work() -> str:
            return "done"

        with pytest.raises(Exception, match="coherence_insufficient|below required"):
            _work(_gate_chain=chain)

    def test_recovery_re_allows_gate(self) -> None:
        """End-to-end narrative: chain fails, gate denies, chain recovers,
        gate allows. This is the full recurrence → crystallization → admit
        loop BITC describes."""
        chain = InteractionChain()
        chain.register_coherence_probe(success_rate_probe(chain))

        coherence_gate, CoherenceError = _coherence_gate_and_error()

        @coherence_gate(min_R=0.5)
        def _admit() -> str:
            return "admitted"

        # 1. Fail multiple times — chain enters degraded state
        for _ in range(4):
            chain.record(ActionType.LLM_REPLY, "fail", success=False)
        with pytest.raises(Exception, match="coherence_insufficient|below required"):
            _admit(_gate_chain=chain)

        # 2. Recovery — enough successes that the gate's mean over
        # the last 16 steps climbs back above the threshold. The
        # probe's kuramoto_r lags because its rolling window
        # includes old failures until it fully rotates out.
        for _ in range(16):
            chain.record(ActionType.LLM_REPLY, "ok", success=True)

        # 3. Gate admits again
        assert _admit(_gate_chain=chain) == "admitted"


class TestProbeAuditTrail:
    def test_probe_call_populated_when_success_rate_probe_fires(self) -> None:
        chain = InteractionChain()
        chain.register_coherence_probe(success_rate_probe(chain))
        chain.record(ActionType.TOOL_CALL, "warmup", success=True)
        step = chain.record(ActionType.TOOL_CALL, "audited", success=True)
        assert step.probe_call is not None
        assert step.probe_call["populated"] == ["kuramoto_r"]
