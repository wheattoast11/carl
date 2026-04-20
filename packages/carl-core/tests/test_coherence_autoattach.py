"""v0.10 W10 — coherence auto-attach via registered probe.

Peer-review finding P1-2 (2026-04-20): Step coherence fields (phi,
kuramoto_r, channel_coherence) were optional and rarely populated
because there was no auto-attach at LLM_REPLY / TOOL_CALL boundaries.
This closes the gap.

Contract:
1. Default (no probe registered) — record() behaves identically to pre-v0.10.
2. Probe registered + action is auto-attach-eligible + no explicit
   coherence kwargs → probe is called and its return populates fields.
3. Probe registered + action NOT eligible (CLI_CMD, USER_INPUT) →
   probe is NOT called; fields stay None.
4. Explicit coherence kwargs override probe (probe is NOT called).
5. Probe raising → swallowed; step is still recorded with None fields.
6. Probe returning non-dict → ignored; fields stay None.
7. Partial probe return (only phi) → only phi populated; others stay None.
8. clear_coherence_probe() restores default behavior.
"""

from __future__ import annotations

from typing import Any

import pytest

from carl_core.interaction import ActionType, InteractionChain


def _eligible_actions() -> list[ActionType]:
    return [
        ActionType.LLM_REPLY,
        ActionType.TOOL_CALL,
        ActionType.TRAINING_STEP,
        ActionType.EVAL_PHASE,
        ActionType.REWARD,
    ]


def _non_eligible_actions() -> list[ActionType]:
    return [
        ActionType.USER_INPUT,
        ActionType.CLI_CMD,
        ActionType.GATE_CHECK,
        ActionType.MEMORY_READ,
        ActionType.STICKY_NOTE,
    ]


class TestAutoAttachContract:
    def test_default_no_probe_leaves_fields_none(self) -> None:
        chain = InteractionChain()
        step = chain.record(ActionType.LLM_REPLY, "default", success=True)
        assert step.phi is None
        assert step.kuramoto_r is None
        assert step.channel_coherence is None

    @pytest.mark.parametrize("action", _eligible_actions())
    def test_probe_populates_on_eligible_action(self, action: ActionType) -> None:
        chain = InteractionChain()
        chain.register_coherence_probe(
            lambda **kw: {"phi": 0.42, "kuramoto_r": 0.73, "channel_coherence": {"lm": 0.9}}
        )
        step = chain.record(action, "eligible", success=True)
        assert step.phi == pytest.approx(0.42)
        assert step.kuramoto_r == pytest.approx(0.73)
        assert step.channel_coherence == {"lm": 0.9}

    @pytest.mark.parametrize("action", _non_eligible_actions())
    def test_probe_NOT_called_on_non_eligible_action(
        self, action: ActionType
    ) -> None:
        calls: list[Any] = []
        chain = InteractionChain()
        chain.register_coherence_probe(
            lambda **kw: (calls.append(kw), {"phi": 0.5})[1]
        )
        step = chain.record(action, "ineligible", success=True)
        assert calls == []
        assert step.phi is None

    def test_explicit_kwargs_override_probe(self) -> None:
        calls: list[Any] = []
        chain = InteractionChain()
        chain.register_coherence_probe(
            lambda **kw: (calls.append(kw), {"phi": 0.1, "kuramoto_r": 0.2})[1]
        )
        step = chain.record(
            ActionType.LLM_REPLY,
            "explicit",
            phi=0.9,
            kuramoto_r=0.8,
        )
        # Probe NOT invoked when explicit values are passed
        assert calls == []
        assert step.phi == pytest.approx(0.9)
        assert step.kuramoto_r == pytest.approx(0.8)

    def test_probe_exception_is_swallowed(self) -> None:
        chain = InteractionChain()

        def _boom(**kwargs: Any) -> dict[str, Any]:
            raise RuntimeError("probe crashed")

        chain.register_coherence_probe(_boom)
        # Must not raise
        step = chain.record(ActionType.TOOL_CALL, "boom")
        assert step.phi is None
        assert step.kuramoto_r is None

    def test_probe_returning_non_dict_ignored(self) -> None:
        chain = InteractionChain()
        chain.register_coherence_probe(lambda **kw: "not a dict")
        step = chain.record(ActionType.LLM_REPLY, "bad")
        assert step.phi is None

    def test_partial_probe_populates_only_given_fields(self) -> None:
        chain = InteractionChain()
        chain.register_coherence_probe(lambda **kw: {"phi": 0.5})
        step = chain.record(ActionType.LLM_REPLY, "partial")
        assert step.phi == pytest.approx(0.5)
        assert step.kuramoto_r is None
        assert step.channel_coherence is None

    def test_clear_probe_restores_default(self) -> None:
        chain = InteractionChain()
        chain.register_coherence_probe(lambda **kw: {"phi": 0.5})
        # Confirm probe works
        s1 = chain.record(ActionType.LLM_REPLY, "with-probe")
        assert s1.phi == pytest.approx(0.5)
        # Clear
        chain.clear_coherence_probe()
        s2 = chain.record(ActionType.LLM_REPLY, "no-probe")
        assert s2.phi is None

    def test_probe_receives_action_and_context(self) -> None:
        received: dict[str, Any] = {}
        chain = InteractionChain()

        def _probe(**kwargs: Any) -> dict[str, Any]:
            received.update(kwargs)
            return {"phi": 0.5}

        chain.register_coherence_probe(_probe)
        chain.record(
            ActionType.LLM_REPLY,
            "my-step",
            input={"prompt": "hi"},
            output={"reply": "hello"},
        )
        assert received["action"] == ActionType.LLM_REPLY
        assert received["name"] == "my-step"
        assert received["input"] == {"prompt": "hi"}
        assert received["output"] == {"reply": "hello"}
