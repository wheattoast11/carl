"""v0.14 — ToolDispatcher.execute_block tests.

Pins the full per-block lifecycle so the chat_agent loop body can shrink
to a delegation call with confidence. Covers the four outcome branches
identical to the original inline loop:

1. ok             — pre-hook ALLOW, validator clean, dispatch ok
2. denied         — pre-hook returns DENY
3. schema_error   — validator returns a string
4. error          — dispatch returns (result, is_error=True)

Plus hook resilience:
5. pre-hook raising → swallowed; permission defaults to ALLOW; error event emitted
6. post-hook raising → swallowed; result still returned; error event emitted
7. events sequence matches the agent's pre-v0.14 AgentEvent sequence
"""

from __future__ import annotations

from typing import Any

import pytest

from carl_studio.tool_dispatcher import (
    ToolDispatcher,
    ToolEvent,
    ToolOutcome,
    ToolPermission,
)


def _make_dispatcher_with_echo() -> ToolDispatcher:
    d = ToolDispatcher()
    d.register_simple("echo", lambda args: f"echo:{args.get('msg', '')}")
    return d


def _make_dispatcher_with_error_tool() -> ToolDispatcher:
    d = ToolDispatcher()
    d.register("boom", lambda args: ("pre-computed failure", True))
    return d


class TestExecuteBlockOutcomes:
    def test_ok_path(self) -> None:
        d = _make_dispatcher_with_echo()
        outcome, events = d.execute_block(
            tool_name="echo",
            tool_input={"msg": "hi"},
            tool_use_id="tu_1",
        )
        assert outcome.outcome == "ok"
        assert outcome.is_error is False
        assert outcome.result == "echo:hi"
        assert outcome.duration_ms >= 0.0
        assert len(events) == 1
        assert events[0].kind == "tool_result"

    def test_denied_path(self) -> None:
        d = _make_dispatcher_with_echo()
        outcome, events = d.execute_block(
            tool_name="echo",
            tool_input={"msg": "hi"},
            tool_use_id="tu_2",
            pre_hook=lambda name, args: ToolPermission.DENY,
        )
        assert outcome.outcome == "denied"
        assert outcome.is_error is True
        assert "Blocked by permission policy" in outcome.result
        assert any(e.kind == "tool_blocked" for e in events)

    def test_schema_error_path(self) -> None:
        d = _make_dispatcher_with_echo()
        outcome, events = d.execute_block(
            tool_name="echo",
            tool_input={},
            tool_use_id="tu_3",
            validator=lambda name, args: "missing required field: msg",
        )
        assert outcome.outcome == "schema_error"
        assert outcome.is_error is True
        assert "missing required" in outcome.result
        assert events[0].code == "carl.tool_schema"

    def test_dispatch_error_path(self) -> None:
        d = _make_dispatcher_with_error_tool()
        outcome, events = d.execute_block(
            tool_name="boom",
            tool_input={},
            tool_use_id="tu_4",
        )
        assert outcome.outcome == "error"
        assert outcome.is_error is True
        assert outcome.result == "pre-computed failure"


class TestHookResilience:
    def test_pre_hook_exception_defaults_allow_and_emits_error(self) -> None:
        d = _make_dispatcher_with_echo()

        def _boom(name: str, args: dict[str, Any]) -> ToolPermission:
            raise RuntimeError("pre hook exploded")

        outcome, events = d.execute_block(
            tool_name="echo",
            tool_input={"msg": "hi"},
            tool_use_id="tu_5",
            pre_hook=_boom,
        )
        assert outcome.outcome == "ok"
        # First event should be the hook-failure error
        assert events[0].kind == "error"
        assert events[0].code == "carl.hook_failed"

    def test_post_hook_exception_does_not_change_outcome(self) -> None:
        d = _make_dispatcher_with_echo()

        def _boom(name: str, args: dict[str, Any], result: str) -> None:
            raise RuntimeError("post hook exploded")

        outcome, events = d.execute_block(
            tool_name="echo",
            tool_input={"msg": "hi"},
            tool_use_id="tu_6",
            post_hook=_boom,
        )
        assert outcome.outcome == "ok"
        assert outcome.result == "echo:hi"
        # tool_result event + post-hook error event
        assert any(e.kind == "error" and e.code == "carl.hook_failed" for e in events)


class TestOutcomeFieldPreservation:
    def test_outcome_carries_original_input(self) -> None:
        d = _make_dispatcher_with_echo()
        input_dict = {"msg": "preserve me", "extra": 42}
        outcome, _ = d.execute_block(
            tool_name="echo",
            tool_input=input_dict,
            tool_use_id="tu_7",
        )
        assert outcome.input == input_dict

    def test_outcome_immutable(self) -> None:
        """Frozen dataclass — caller can't mutate outcome after the fact."""
        d = _make_dispatcher_with_echo()
        outcome, _ = d.execute_block(
            tool_name="echo", tool_input={}, tool_use_id="tu_8"
        )
        with pytest.raises(Exception):  # FrozenInstanceError
            outcome.result = "mutated"  # type: ignore[misc]
