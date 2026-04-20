"""v0.10 W1 — tool-call witness completeness.

Peer-review finding P0-1 (2026-04-20): the chat_agent.py tool-execution loop
executes tools but previously did NOT record ``ActionType.TOOL_CALL`` steps
to the InteractionChain. CLI and memory operations WERE logged; tool calls
were NOT. That violated the "complete witness log" claim in the IRE paper
series and invalidated session replay through the tool path.

These tests pin the contract:

1. A successful tool dispatch records one ``TOOL_CALL`` step with
   ``success=True`` and ``outcome=ok``.
2. A tool denied by the ``pre_tool_use`` permission hook records a
   ``TOOL_CALL`` step with ``success=False`` and ``outcome=denied``.
3. A tool whose args fail schema validation records a ``TOOL_CALL`` step
   with ``success=False`` and ``outcome=schema_error``.
4. A tool whose dispatch raises records a ``TOOL_CALL`` step with
   ``success=False`` and ``outcome=error``.
5. Multiple tool_use blocks in one turn each produce their own
   ``TOOL_CALL`` step (no collapsing).
6. The recording never raises into the caller — observability is
   fire-and-forget.
"""

from __future__ import annotations

from typing import Any, Iterable, Iterator
from unittest.mock import MagicMock

import pytest

from carl_studio.chat_agent import CARLAgent, ToolPermission


# ---------------------------------------------------------------------------
# Minimal Anthropic SDK stream stubs (mirrors tests/test_chat_agent_robustness.py)
# ---------------------------------------------------------------------------


class _Usage:
    def __init__(self) -> None:
        self.input_tokens = 10
        self.output_tokens = 5
        self.cache_read_input_tokens = 0
        self.cache_creation_input_tokens = 0


class _TextBlock:
    def __init__(self, text: str) -> None:
        self.type = "text"
        self.text = text


class _ToolUseBlock:
    def __init__(self, name: str, args: dict[str, Any], block_id: str = "tu_1") -> None:
        self.type = "tool_use"
        self.name = name
        self.input = args
        self.id = block_id


class _Response:
    def __init__(
        self,
        content: Iterable[Any],
        *,
        stop_reason: str = "end_turn",
    ) -> None:
        self.content = list(content)
        self.stop_reason = stop_reason
        self.usage = _Usage()


class _StubStream:
    """Scripted stream; returns a fixed `final` message on get_final_message."""

    def __init__(self, final: _Response) -> None:
        self._final = final

    def __enter__(self) -> _StubStream:
        return self

    def __exit__(self, *_: Any) -> bool:
        return False

    def __iter__(self) -> Iterator[Any]:
        return iter(())

    def get_final_message(self) -> _Response:
        return self._final


def _client_with_responses(*responses: _Response) -> MagicMock:
    """Return a MagicMock client whose ``messages.stream`` yields the given
    responses in sequence across multiple calls (for multi-turn tests)."""

    streams = iter(_StubStream(r) for r in responses)
    client = MagicMock()
    client.messages.stream = MagicMock(side_effect=lambda *a, **kw: next(streams))
    return client


def _tool_call_steps(agent: CARLAgent) -> list[Any]:
    """Return all TOOL_CALL steps recorded on the agent's chain."""

    chain = agent._get_chain()
    if chain is None:
        return []
    from carl_core.interaction import ActionType

    return [s for s in chain.steps if s.action == ActionType.TOOL_CALL]


# ---------------------------------------------------------------------------
# Stub tool registrations so _dispatch_tool_safe routes through a known fn
# ---------------------------------------------------------------------------


@pytest.fixture
def registered_echo_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    """Register a trivial ``echo`` tool and point the dispatcher at it."""

    from carl_studio import tool_dispatcher

    def _echo(args: dict[str, Any]) -> str:
        return f"echo:{args.get('msg', '')}"

    # _dispatch_tool_safe consults TOOLS + an internal registry. The
    # simplest path: monkeypatch the dispatcher's runtime resolution so
    # our echo function is called regardless of schema.
    monkeypatch.setattr(
        tool_dispatcher,
        "_runtime_tool_registry",
        {"echo": _echo},
        raising=False,
    )
    monkeypatch.setattr(
        "carl_studio.chat_agent._validate_tool_args",
        lambda name, args: None,  # bypass schema check by default
    )
    monkeypatch.setattr(
        CARLAgent,
        "_dispatch_tool_safe",
        lambda self, name, args: (
            (_echo(args), False) if name == "echo" else (f"unknown tool: {name}", True)
        ),
    )


# ---------------------------------------------------------------------------
# Tests — one per outcome branch of the tool-execution loop
# ---------------------------------------------------------------------------


class TestToolCallWitness:
    """v0.10 W1 — chain.record(TOOL_CALL, ...) coverage across all branches."""

    def test_successful_tool_call_recorded(self, registered_echo_tool: None) -> None:
        """Happy path: tool runs, outcome=ok, success=True."""

        turn1 = _Response(
            [_ToolUseBlock("echo", {"msg": "hi"}, block_id="tu_ok")],
            stop_reason="tool_use",
        )
        turn2 = _Response([_TextBlock("done")], stop_reason="end_turn")
        agent = CARLAgent(
            model="test-model",
            _client=_client_with_responses(turn1, turn2),
            api_key="sk-ant-stub",
        )

        list(agent.chat("test"))

        steps = _tool_call_steps(agent)
        assert len(steps) == 1, "exactly one TOOL_CALL step expected"
        step = steps[0]
        assert step.success is True
        assert step.input["tool_name"] == "echo"
        assert step.input["args"] == {"msg": "hi"}
        assert step.output["outcome"] == "ok"
        assert step.output["result"] == "echo:hi"
        assert step.duration_ms is not None and step.duration_ms >= 0.0

    def test_permission_denied_tool_call_recorded(
        self, registered_echo_tool: None
    ) -> None:
        """pre_tool_use DENY path is still a witness-log entry."""

        turn1 = _Response(
            [_ToolUseBlock("echo", {"msg": "x"}, block_id="tu_deny")],
            stop_reason="tool_use",
        )
        turn2 = _Response([_TextBlock("ok")], stop_reason="end_turn")
        agent = CARLAgent(
            model="test-model",
            _client=_client_with_responses(turn1, turn2),
            api_key="sk-ant-stub",
            pre_tool_use=lambda name, args: ToolPermission.DENY,
        )

        list(agent.chat("test"))

        steps = _tool_call_steps(agent)
        assert len(steps) == 1
        step = steps[0]
        assert step.success is False
        assert step.output["outcome"] == "denied"
        assert "Blocked by permission policy" in step.output["result"]

    def test_schema_error_tool_call_recorded(
        self, registered_echo_tool: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Invalid args per the tool schema are still a witness-log entry."""

        monkeypatch.setattr(
            "carl_studio.chat_agent._validate_tool_args",
            lambda name, args: "missing required field: msg",
        )
        turn1 = _Response(
            [_ToolUseBlock("echo", {}, block_id="tu_schema")],
            stop_reason="tool_use",
        )
        turn2 = _Response([_TextBlock("ok")], stop_reason="end_turn")
        agent = CARLAgent(
            model="test-model",
            _client=_client_with_responses(turn1, turn2),
            api_key="sk-ant-stub",
        )

        list(agent.chat("test"))

        steps = _tool_call_steps(agent)
        assert len(steps) == 1
        step = steps[0]
        assert step.success is False
        assert step.output["outcome"] == "schema_error"
        assert "missing required field" in step.output["result"]

    def test_dispatch_error_tool_call_recorded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A tool whose dispatch returns is_error=True is logged as outcome=error."""

        monkeypatch.setattr(
            "carl_studio.chat_agent._validate_tool_args",
            lambda name, args: None,
        )
        monkeypatch.setattr(
            CARLAgent,
            "_dispatch_tool_safe",
            lambda self, name, args: ("boom", True),
        )
        turn1 = _Response(
            [_ToolUseBlock("echo", {"msg": "x"}, block_id="tu_err")],
            stop_reason="tool_use",
        )
        turn2 = _Response([_TextBlock("ok")], stop_reason="end_turn")
        agent = CARLAgent(
            model="test-model",
            _client=_client_with_responses(turn1, turn2),
            api_key="sk-ant-stub",
        )

        list(agent.chat("test"))

        steps = _tool_call_steps(agent)
        assert len(steps) == 1
        step = steps[0]
        assert step.success is False
        assert step.output["outcome"] == "error"
        assert step.output["result"] == "boom"

    def test_multiple_tools_in_one_turn_each_recorded(
        self, registered_echo_tool: None
    ) -> None:
        """Three tool_use blocks in one turn => three TOOL_CALL steps."""

        turn1 = _Response(
            [
                _ToolUseBlock("echo", {"msg": "a"}, block_id="tu_a"),
                _ToolUseBlock("echo", {"msg": "b"}, block_id="tu_b"),
                _ToolUseBlock("echo", {"msg": "c"}, block_id="tu_c"),
            ],
            stop_reason="tool_use",
        )
        turn2 = _Response([_TextBlock("ok")], stop_reason="end_turn")
        agent = CARLAgent(
            model="test-model",
            _client=_client_with_responses(turn1, turn2),
            api_key="sk-ant-stub",
        )

        list(agent.chat("test"))

        steps = _tool_call_steps(agent)
        assert len(steps) == 3
        assert [s.input["args"]["msg"] for s in steps] == ["a", "b", "c"]
        assert all(s.success for s in steps)
        assert all(s.output["outcome"] == "ok" for s in steps)

    def test_recording_never_raises_into_caller(
        self, registered_echo_tool: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If chain.record() itself raises, the loop still completes cleanly."""

        # Break chain.record so any call raises
        class _BrokenChain:
            steps: list[Any] = []

            def record(self, *args: Any, **kwargs: Any) -> None:
                raise RuntimeError("chain persistence offline")

        turn1 = _Response(
            [_ToolUseBlock("echo", {"msg": "x"}, block_id="tu_rb")],
            stop_reason="tool_use",
        )
        turn2 = _Response([_TextBlock("ok")], stop_reason="end_turn")
        agent = CARLAgent(
            model="test-model",
            _client=_client_with_responses(turn1, turn2),
            api_key="sk-ant-stub",
        )
        agent._chain = _BrokenChain()  # type: ignore[assignment]

        # Must not raise.
        list(agent.chat("test"))
