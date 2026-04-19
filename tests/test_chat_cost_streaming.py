"""Tests for the per-turn cost streaming in ``carl chat``.

Covers the helpers ``_read_total_cost`` and ``_emit_turn_cost_delta`` in
``carl_studio.cli.chat``. These back the per-turn visibility line:

    turn: $0.0120  session: $0.0450

The helpers must be defensive — chat observability is valuable but
never load-bearing, so a mock agent with a missing ``cost_summary`` or
a callable-instead-of-property shape should never raise.
"""

from __future__ import annotations

from typing import Any

import pytest

from carl_studio.cli.chat import _emit_turn_cost_delta, _read_total_cost


class _RecordingConsole:
    """Minimal stand-in for the CampConsole used by chat_cmd."""

    def __init__(self) -> None:
        self.info_lines: list[str] = []

    def info(self, msg: str) -> None:
        self.info_lines.append(msg)


class _AgentWithProperty:
    """Mimics the real CARLAgent — cost_summary is a property."""

    def __init__(self, total_cost_usd: float, turn_count: int = 1) -> None:
        self._total = total_cost_usd
        self._turns = turn_count

    @property
    def cost_summary(self) -> dict[str, Any]:
        return {
            "total_cost_usd": self._total,
            "total_input_tokens": 10,
            "total_output_tokens": 5,
            "turn_count": self._turns,
        }


class _AgentWithCallable:
    """Shape for tests where cost_summary is a method (defensive path)."""

    def __init__(self, total_cost_usd: float) -> None:
        self._total = total_cost_usd

    def cost_summary(self) -> dict[str, Any]:
        return {"total_cost_usd": self._total}


class _AgentWithoutSummary:
    """Fallback path — only the private ``_total_cost_usd`` is exposed."""

    def __init__(self, total_cost_usd: float) -> None:
        self._total_cost_usd = total_cost_usd


class _AgentBroken:
    """Agent whose cost_summary raises — must not bubble up."""

    @property
    def cost_summary(self) -> dict[str, Any]:
        raise RuntimeError("boom")


def test_read_total_from_property() -> None:
    assert _read_total_cost(_AgentWithProperty(0.0120)) == pytest.approx(0.0120)


def test_read_total_from_callable() -> None:
    assert _read_total_cost(_AgentWithCallable(0.0037)) == pytest.approx(0.0037)


def test_read_total_fallback_to_private_attr() -> None:
    assert _read_total_cost(_AgentWithoutSummary(0.0099)) == pytest.approx(0.0099)


def test_read_total_safe_on_exception() -> None:
    """A broken cost_summary must not crash the chat loop."""
    assert _read_total_cost(_AgentBroken()) == 0.0


def test_read_total_safe_on_empty_agent() -> None:
    class _Empty: ...
    assert _read_total_cost(_Empty()) == 0.0


def test_emit_positive_delta_prints_line() -> None:
    c = _RecordingConsole()
    agent = _AgentWithProperty(0.01)
    new_total = _emit_turn_cost_delta(c, agent, prev_total=0.0)

    assert new_total == pytest.approx(0.01)
    assert len(c.info_lines) == 1
    line = c.info_lines[0]
    assert "turn:" in line
    assert "session:" in line
    assert "$0.0100" in line


def test_emit_zero_delta_suppresses_line() -> None:
    """No delta → no visible noise; still updates the running total."""
    c = _RecordingConsole()
    agent = _AgentWithProperty(0.05)
    new_total = _emit_turn_cost_delta(c, agent, prev_total=0.05)

    assert new_total == pytest.approx(0.05)
    assert c.info_lines == []


def test_emit_negative_delta_suppresses_line() -> None:
    """Defensive: if an agent rewinds cost (shouldn't happen), don't print."""
    c = _RecordingConsole()
    agent = _AgentWithProperty(0.04)
    new_total = _emit_turn_cost_delta(c, agent, prev_total=0.05)

    assert new_total == pytest.approx(0.04)
    assert c.info_lines == []


def test_emit_accumulates_across_turns() -> None:
    c = _RecordingConsole()
    agent = _AgentWithProperty(0.01)
    prev = _emit_turn_cost_delta(c, agent, 0.0)

    agent._total = 0.03
    prev = _emit_turn_cost_delta(c, agent, prev)
    assert prev == pytest.approx(0.03)

    agent._total = 0.05
    prev = _emit_turn_cost_delta(c, agent, prev)
    assert prev == pytest.approx(0.05)

    assert len(c.info_lines) == 3
    assert "$0.0100" in c.info_lines[0]
    assert "$0.0200" in c.info_lines[1]
    assert "$0.0200" in c.info_lines[2]
    assert "$0.0500" in c.info_lines[2]


def test_emit_format_shape() -> None:
    """The rendered line must match the documented format exactly."""
    c = _RecordingConsole()
    agent = _AgentWithProperty(0.1234)
    _emit_turn_cost_delta(c, agent, prev_total=0.1114)

    line = c.info_lines[0]
    # Expected: "  turn: $0.0120  session: $0.1234"
    assert line.startswith("  turn: $")
    assert "  session: $" in line
    assert "0.0120" in line
    assert "0.1234" in line


def test_emit_handles_broken_agent_safely() -> None:
    """If the agent blows up, the helper should not leak the exception."""
    c = _RecordingConsole()
    new_total = _emit_turn_cost_delta(c, _AgentBroken(), prev_total=0.01)
    assert new_total == 0.0
    # No delta line because total went from 0.01 to 0.0 (negative delta).
    assert c.info_lines == []
