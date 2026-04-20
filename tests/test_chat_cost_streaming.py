"""Tests for the per-turn cost streaming in ``carl chat``.

Covers the helpers ``_read_total_cost`` and ``_emit_turn_cost_delta`` in
``carl_studio.cli.chat``. These back the per-turn visibility line:

    turn: $0.0120  session: $0.0450

R2-007 collapsed the helper's three-way fallback (property -> callable
-> private attribute) to a single property-read path. The tests here
exercise:

* the canonical shape (``cost_summary`` is a property returning a dict);
* broken agents (raise / missing) — helper must never bubble up;
* delta math (positive prints, zero/negative suppress).

Agents that only expose the legacy private ``_total_cost_usd`` attribute
now resolve to ``0.0`` — this is intentional: tests that want cost
visibility should provide ``cost_summary`` directly, matching production.
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


class _AgentBroken:
    """Agent whose cost_summary raises — must not bubble up."""

    @property
    def cost_summary(self) -> dict[str, Any]:
        raise RuntimeError("boom")


class _AgentWithoutSummary:
    """Agent that lacks cost_summary entirely — must collapse to 0.0."""


def test_read_total_from_property() -> None:
    assert _read_total_cost(_AgentWithProperty(0.0120)) == pytest.approx(0.0120)


def test_read_total_safe_on_exception() -> None:
    """A broken cost_summary must not crash the chat loop."""
    assert _read_total_cost(_AgentBroken()) == 0.0


def test_read_total_safe_on_missing_summary() -> None:
    """Agents without cost_summary collapse to 0.0 (no private-attr fallback)."""
    assert _read_total_cost(_AgentWithoutSummary()) == 0.0


def test_read_total_safe_on_empty_agent() -> None:
    class _Empty: ...
    assert _read_total_cost(_Empty()) == 0.0


def test_read_total_handles_non_dict_summary() -> None:
    """If cost_summary is not a dict (e.g. a bare number), collapse to 0.0."""

    class _Weird:
        @property
        def cost_summary(self) -> Any:  # type: ignore[override]
            return 42  # no .get() method

    assert _read_total_cost(_Weird()) == 0.0


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
    """No delta -> no visible noise; still updates the running total."""
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
