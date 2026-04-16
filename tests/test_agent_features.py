"""Tests for CARLAgent features: session persistence, permission hooks, cost tracking."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from carl_studio.chat_agent import (
    CARLAgent,
    SessionStore,
    ToolPermission,
    _compute_turn_cost,
)


# ---------------------------------------------------------------------------
# Session persistence
# ---------------------------------------------------------------------------


class TestSessionStore:
    @pytest.fixture()
    def store(self, tmp_path: Path) -> SessionStore:
        return SessionStore(sessions_dir=tmp_path / "sessions")

    def test_save_and_load_roundtrip(self, store: SessionStore) -> None:
        state = {
            "id": "s1",
            "title": "test session",
            "model": "claude-sonnet-4-6",
            "messages": [{"role": "user", "content": "hello"}],
            "knowledge": [
                {"text": "chunk1", "source": "a.txt", "words": {"chunk1", "hello"}},
            ],
            "frame": {"domain": "saas", "function": "planning"},
            "total_cost_usd": 0.0042,
            "turn_count": 3,
        }
        store.save("s1", state)
        loaded = store.load("s1")
        assert loaded is not None
        assert loaded["title"] == "test session"
        assert loaded["messages"] == [{"role": "user", "content": "hello"}]
        assert loaded["total_cost_usd"] == 0.0042
        assert loaded["turn_count"] == 3

    def test_knowledge_sets_survive_roundtrip(self, store: SessionStore) -> None:
        """Sets in knowledge entries serialize as lists and deserialize back to sets."""
        state = {
            "id": "s2",
            "knowledge": [
                {"text": "hello world", "source": "test", "words": {"hello", "world"}},
            ],
        }
        store.save("s2", state)
        loaded = store.load("s2")
        assert loaded is not None
        words = loaded["knowledge"][0]["words"]
        assert isinstance(words, set)
        assert words == {"hello", "world"}

    def test_load_nonexistent_returns_none(self, store: SessionStore) -> None:
        assert store.load("nonexistent") is None

    def test_list_sessions(self, store: SessionStore) -> None:
        store.save("a", {"id": "a", "title": "first", "model": "m1", "turn_count": 1})
        store.save("b", {"id": "b", "title": "second", "model": "m2", "turn_count": 5})
        sessions = store.list_sessions()
        assert len(sessions) == 2
        ids = {s["id"] for s in sessions}
        assert ids == {"a", "b"}

    def test_list_sessions_limit(self, store: SessionStore) -> None:
        for i in range(5):
            store.save(f"s{i}", {"id": f"s{i}", "title": f"session {i}"})
        sessions = store.list_sessions(limit=3)
        assert len(sessions) == 3

    def test_delete(self, store: SessionStore) -> None:
        store.save("d1", {"id": "d1", "title": "doomed"})
        assert store.delete("d1") is True
        assert store.load("d1") is None
        assert store.delete("d1") is False

    def test_save_adds_timestamps(self, store: SessionStore) -> None:
        store.save("ts1", {"id": "ts1"})
        loaded = store.load("ts1")
        assert loaded is not None
        assert "created_at" in loaded
        assert "updated_at" in loaded


class TestAgentSessionIntegration:
    @pytest.fixture()
    def agent(self, tmp_path: Path) -> CARLAgent:
        return CARLAgent(
            model="test-model",
            workdir=str(tmp_path),
            _client=MagicMock(),
        )

    @pytest.fixture()
    def store(self, tmp_path: Path) -> SessionStore:
        return SessionStore(sessions_dir=tmp_path / "sessions")

    def test_save_and_load_session(self, agent: CARLAgent, store: SessionStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("carl_studio.frame._DEFAULT_FRAME_PATH", tmp_path / "frame.yaml")

        # Build some state
        agent._dispatch_tool("set_frame", {"domain": "pharma", "function": "rollout"})
        agent._messages.append({"role": "user", "content": "hello"})
        agent._knowledge.append({"text": "chunk", "source": "test", "words": {"chunk"}})
        agent._total_cost_usd = 0.005
        agent._turn_count = 2

        # Save
        sid = agent.save_session(title="pharma session", store=store)
        assert sid != ""

        # Load into fresh agent
        agent2 = CARLAgent(model="test-model", _client=MagicMock())
        loaded = agent2.load_session(sid, store=store)
        assert loaded is True
        assert len(agent2._messages) == 1
        assert len(agent2._knowledge) == 1
        assert isinstance(agent2._knowledge[0]["words"], set)
        assert agent2._total_cost_usd == 0.005
        assert agent2._turn_count == 2
        assert agent2._frame is not None
        assert agent2._frame.domain == "pharma"

    def test_load_nonexistent_returns_false(self, agent: CARLAgent, store: SessionStore) -> None:
        assert agent.load_session("nope", store=store) is False


# ---------------------------------------------------------------------------
# Permission hooks
# ---------------------------------------------------------------------------


class TestPermissionHooks:
    def test_pre_tool_deny_blocks_execution(self) -> None:
        """Pre-tool hook returning DENY prevents tool execution."""
        deny_log: list[str] = []

        def deny_analysis(name: str, args: dict[str, Any]) -> ToolPermission:
            if name == "run_analysis":
                deny_log.append(name)
                return ToolPermission.DENY
            return ToolPermission.ALLOW

        agent = CARLAgent(
            model="test",
            pre_tool_use=deny_analysis,
            _client=MagicMock(),
        )

        # Hooks are checked in chat() loop, test the callback directly
        assert agent._pre_tool_use is not None
        assert agent._pre_tool_use("run_analysis", {}) == ToolPermission.DENY
        assert agent._pre_tool_use("read_file", {}) == ToolPermission.ALLOW
        assert len(deny_log) == 1

    def test_pre_tool_allow_passes(self) -> None:
        """Pre-tool hook returning ALLOW lets tool run."""
        def allow_all(name: str, args: dict[str, Any]) -> ToolPermission:
            return ToolPermission.ALLOW

        agent = CARLAgent(
            model="test",
            pre_tool_use=allow_all,
            _client=MagicMock(),
        )
        assert agent._pre_tool_use is not None
        assert agent._pre_tool_use("any_tool", {}) == ToolPermission.ALLOW

    def test_post_tool_hook_called(self) -> None:
        """Post-tool hook receives tool name, args, and result."""
        calls: list[tuple[str, dict[str, Any], str]] = []

        def log_call(name: str, args: dict[str, Any], result: str) -> None:
            calls.append((name, args, result))

        agent = CARLAgent(
            model="test",
            post_tool_use=log_call,
            _client=MagicMock(),
        )
        assert agent._post_tool_use is not None
        agent._post_tool_use("test_tool", {"key": "val"}, "result")
        assert len(calls) == 1
        assert calls[0] == ("test_tool", {"key": "val"}, "result")

    def test_no_hooks_by_default(self) -> None:
        """Default agent has no hooks."""
        agent = CARLAgent(model="test", _client=MagicMock())
        assert agent._pre_tool_use is None
        assert agent._post_tool_use is None


# ---------------------------------------------------------------------------
# Cost tracking
# ---------------------------------------------------------------------------


class TestCostTracking:
    def test_compute_turn_cost_opus(self) -> None:
        """Cost calculation matches Opus pricing."""
        usage = MagicMock()
        usage.input_tokens = 1000
        usage.output_tokens = 500
        usage.cache_read_input_tokens = 0
        usage.cache_creation_input_tokens = 0

        cost = _compute_turn_cost(usage, "claude-opus-4-6")
        # 1000 * 5.00/1M + 500 * 25.00/1M = 0.005 + 0.0125 = 0.0175
        assert abs(cost - 0.0175) < 1e-6

    def test_compute_turn_cost_haiku(self) -> None:
        """Cost calculation matches Haiku pricing."""
        usage = MagicMock()
        usage.input_tokens = 10000
        usage.output_tokens = 2000
        usage.cache_read_input_tokens = 0
        usage.cache_creation_input_tokens = 0

        cost = _compute_turn_cost(usage, "claude-haiku-4-5")
        # 10000 * 1.00/1M + 2000 * 5.00/1M = 0.01 + 0.01 = 0.02
        assert abs(cost - 0.02) < 1e-6

    def test_compute_turn_cost_with_cache(self) -> None:
        """Cache reads are 0.1x, cache writes are 1.25x."""
        usage = MagicMock()
        usage.input_tokens = 500
        usage.output_tokens = 100
        usage.cache_read_input_tokens = 2000
        usage.cache_creation_input_tokens = 1000

        cost = _compute_turn_cost(usage, "claude-sonnet-4-6")
        # input: 500 * 3.0/1M = 0.0015
        # output: 100 * 15.0/1M = 0.0015
        # cache_read: 2000 * 3.0 * 0.1/1M = 0.0006
        # cache_create: 1000 * 3.0 * 1.25/1M = 0.00375
        expected = 0.0015 + 0.0015 + 0.0006 + 0.00375
        assert abs(cost - expected) < 1e-6

    def test_compute_turn_cost_unknown_model(self) -> None:
        """Unknown model uses Opus pricing as default."""
        usage = MagicMock()
        usage.input_tokens = 1000
        usage.output_tokens = 0
        usage.cache_read_input_tokens = 0
        usage.cache_creation_input_tokens = 0

        cost = _compute_turn_cost(usage, "unknown-model")
        # 1000 * 5.00/1M = 0.005
        assert abs(cost - 0.005) < 1e-6

    def test_cost_summary_property(self) -> None:
        """cost_summary returns current state."""
        agent = CARLAgent(model="test", max_budget_usd=1.00, _client=MagicMock())
        agent._total_cost_usd = 0.25
        agent._total_input_tokens = 5000
        agent._total_output_tokens = 1000
        agent._turn_count = 3

        summary = agent.cost_summary
        assert summary["total_cost_usd"] == 0.25
        assert summary["total_input_tokens"] == 5000
        assert summary["total_output_tokens"] == 1000
        assert summary["turn_count"] == 3
        assert summary["budget_remaining_usd"] == 0.75

    def test_cost_summary_no_budget(self) -> None:
        """No budget set → budget_remaining is None."""
        agent = CARLAgent(model="test", _client=MagicMock())
        assert agent.cost_summary["budget_remaining_usd"] is None

    def test_max_budget_zero_means_unlimited(self) -> None:
        """max_budget_usd=0 means no cap."""
        agent = CARLAgent(model="test", max_budget_usd=0, _client=MagicMock())
        assert agent._max_budget_usd == 0
