"""Tests for carl_studio.agent -- CARLAgent, tool dispatch, context management."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from carl_studio.chat_agent import (
    TOOLS,
    AgentEvent,
    CARLAgent,
    _COMPACT_THRESHOLD,
    _KEEP_RECENT,
    _TOOL_RESULT_MAX,
)


class FakeBlock:
    """Mock a Claude API content block."""

    def __init__(self, block_type: str, text: str = "", **kwargs: Any) -> None:
        self.type = block_type
        self.text = text
        self.id = kwargs.get("id", "tool_1")
        self.name = kwargs.get("name", "")
        self.input = kwargs.get("input", {})


class FakeResponse:
    """Mock a Claude API response."""

    def __init__(
        self,
        content: list[FakeBlock],
        stop_reason: str = "end_turn",
        input_tokens: int = 5000,
    ) -> None:
        self.content = content
        self.stop_reason = stop_reason
        self.usage = MagicMock(input_tokens=input_tokens)


class TestToolSchemas:
    def test_all_tools_have_required_fields(self) -> None:
        for tool in TOOLS:
            assert "name" in tool
            assert "description" in tool
            assert "input_schema" in tool
            assert tool["input_schema"]["type"] == "object"

    def test_tool_count(self) -> None:
        assert len(TOOLS) == 7

    def test_tool_names(self) -> None:
        names = {t["name"] for t in TOOLS}
        expected = {
            "ingest_source", "query_knowledge", "run_analysis",
            "create_file", "read_file", "set_frame", "list_files",
        }
        assert names == expected


class TestAgentEvent:
    def test_text_event(self) -> None:
        e = AgentEvent(kind="text", content="hello")
        assert e.kind == "text"
        assert e.content == "hello"

    def test_tool_call_event(self) -> None:
        e = AgentEvent(kind="tool_call", tool_name="run_analysis", tool_args={"code": "print(1)"})
        assert e.tool_name == "run_analysis"


class TestCARLAgentToolDispatch:
    @pytest.fixture()
    def agent(self, tmp_path: Path) -> CARLAgent:
        return CARLAgent(model="test-model", workdir=str(tmp_path), _client=MagicMock())

    def test_tool_read_file(self, agent: CARLAgent, tmp_path: Path) -> None:
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello world")
        result = agent._dispatch_tool("read_file", {"path": str(test_file)})
        assert "hello world" in result

    def test_tool_read_file_not_found(self, agent: CARLAgent, tmp_path: Path) -> None:
        result = agent._dispatch_tool("read_file", {"path": str(tmp_path / "nonexistent.txt")})
        assert "not found" in result.lower()

    def test_tool_read_file_truncation(self, agent: CARLAgent, tmp_path: Path) -> None:
        big_file = tmp_path / "big.txt"
        big_file.write_text("x" * 20_000)
        result = agent._dispatch_tool("read_file", {"path": str(big_file)})
        assert "truncated" in result
        assert len(result) < 20_000

    def test_tool_create_file(self, agent: CARLAgent, tmp_path: Path) -> None:
        target = tmp_path / "output.csv"
        result = agent._dispatch_tool("create_file", {"path": str(target), "content": "a,b\n1,2"})
        assert "Created" in result
        assert target.read_text() == "a,b\n1,2"

    def test_tool_list_files(self, agent: CARLAgent, tmp_path: Path) -> None:
        (tmp_path / "a.txt").write_text("x")
        (tmp_path / "b.csv").write_text("y")
        result = agent._dispatch_tool("list_files", {"path": str(tmp_path)})
        assert "a.txt" in result
        assert "b.csv" in result

    def test_tool_list_files_not_dir(self, agent: CARLAgent) -> None:
        result = agent._dispatch_tool("list_files", {"path": "/nonexistent"})
        assert "not a directory" in result.lower()

    def test_tool_run_analysis(self, agent: CARLAgent) -> None:
        result = agent._dispatch_tool("run_analysis", {"code": "print(2 + 2)"})
        assert "4" in result

    def test_tool_run_analysis_error(self, agent: CARLAgent) -> None:
        result = agent._dispatch_tool("run_analysis", {"code": "raise ValueError('boom')"})
        assert "boom" in result

    def test_tool_set_frame(self, agent: CARLAgent, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("carl_studio.frame._DEFAULT_FRAME_PATH", tmp_path / "frame.yaml")
        result = agent._dispatch_tool("set_frame", {
            "domain": "saas_sales",
            "function": "territory_planning",
            "role": "analyst",
        })
        assert "saas_sales" in result
        assert agent._frame is not None
        assert agent._frame.domain == "saas_sales"

    def test_tool_ingest(self, agent: CARLAgent, tmp_path: Path) -> None:
        (tmp_path / "data.txt").write_text("Revenue by territory: North $1M, South $2M")
        result = agent._dispatch_tool("ingest_source", {"path": str(tmp_path)})
        assert "Ingested" in result
        assert len(agent._knowledge) > 0

    def test_tool_query_empty(self, agent: CARLAgent) -> None:
        result = agent._dispatch_tool("query_knowledge", {"question": "anything"})
        assert "empty" in result.lower()

    def test_tool_query_with_knowledge(self, agent: CARLAgent) -> None:
        text1 = "Territory North has 500 accounts and 3 sellers"
        text2 = "Quota attainment last quarter was 87%"
        agent._knowledge = [
            {"text": text1, "source": "test", "words": set(text1.lower().split())},
            {"text": text2, "source": "test", "words": set(text2.lower().split())},
        ]
        result = agent._dispatch_tool("query_knowledge", {"question": "territory accounts sellers"})
        assert "500 accounts" in result

    def test_tool_unknown(self, agent: CARLAgent) -> None:
        result = agent._dispatch_tool("nonexistent", {})
        assert "Unknown" in result


class TestCARLAgentContextManagement:
    @pytest.fixture()
    def agent(self) -> CARLAgent:
        return CARLAgent(model="test-model", _client=MagicMock())

    def test_compact_reduces_messages(self, agent: CARLAgent) -> None:
        # Add many messages
        for i in range(30):
            agent._messages.append({"role": "user", "content": f"message {i}"})
            agent._messages.append({"role": "assistant", "content": f"response {i}"})

        agent._compact()

        # Should have: summary + ack + last KEEP_RECENT messages
        assert len(agent._messages) == 2 + _KEEP_RECENT
        assert "summary" in agent._messages[0]["content"].lower()

    def test_compact_preserves_recent(self, agent: CARLAgent) -> None:
        for i in range(30):
            agent._messages.append({"role": "user", "content": f"message {i}"})

        agent._compact()

        # Last message should be preserved
        recent_contents = [m["content"] for m in agent._messages[2:] if isinstance(m["content"], str)]
        assert "message 29" in recent_contents

    def test_no_compact_when_few_messages(self, agent: CARLAgent) -> None:
        agent._messages = [{"role": "user", "content": "hi"}]
        agent._compact()
        assert len(agent._messages) == 1  # unchanged


class TestCARLAgentSystemPrompt:
    @pytest.fixture()
    def agent(self) -> CARLAgent:
        return CARLAgent(model="test-model", _client=MagicMock())

    def test_system_prompt_without_frame(self, agent: CARLAgent) -> None:
        prompt = agent._build_system_prompt()
        assert "CARL" in prompt
        assert "No frame set" in prompt

    def test_system_prompt_with_frame(self, agent: CARLAgent) -> None:
        from carl_studio.frame import WorkFrame

        agent._frame = WorkFrame(domain="pharma", function="drug_rollout", role="manager")
        prompt = agent._build_system_prompt()
        assert "ACTIVE FRAME" in prompt
        assert "pharma" in prompt

    def test_system_prompt_with_knowledge(self, agent: CARLAgent) -> None:
        agent._knowledge = [{"text": "test", "source": "a"}, {"text": "test2", "source": "b"}]
        prompt = agent._build_system_prompt()
        assert "2 chunks" in prompt
        assert "2 sources" in prompt

    def test_system_prompt_includes_behavioral_instructions(self, agent: CARLAgent) -> None:
        prompt = agent._build_system_prompt()
        assert "DRIVE" in prompt
        assert "set_frame" in prompt
        assert "ingest_source" in prompt
