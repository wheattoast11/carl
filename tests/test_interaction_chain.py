"""Tests for the InteractionChain primitive in carl-core."""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from carl_core.interaction import ActionType, InteractionChain, Step


class TestStep:
    def test_defaults(self) -> None:
        step = Step(action=ActionType.USER_INPUT, name="hello")
        assert step.name == "hello"
        assert step.action == ActionType.USER_INPUT
        assert step.success is True
        assert step.duration_ms is None
        assert step.parent_id is None
        assert len(step.step_id) == 12
        assert isinstance(step.started_at, datetime)
        assert step.started_at.tzinfo == timezone.utc

    def test_to_dict_roundtrips_via_json(self) -> None:
        step = Step(action=ActionType.TOOL_CALL, name="ingest", input={"path": "x"}, output=42, duration_ms=1.5)
        d = step.to_dict()
        encoded = json.dumps(d)
        back = json.loads(encoded)
        assert back["action"] == "tool_call"
        assert back["name"] == "ingest"
        assert back["input"] == {"path": "x"}
        assert back["output"] == 42
        assert back["duration_ms"] == 1.5

    def test_to_dict_falls_back_to_repr_for_unserializable(self) -> None:
        class Unhashable:
            def __repr__(self) -> str:
                return "<Unhashable>"
        step = Step(action=ActionType.EXTERNAL, name="weird", input=Unhashable())
        d = step.to_dict()
        assert d["input"] == "<Unhashable>"


class TestInteractionChain:
    def test_empty_chain(self) -> None:
        chain = InteractionChain()
        assert len(chain) == 0
        assert chain.last() is None
        assert chain.success_rate() == 1.0
        assert chain.chain_id
        assert chain.started_at.tzinfo == timezone.utc

    def test_append(self) -> None:
        chain = InteractionChain()
        step = Step(action=ActionType.USER_INPUT, name="hi")
        chain.append(step)
        assert len(chain) == 1
        assert chain.last() is step

    def test_append_overrides_output(self) -> None:
        chain = InteractionChain()
        step = Step(action=ActionType.TOOL_CALL, name="foo", output="orig")
        chain.append(step, result="override")
        assert chain.last().output == "override"

    def test_record_builds_and_appends(self) -> None:
        chain = InteractionChain()
        step = chain.record(ActionType.CLI_CMD, "carl chat", input="hi")
        assert step.name == "carl chat"
        assert chain.last() is step

    def test_by_action_filters(self) -> None:
        chain = InteractionChain()
        chain.record(ActionType.USER_INPUT, "a")
        chain.record(ActionType.TOOL_CALL, "b")
        chain.record(ActionType.USER_INPUT, "c")
        picked = chain.by_action(ActionType.USER_INPUT)
        assert [s.name for s in picked] == ["a", "c"]

    def test_success_rate(self) -> None:
        chain = InteractionChain()
        chain.record(ActionType.TOOL_CALL, "ok")
        chain.record(ActionType.GATE, "blocked", success=False)
        chain.record(ActionType.LLM_REPLY, "fine")
        assert chain.success_rate() == pytest.approx(2 / 3)


class TestSerialization:
    def test_to_dict_roundtrip(self) -> None:
        chain = InteractionChain()
        chain.record(ActionType.CLI_CMD, "carl doctor", output="ok")
        chain.record(ActionType.GATE, "require:HF_TOKEN", success=False)
        chain.context["project"] = "demo"

        d = chain.to_dict()
        restored = InteractionChain.from_dict(d)
        assert restored.chain_id == chain.chain_id
        assert len(restored.steps) == 2
        assert restored.steps[0].name == "carl doctor"
        assert restored.steps[1].success is False
        assert restored.context == {"project": "demo"}

    def test_to_jsonl_is_parseable(self) -> None:
        chain = InteractionChain()
        chain.record(ActionType.USER_INPUT, "hello")
        encoded = chain.to_jsonl()
        lines = encoded.splitlines()
        assert len(lines) == 2
        header = json.loads(lines[0])
        row = json.loads(lines[1])
        assert header["chain_id"] == chain.chain_id
        assert row["name"] == "hello"

    def test_from_dict_defaults_when_fields_missing(self) -> None:
        chain = InteractionChain.from_dict({"steps": [{"action": "user_input", "name": "bare"}]})
        assert len(chain.steps) == 1
        assert chain.steps[0].action == ActionType.USER_INPUT
        assert chain.steps[0].name == "bare"


class TestActionType:
    def test_enum_values_stable(self) -> None:
        assert ActionType.USER_INPUT.value == "user_input"
        assert ActionType.TOOL_CALL.value == "tool_call"
        assert ActionType.LLM_REPLY.value == "llm_reply"
        assert ActionType.CLI_CMD.value == "cli_cmd"
        assert ActionType.GATE.value == "gate"
        assert ActionType.EXTERNAL.value == "external"
