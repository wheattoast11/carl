"""Tests for A2A protocol spec adapter (carl_studio.a2a.spec).

Covers:
  - agent_card_to_spec: required fields, skills mapping, stdio endpoint
  - task_to_jsonrpc_result: status mapping for all CARL states
  - message_send_to_task: text extraction, missing fields, type validation
  - wrap_jsonrpc_response / wrap_jsonrpc_error: JSON-RPC 2.0 envelope structure
"""
from __future__ import annotations

import uuid

import pytest

from carl_studio.a2a.agent_card import CARLAgentCard
from carl_studio.a2a.spec import (
    INTERNAL_ERROR,
    INVALID_PARAMS,
    METHOD_NOT_FOUND,
    agent_card_to_spec,
    message_send_to_task,
    task_to_jsonrpc_result,
    wrap_jsonrpc_error,
    wrap_jsonrpc_response,
)
from carl_studio.a2a.task import A2ATask, A2ATaskStatus


# ---- agent_card_to_spec -----------------------------------------------------


class TestAgentCardToSpec:
    """Verify CARLAgentCard -> A2A spec AgentCard mapping."""

    def test_required_fields_present(self) -> None:
        card = CARLAgentCard(name="test-agent", version="1.0.0")
        spec = agent_card_to_spec(card)

        assert spec["name"] == "test-agent"
        assert "description" in spec
        assert spec["version"] == "1.0.0"
        assert "capabilities" in spec
        assert "defaultInputModes" in spec
        assert "defaultOutputModes" in spec
        assert "skills" in spec

    def test_description_includes_version(self) -> None:
        card = CARLAgentCard(version="2.5.0")
        spec = agent_card_to_spec(card)
        assert "2.5.0" in spec["description"]

    def test_stdio_endpoint_maps_to_none(self) -> None:
        card = CARLAgentCard(endpoint="stdio")
        spec = agent_card_to_spec(card)
        assert spec["url"] is None

    def test_http_endpoint_preserved(self) -> None:
        card = CARLAgentCard(endpoint="http://localhost:8080")
        spec = agent_card_to_spec(card)
        assert spec["url"] == "http://localhost:8080"

    def test_skills_mapped_with_id_and_name(self) -> None:
        card = CARLAgentCard(skills=["train", "eval", "bench"])
        spec = agent_card_to_spec(card)

        assert len(spec["skills"]) == 3
        for skill in spec["skills"]:
            assert "id" in skill
            assert "name" in skill
            assert "description" in skill
            assert skill["id"] == skill["name"]

    def test_empty_skills(self) -> None:
        card = CARLAgentCard(skills=[])
        spec = agent_card_to_spec(card)
        assert spec["skills"] == []

    def test_capabilities_structure(self) -> None:
        card = CARLAgentCard()
        spec = agent_card_to_spec(card)
        caps = spec["capabilities"]
        assert isinstance(caps, dict)
        assert "streaming" in caps
        assert "pushNotifications" in caps

    def test_default_input_output_modes(self) -> None:
        card = CARLAgentCard()
        spec = agent_card_to_spec(card)
        assert spec["defaultInputModes"] == ["text"]
        assert spec["defaultOutputModes"] == ["text"]

    def test_to_a2a_spec_method(self) -> None:
        """CARLAgentCard.to_a2a_spec() delegates to agent_card_to_spec."""
        card = CARLAgentCard(name="roundtrip", version="0.1.0", skills=["align"])
        via_method = card.to_a2a_spec()
        via_function = agent_card_to_spec(card)
        assert via_method == via_function


# ---- task_to_jsonrpc_result --------------------------------------------------


class TestTaskToJsonrpcResult:
    """Verify A2ATask -> JSON-RPC result mapping."""

    @pytest.mark.parametrize(
        ("carl_status", "spec_state"),
        [
            (A2ATaskStatus.PENDING, "submitted"),
            (A2ATaskStatus.RUNNING, "working"),
            (A2ATaskStatus.DONE, "completed"),
            (A2ATaskStatus.FAILED, "failed"),
            (A2ATaskStatus.CANCELLED, "canceled"),
        ],
    )
    def test_status_mapping(
        self,
        carl_status: A2ATaskStatus,
        spec_state: str,
    ) -> None:
        task = A2ATask(
            id="task-1",
            skill="train",
            status=carl_status,
            sender="agent-a",
            receiver="carl-studio",
        )
        result = task_to_jsonrpc_result(task)

        assert result["id"] == "task-1"
        assert result["status"]["state"] == spec_state

    def test_metadata_fields(self) -> None:
        task = A2ATask(
            id="task-2",
            skill="eval",
            sender="orchestrator",
            receiver="worker-1",
        )
        result = task_to_jsonrpc_result(task)

        assert result["metadata"]["skill"] == "eval"
        assert result["metadata"]["sender"] == "orchestrator"
        assert result["metadata"]["receiver"] == "worker-1"

    def test_result_structure_keys(self) -> None:
        task = A2ATask(id="task-3", skill="bench")
        result = task_to_jsonrpc_result(task)
        assert set(result.keys()) == {"id", "status", "metadata"}
        assert "state" in result["status"]


# ---- message_send_to_task ----------------------------------------------------


class TestMessageSendToTask:
    """Verify A2A message/send params -> CARL task kwargs."""

    def test_extracts_text_from_parts(self) -> None:
        params = {
            "message": {
                "role": "user",
                "parts": [
                    {"kind": "text", "text": "Train a model on "},
                    {"kind": "text", "text": "CIFAR-10"},
                ],
            },
            "metadata": {"skill": "train"},
        }
        kwargs = message_send_to_task(params)

        assert kwargs["skill"] == "train"
        assert kwargs["inputs"]["text"] == "Train a model on CIFAR-10"
        assert kwargs["sender"] == "user"

    def test_ignores_non_text_parts(self) -> None:
        params = {
            "message": {
                "role": "agent",
                "parts": [
                    {"kind": "image", "data": "base64..."},
                    {"kind": "text", "text": "hello"},
                ],
            },
            "metadata": {"skill": "eval"},
        }
        kwargs = message_send_to_task(params)
        assert kwargs["inputs"]["text"] == "hello"

    def test_missing_message_defaults(self) -> None:
        params: dict = {}
        kwargs = message_send_to_task(params)

        assert kwargs["skill"] == ""
        assert kwargs["inputs"]["text"] == ""
        assert kwargs["sender"] == "user"

    def test_missing_metadata_defaults(self) -> None:
        params = {
            "message": {
                "role": "assistant",
                "parts": [{"kind": "text", "text": "ok"}],
            },
        }
        kwargs = message_send_to_task(params)
        assert kwargs["skill"] == ""
        assert kwargs["sender"] == "assistant"

    def test_empty_parts_list(self) -> None:
        params = {
            "message": {"role": "user", "parts": []},
            "metadata": {"skill": "align"},
        }
        kwargs = message_send_to_task(params)
        assert kwargs["inputs"]["text"] == ""

    def test_type_error_on_non_dict(self) -> None:
        with pytest.raises(TypeError, match="params must be a dict"):
            message_send_to_task("not a dict")  # type: ignore[arg-type]


# ---- wrap_jsonrpc_response ---------------------------------------------------


class TestWrapJsonrpcResponse:
    """Verify JSON-RPC 2.0 response envelope."""

    def test_string_id(self) -> None:
        resp = wrap_jsonrpc_response("req-1", {"data": "ok"})
        assert resp["jsonrpc"] == "2.0"
        assert resp["id"] == "req-1"
        assert resp["result"] == {"data": "ok"}

    def test_integer_id(self) -> None:
        resp = wrap_jsonrpc_response(42, [1, 2, 3])
        assert resp["id"] == 42
        assert resp["result"] == [1, 2, 3]

    def test_no_error_key(self) -> None:
        resp = wrap_jsonrpc_response("x", None)
        assert "error" not in resp

    def test_null_result(self) -> None:
        resp = wrap_jsonrpc_response("x", None)
        assert resp["result"] is None


# ---- wrap_jsonrpc_error ------------------------------------------------------


class TestWrapJsonrpcError:
    """Verify JSON-RPC 2.0 error envelope."""

    def test_structure(self) -> None:
        resp = wrap_jsonrpc_error("req-2", METHOD_NOT_FOUND, "Method not found")
        assert resp["jsonrpc"] == "2.0"
        assert resp["id"] == "req-2"
        assert resp["error"]["code"] == METHOD_NOT_FOUND
        assert resp["error"]["message"] == "Method not found"

    def test_null_id(self) -> None:
        resp = wrap_jsonrpc_error(None, INTERNAL_ERROR, "Internal error")
        assert resp["id"] is None

    def test_no_result_key(self) -> None:
        resp = wrap_jsonrpc_error("x", INVALID_PARAMS, "bad params")
        assert "result" not in resp

    def test_standard_error_codes(self) -> None:
        """Verify exported error code constants match JSON-RPC 2.0 spec."""
        from carl_studio.a2a.spec import INVALID_REQUEST, PARSE_ERROR

        assert PARSE_ERROR == -32700
        assert INVALID_REQUEST == -32600
        assert METHOD_NOT_FOUND == -32601
        assert INVALID_PARAMS == -32602
        assert INTERNAL_ERROR == -32603
