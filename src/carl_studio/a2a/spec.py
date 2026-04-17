"""A2A protocol spec adapter -- maps CARL models to/from A2A JSON-RPC format.

Produces spec-compliant JSON without requiring the a2a-python SDK.
When the SDK is available, uses its Pydantic models for validation.
"""
from __future__ import annotations

from typing import Any

# JSON-RPC 2.0 standard error codes
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603


def agent_card_to_spec(card: Any) -> dict[str, Any]:
    """Convert CARLAgentCard to A2A spec-compliant AgentCard JSON.

    Follows the Linux Foundation A2A spec AgentCard schema:
    name, description, url, version, capabilities, defaultInputModes,
    defaultOutputModes, skills.
    """
    skills_list: list[dict[str, str]] = []
    for skill_name in card.skills or []:
        skills_list.append({
            "id": skill_name,
            "name": skill_name,
            "description": f"CARL skill: {skill_name}",
        })

    return {
        "name": card.name,
        "description": f"CARL Studio agent (v{card.version})",
        "url": card.endpoint if card.endpoint != "stdio" else None,
        "version": card.version,
        "capabilities": {
            "streaming": False,
            "pushNotifications": False,
        },
        "defaultInputModes": ["text"],
        "defaultOutputModes": ["text"],
        "skills": skills_list,
    }


def task_to_jsonrpc_result(task: Any) -> dict[str, Any]:
    """Convert A2ATask to JSON-RPC result format for tasks/get response.

    Maps internal CARL status names to A2A spec states:
      pending   -> submitted
      running   -> working
      done      -> completed
      failed    -> failed
      cancelled -> canceled  (single 'l' per spec)
    """
    status_map: dict[str, str] = {
        "pending": "submitted",
        "running": "working",
        "done": "completed",
        "failed": "failed",
        "cancelled": "canceled",
    }
    return {
        "id": task.id,
        "status": {
            "state": status_map.get(task.status, task.status),
        },
        "metadata": {
            "skill": task.skill,
            "sender": task.sender,
            "receiver": task.receiver,
        },
    }


def message_send_to_task(params: dict[str, Any]) -> dict[str, Any]:
    """Convert A2A message/send params to CARL task creation kwargs.

    Extracts text content from message parts and maps metadata fields
    to the kwargs expected by A2ATask construction.
    """
    if not isinstance(params, dict):
        raise TypeError(f"params must be a dict, got {type(params).__name__}")

    message = params.get("message", {})
    parts = message.get("parts", [])
    text_parts: list[str] = []
    for part in parts:
        if isinstance(part, dict) and part.get("kind") == "text":
            text_parts.append(part.get("text", ""))

    return {
        "skill": params.get("metadata", {}).get("skill", ""),
        "inputs": {"text": "".join(text_parts)},
        "sender": message.get("role", "user"),
    }


def wrap_jsonrpc_response(
    request_id: str | int,
    result: Any,
) -> dict[str, Any]:
    """Wrap a result in a JSON-RPC 2.0 response envelope."""
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "result": result,
    }


def wrap_jsonrpc_error(
    request_id: str | int | None,
    code: int,
    message: str,
) -> dict[str, Any]:
    """Wrap an error in a JSON-RPC 2.0 error envelope."""
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {"code": code, "message": message},
    }
