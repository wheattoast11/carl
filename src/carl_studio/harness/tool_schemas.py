"""Lightweight schemas shared by chat and local harness commands."""

from __future__ import annotations

from typing import Any


def tool_schemas() -> list[dict[str, Any]]:
    """Return model-visible schemas without importing optional runtimes."""
    delegate = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "host": {"type": "string", "enum": ["codex", "claude", "opencode"]},
            "instruction": {"type": "string", "minLength": 1},
            "workdir": {"type": "string"},
            "write": {"type": "boolean"},
            "request_id": {"type": "string"},
            "timeout_s": {"type": "number"},
            "model": {"type": ["string", "null"]},
            "native_session": {"type": ["string", "null"]},
        },
        "required": ["host", "instruction"],
    }
    tools: list[dict[str, Any]] = [
        {
            "name": "list_agent_harnesses",
            "description": "Discover installed native agent harnesses.",
            "input_schema": {"type": "object", "properties": {}},
        },
        {
            "name": "delegate_agent",
            "description": "Delegate bounded work within this workspace.",
            "input_schema": delegate,
        },
    ]
    for name in ("tasks_get", "tasks_cancel", "tasks_reply", "read_agent_result"):
        properties: dict[str, Any] = {"task_id": {"type": "string"}}
        required = ["task_id"]
        if name == "tasks_reply":
            properties.update({"request_id": {"type": "string"}, "approve": {"type": "boolean"}})
            required.append("request_id")
        tools.append(
            {
                "name": name,
                "description": f"{name} for the caller's delegated task.",
                "input_schema": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                    "additionalProperties": False,
                },
            }
        )
    return tools
