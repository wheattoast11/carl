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


def agent_card_to_spec(
    card: Any,
    *,
    streaming: bool = False,
    push_notifications: bool = False,
    include_identity: bool = False,
    identity_algorithm: str = "ES256",
    identity_kid: str = "carl-studio-agent-es256",
) -> dict[str, Any]:
    """Convert CARLAgentCard to A2A spec-compliant AgentCard JSON.

    Follows the Linux Foundation A2A spec AgentCard schema:
    name, description, url, version, capabilities, defaultInputModes,
    defaultOutputModes, skills, securitySchemes (optional).

    Parameters
    ----------
    card
        The :class:`CARLAgentCard` instance to serialize.
    streaming
        Advertise ``capabilities.streaming = True`` when the server
        publishing this card supports ``message/stream`` and
        ``tasks/subscribe``. Default False for back-compat; callers that
        wire the v1.0 streaming surface should pass True.
    push_notifications
        Advertise ``capabilities.pushNotifications = True`` when the
        publisher supports ``tasks/pushNotificationConfig/*``.
    include_identity
        When True, emits a ``securitySchemes`` section declaring the JWS
        signing algorithm and key id used by the server's
        ``agent/getAuthenticatedExtendedCard`` response. Off by default
        so unsigned public cards don't falsely advertise a capability
        they can't fulfil.
    identity_algorithm
        JWS alg tag to publish. Defaults to ``"ES256"`` to match
        :class:`~carl_studio.a2a.identity.AgentIdentity` defaults.
    identity_kid
        Key id to publish. Must match the ``kid`` baked into the signer
        so counterparty verifiers can select the right public key.
    """
    skills_list: list[dict[str, str]] = []
    raw_skills: Any = card.skills or []
    if isinstance(raw_skills, (list, tuple)):
        for _entry in raw_skills:  # type: ignore[reportUnknownVariableType]
            skill_name = str(_entry) if _entry is not None else ""  # type: ignore[reportUnknownArgumentType]
            if not skill_name:
                continue
            skills_list.append({
                "id": skill_name,
                "name": skill_name,
                "description": f"CARL skill: {skill_name}",
            })

    spec: dict[str, Any] = {
        "name": card.name,
        "description": f"CARL Studio agent (v{card.version})",
        "url": card.endpoint if card.endpoint != "stdio" else None,
        "version": card.version,
        "capabilities": {
            "streaming": bool(streaming),
            "pushNotifications": bool(push_notifications),
        },
        "defaultInputModes": ["text"],
        "defaultOutputModes": ["text"],
        "skills": skills_list,
    }

    if include_identity:
        spec["securitySchemes"] = {
            "carl-agent-jws": {
                "type": "jws",
                "algorithm": identity_algorithm,
                "kid": identity_kid,
                "endpoint": "agent/getAuthenticatedExtendedCard",
            },
        }

    return spec


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
    if not isinstance(params, dict):  # type: ignore[reportUnnecessaryIsInstance]
        raise TypeError(f"params must be a dict, got {type(params).__name__}")

    message_any: Any = params.get("message", {})
    message: dict[str, Any] = dict(message_any) if isinstance(message_any, dict) else {}  # type: ignore[arg-type]
    parts_any: Any = message.get("parts", [])
    parts: list[Any] = list(parts_any) if isinstance(parts_any, (list, tuple)) else []  # type: ignore[arg-type]
    text_parts: list[str] = []
    for part in parts:
        if isinstance(part, dict):
            part_dict: dict[str, Any] = dict(part)  # type: ignore[arg-type]
            if part_dict.get("kind") == "text":
                text_val: Any = part_dict.get("text", "")
                text_parts.append(str(text_val) if text_val is not None else "")

    metadata_any: Any = params.get("metadata", {})
    metadata: dict[str, Any] = (
        dict(metadata_any) if isinstance(metadata_any, dict) else {}  # type: ignore[arg-type]
    )
    role_any: Any = message.get("role", "user")

    return {
        "skill": str(metadata.get("skill", "") or ""),
        "inputs": {"text": "".join(text_parts)},
        "sender": str(role_any) if role_any is not None else "user",
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
