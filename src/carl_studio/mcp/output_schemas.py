"""JSON Schemas for the 16 built-in MCP tools.

MCP 2025-11-25 lets each tool declare an ``output_schema`` so typed clients
can validate the response shape. Every tool currently returns a
JSON-stringified dict (or raw string) — the schemas below describe the
*parsed* JSON shape.

Rules
-----
* Schemas use JSON Schema draft-07 semantics (the default the MCP SDK
  emits). We keep them minimal: ``type``, ``properties``, ``required``.
* Unknown-shape payloads (e.g. ``generate_bundle`` returns the raw bundle
  source) use ``{"type": "string"}`` without properties.
* Error envelopes (``{"error": "...", ...}``) are captured as
  alternatives via ``oneOf`` when a tool deliberately switches between
  a happy-path object and an error object. This mirrors what the tools
  *actually* emit today so round-tripping existing responses validates.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from mcp.server.fastmcp import FastMCP


# ---------------------------------------------------------------------------
# Schema building blocks
# ---------------------------------------------------------------------------


_TIER_ERROR_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["error", "current_tier"],
    "properties": {
        "error": {"type": "string"},
        "current_tier": {"type": "string"},
        "upgrade": {"type": "string"},
        "hint": {"type": "string"},
    },
    "additionalProperties": True,
}


def _one_of_tier_error(happy: dict[str, Any]) -> dict[str, Any]:
    """Wrap a happy-path schema as ``oneOf(happy, tier_error)``."""
    return {"oneOf": [happy, _TIER_ERROR_SCHEMA]}


# ---------------------------------------------------------------------------
# Per-tool schemas
# ---------------------------------------------------------------------------


_START_TRAINING: dict[str, Any] = {
    "type": "object",
    "required": ["run_id", "phase", "model", "method"],
    "properties": {
        "run_id": {"type": "string"},
        "phase": {"type": "string"},
        "hub_job_id": {"type": ["string", "null"]},
        "model": {"type": "string"},
        "method": {"type": "string"},
    },
    "additionalProperties": False,
}


_GET_RUN_STATUS: dict[str, Any] = {
    "type": "object",
    "required": ["run_id", "status"],
    "properties": {
        "run_id": {"type": "string"},
        "status": {},
    },
    "additionalProperties": True,
}


_GET_LOGS: dict[str, Any] = {
    "type": "object",
    "required": ["run_id", "lines"],
    "properties": {
        "run_id": {"type": "string"},
        "lines": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "additionalProperties": False,
}


_OBSERVE_NOW: dict[str, Any] = {
    "type": "object",
    "properties": {
        "status": {"type": "string"},
        "diagnosis": {"type": "string"},
        "signals": {"type": "object"},
        "recommendations": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "additionalProperties": True,
}


_GET_COHERENCE_METRICS: dict[str, Any] = {
    "type": "object",
    "required": [
        "kappa",
        "sigma",
        "kappa_x_sigma",
        "t_star",
        "embedding_dim",
    ],
    "properties": {
        "kappa": {"type": "number"},
        "sigma": {"type": "number"},
        "kappa_x_sigma": {"type": "number"},
        "t_star": {"type": "number"},
        "embedding_dim": {"type": "integer"},
    },
    "additionalProperties": False,
}


_STOP_TRAINING: dict[str, Any] = {
    "type": "object",
    "required": ["run_id", "status"],
    "properties": {
        "run_id": {"type": "string"},
        "status": {"type": "string"},
    },
    "additionalProperties": False,
}


_LIST_BACKENDS: dict[str, Any] = {
    "type": "object",
    "required": ["backends"],
    "properties": {
        "backends": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["name", "type", "description"],
                "properties": {
                    "name": {"type": "string"},
                    "type": {"type": "string"},
                    "description": {"type": "string"},
                },
                "additionalProperties": False,
            },
        },
    },
    "additionalProperties": False,
}


_VALIDATE_CONFIG: dict[str, Any] = {
    "oneOf": [
        {
            "type": "object",
            "required": ["valid"],
            "properties": {
                "valid": {"type": "boolean", "const": True},
                "model": {"type": "string"},
                "method": {"type": "string"},
                "compute": {"type": "string"},
            },
            "additionalProperties": False,
        },
        {
            "type": "object",
            "required": ["valid", "error"],
            "properties": {
                "valid": {"type": "boolean", "const": False},
                "error": {"type": "string"},
            },
            "additionalProperties": False,
        },
    ]
}


_GENERATE_BUNDLE: dict[str, Any] = {"type": "string"}


_AUTHENTICATE: dict[str, Any] = {
    "oneOf": [
        {
            "type": "object",
            "required": ["authenticated", "tier", "user_id", "features_available"],
            "properties": {
                "authenticated": {"type": "boolean", "const": True},
                "tier": {"type": "string"},
                "user_id": {"type": "string"},
                "features_available": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "additionalProperties": False,
        },
        {
            "type": "object",
            "required": ["authenticated", "error"],
            "properties": {
                "authenticated": {"type": "boolean", "const": False},
                "error": {"type": "string"},
                "hint": {"type": "string"},
            },
            "additionalProperties": True,
        },
    ]
}


_GET_TIER_STATUS: dict[str, Any] = {
    "type": "object",
    "required": ["tier", "authenticated", "user_id", "features"],
    "properties": {
        "tier": {"type": "string"},
        "authenticated": {"type": "boolean"},
        "user_id": {"type": "string"},
        "features": {
            "type": "object",
            "additionalProperties": {"type": "boolean"},
        },
    },
    "additionalProperties": False,
}


_GET_PROJECT_STATE: dict[str, Any] = {
    "type": "object",
    "required": ["project", "pending_sync"],
    "properties": {
        "project": {"type": "object"},
        "last_run": {"type": ["object", "null"]},
        "last_eval": {"type": ["object", "null"]},
        "pending_sync": {"type": "integer"},
    },
    "additionalProperties": True,
}


_RUN_SKILL: dict[str, Any] = _one_of_tier_error(
    {
        "type": "object",
        "properties": {
            "success": {"type": "boolean"},
            "skill_name": {"type": "string"},
            "output": {},
            "error": {"type": "string"},
        },
        "additionalProperties": True,
    }
)


_LIST_SKILLS: dict[str, Any] = {
    "oneOf": [
        {
            "type": "object",
            "required": ["skills"],
            "properties": {
                "skills": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["name", "badge", "description", "requires_tier"],
                        "properties": {
                            "name": {"type": "string"},
                            "badge": {"type": "string"},
                            "description": {"type": "string"},
                            "requires_tier": {"type": "string"},
                        },
                        "additionalProperties": False,
                    },
                },
            },
            "additionalProperties": False,
        },
        {
            "type": "object",
            "required": ["error"],
            "properties": {"error": {"type": "string"}},
            "additionalProperties": False,
        },
    ]
}


_GET_AGENT_CARD: dict[str, Any] = {
    "type": "object",
    "additionalProperties": True,
}


_DISPATCH_A2A_TASK: dict[str, Any] = _one_of_tier_error(
    {
        "type": "object",
        "required": ["task_id", "status"],
        "properties": {
            "task_id": {"type": ["string", "null"]},
            "status": {"type": "string"},
            "skill": {"type": "string"},
            "sender": {"type": "string"},
            "error": {"type": "string"},
        },
        "additionalProperties": True,
    }
)


_SYNC_DATA: dict[str, Any] = _one_of_tier_error(
    {
        "type": "object",
        "required": ["direction", "entity_types", "status"],
        "properties": {
            "direction": {"type": "string"},
            "entity_types": {
                "type": "array",
                "items": {"type": "string"},
            },
            "status": {"type": "string"},
            "results": {},
            "error": {"type": "string"},
        },
        "additionalProperties": True,
    }
)


_SUBMIT_ASYNC_TRAINING: dict[str, Any] = {
    "type": "object",
    "required": ["task_id", "status"],
    "properties": {
        "task_id": {"type": "string"},
        "status": {"type": "string"},
        "tool_name": {"type": "string"},
        "submitted_at": {"type": "string"},
    },
    "additionalProperties": True,
}


_TASKS_GET: dict[str, Any] = {
    "oneOf": [
        {
            "type": "object",
            "required": ["task_id", "tool_name", "status", "submitted_at"],
            "properties": {
                "task_id": {"type": "string"},
                "tool_name": {"type": "string"},
                "params_hash": {"type": "string"},
                "status": {
                    "type": "string",
                    "enum": [
                        "pending",
                        "running",
                        "completed",
                        "failed",
                        "cancelled",
                    ],
                },
                "submitted_at": {"type": "string"},
                "completed_at": {"type": ["string", "null"]},
                "result": {},
                "error": {"type": ["object", "null"]},
                "progress": {"type": "number"},
            },
            "additionalProperties": True,
        },
        {
            "type": "object",
            "required": ["error"],
            "properties": {
                "error": {"type": "string"},
                "task_id": {"type": "string"},
            },
            "additionalProperties": False,
        },
    ]
}


_TASKS_CANCEL: dict[str, Any] = {
    "type": "object",
    "required": ["cancelled"],
    "properties": {
        "cancelled": {"type": "boolean"},
        "task_id": {"type": "string"},
        "reason": {"type": "string"},
        "status": {"type": "string"},
    },
    "additionalProperties": False,
}


# ---------------------------------------------------------------------------
# Public registry
# ---------------------------------------------------------------------------


OUTPUT_SCHEMAS: dict[str, dict[str, Any]] = {
    # Core 9
    "start_training": _START_TRAINING,
    "get_run_status": _GET_RUN_STATUS,
    "get_logs": _GET_LOGS,
    "observe_now": _OBSERVE_NOW,
    "get_coherence_metrics": _GET_COHERENCE_METRICS,
    "stop_training": _STOP_TRAINING,
    "list_backends": _LIST_BACKENDS,
    "validate_config": _VALIDATE_CONFIG,
    "generate_bundle": _GENERATE_BUNDLE,
    # Auth + tier + skills + a2a + sync
    "authenticate": _AUTHENTICATE,
    "get_tier_status": _GET_TIER_STATUS,
    "get_project_state": _GET_PROJECT_STATE,
    "run_skill": _RUN_SKILL,
    "list_skills": _LIST_SKILLS,
    "get_agent_card": _GET_AGENT_CARD,
    "dispatch_a2a_task": _DISPATCH_A2A_TASK,
    "sync_data": _SYNC_DATA,
    # New in 2025-11-25 surface
    "submit_async_training": _SUBMIT_ASYNC_TRAINING,
    "tasks_get": _TASKS_GET,
    "tasks_cancel": _TASKS_CANCEL,
}


# The 16 tools that existed before the 2025-11-25 bump. Used by tests to
# assert the schema registry fully covers the legacy surface.
LEGACY_TOOL_NAMES: tuple[str, ...] = (
    "start_training",
    "get_run_status",
    "get_logs",
    "observe_now",
    "get_coherence_metrics",
    "stop_training",
    "list_backends",
    "validate_config",
    "generate_bundle",
    "authenticate",
    "get_tier_status",
    "get_project_state",
    "run_skill",
    "list_skills",
    "get_agent_card",
    "dispatch_a2a_task",
    "sync_data",
)


def register_output_schemas(mcp_instance: "FastMCP | Any") -> dict[str, dict[str, Any]]:
    """Attach output schemas to every registered tool on ``mcp_instance``.

    FastMCP 1.10+ accepts an ``output_schema`` kwarg on ``@mcp.tool()``.
    Our existing ``server.py`` registers tools without that kwarg because
    the decorators ran at import time against an older SDK signature; we
    patch the schemas on after the fact.

    Strategy
    --------
    1. If the tool manager has a ``set_output_schema(name, schema)`` API,
       call it. (Reserved for future SDK versions.)
    2. Otherwise, attach the schemas to a module-level
       ``_carl_output_schemas`` attribute so clients/tests can retrieve
       them via introspection.

    Returns the dict of schemas actually attached (excludes tools the
    instance doesn't know about).
    """
    attached: dict[str, dict[str, Any]] = {}
    manager = getattr(mcp_instance, "_tool_manager", None)

    # Helper: find the tool object by name through whichever API is present.
    def _get_tool(name: str) -> Any:
        if manager is None:
            return None
        getter = getattr(manager, "get_tool", None)
        if callable(getter):
            try:
                return getter(name)
            except Exception:
                return None
        tools = getattr(manager, "_tools", None)
        if isinstance(tools, dict):
            typed_tools: dict[str, Any] = tools  # type: ignore[assignment]
            return typed_tools.get(name)
        return None

    for name, schema in OUTPUT_SCHEMAS.items():
        tool_obj = _get_tool(name)
        if tool_obj is None:
            # Tool not registered on this FastMCP instance — skip silently.
            continue
        # Prefer the official attribute if the SDK exposes it.
        if hasattr(tool_obj, "output_schema"):
            try:
                setattr(tool_obj, "output_schema", schema)
            except Exception:
                pass
        # Always stash our own attribute so downstream code and tests have
        # a stable surface independent of SDK version.
        try:
            setattr(tool_obj, "_carl_output_schema", schema)
        except Exception:
            pass
        attached[name] = schema

    # Mirror onto the FastMCP instance itself for global introspection.
    existing = getattr(mcp_instance, "_carl_output_schemas", None)
    if isinstance(existing, dict):
        typed_existing: dict[str, Any] = existing  # type: ignore[assignment]
        typed_existing.update(attached)
    else:
        try:
            setattr(mcp_instance, "_carl_output_schemas", dict(attached))
        except Exception:
            pass
    return attached


__all__ = [
    "LEGACY_TOOL_NAMES",
    "OUTPUT_SCHEMAS",
    "register_output_schemas",
]
