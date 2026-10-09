"""Public SDK dispatch with CARL's declared structured-output contracts."""

from __future__ import annotations

import json
import os
from contextvars import ContextVar
from typing import Any

from jsonschema import ValidationError, validate
from mcp.server.mcpserver import Context, MCPServer
from mcp.types import CallToolResult, InputRequiredResult, Tool

ACTIVE_CONTEXT: ContextVar[Any] = ContextVar("carl_mcp_context", default=None)
CHILD_TOOLS = frozenset(
    {
        "validate_config",
        "prepare_training",
        "encode_data",
        "interpret",
        "interpretation_feedback",
        "get_coherence_metrics",
        "list_backends",
        "list_skills",
        "list_agent_harnesses",
    }
)


def wire_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Normalize legacy schemas to MCP's object-root output contract."""
    if schema.get("type") == "string":
        return {"type": "object", "properties": {"result": schema}, "required": ["result"]}
    return {"type": "object", **schema}


class CARLMCPServer(MCPServer[Any]):
    """Validate declared outputs on the actual protocol dispatch path."""

    def tool(self, *args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("structured_output", False)
        return super().tool(*args, **kwargs)

    async def list_tools(self) -> list[Tool]:
        from carl_studio.mcp.output_schemas import OUTPUT_SCHEMAS

        tools = await super().list_tools()
        child = os.environ.get("CARL_CHILD_SCOPE") == "1"
        return [
            tool.model_copy(update={"output_schema": wire_schema(OUTPUT_SCHEMAS[tool.name])})
            if tool.name in OUTPUT_SCHEMAS
            else tool
            for tool in tools
            if not child or tool.name in CHILD_TOOLS
        ]

    async def call_tool(
        self, name: str, arguments: dict[str, Any], context: Context[Any, Any] | None = None
    ) -> CallToolResult | InputRequiredResult:
        from carl_studio.mcp.output_schemas import OUTPUT_SCHEMAS

        if os.environ.get("CARL_CHILD_SCOPE") == "1" and name not in CHILD_TOOLS:
            return CallToolResult(content=[], is_error=True)
        token = ACTIVE_CONTEXT.set(context)
        try:
            result = await super().call_tool(name, arguments, context)
        finally:
            ACTIVE_CONTEXT.reset(token)
        if isinstance(result, InputRequiredResult) or result.is_error or name not in OUTPUT_SCHEMAS:
            return result
        schema = OUTPUT_SCHEMAS[name]
        text = "".join(item.text for item in result.content if item.type == "text")
        if schema.get("type") == "string":
            value = {"result": text}
        else:
            try:
                value = json.loads(text)
            except ValueError:
                return result.model_copy(update={"is_error": True})
        try:
            validate(value, wire_schema(schema))
        except ValidationError:
            return result.model_copy(update={"is_error": True})
        return result.model_copy(update={"structured_content": value})
