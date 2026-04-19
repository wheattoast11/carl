"""CARL Studio MCP Server.

Phase 1 of the carl_core.connection port surfaces the existing FastMCP
server behind :class:`MCPServerConnection` — a typed facade that hooks the
server's lifecycle and per-tool telemetry onto the Connection primitive.
The legacy ``mcp`` module-level singleton and the ``_session`` dict are
unchanged for back-compat; Phase 3 (T2) migrates session ownership onto
the connection.

Phase 3 / MCP 2025-11-25 adds the async-task surface
(:class:`MCPTaskStore`, :func:`async_task`), client-side elicitation
(:func:`elicit`, :func:`elicits_on_missing`), client-side sampling
(:func:`sample`), and per-tool output-schema registration
(:data:`OUTPUT_SCHEMAS`, :func:`register_output_schemas`).
"""

from __future__ import annotations

from carl_studio.mcp.connection import MCPServerConnection
from carl_studio.mcp.elicitation import (
    ElicitationRequest,
    ElicitationResponse,
    elicit,
    elicits_on_missing,
)
from carl_studio.mcp.output_schemas import (
    OUTPUT_SCHEMAS,
    register_output_schemas,
)
from carl_studio.mcp.sampling import (
    SamplingCostEvent,
    SamplingRequest,
    SamplingResponse,
    sample,
)
from carl_studio.mcp.session import (
    MCPSession,
    session_from_dict,
    session_to_dict,
)
from carl_studio.mcp.tasks import (
    MCPTask,
    MCPTaskStore,
    async_task,
)

# ``server.py`` imports are heavier (they execute @mcp.tool decorators);
# keep them at the end so the lighter primitives above are available
# even when FastMCP isn't importable.
from carl_studio.mcp.server import (
    bind_connection,
    get_bound_connection,
    mcp,
)

__all__ = [
    # Core MCP surface
    "mcp",
    "MCPServerConnection",
    "MCPSession",
    "bind_connection",
    "get_bound_connection",
    "session_from_dict",
    "session_to_dict",
    # Async tasks
    "MCPTask",
    "MCPTaskStore",
    "async_task",
    # Elicitation
    "ElicitationRequest",
    "ElicitationResponse",
    "elicit",
    "elicits_on_missing",
    # Sampling
    "SamplingRequest",
    "SamplingResponse",
    "SamplingCostEvent",
    "sample",
    # Output schemas
    "OUTPUT_SCHEMAS",
    "register_output_schemas",
]
