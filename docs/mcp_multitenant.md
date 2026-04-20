---
last_updated: 2026-04-20
author: Tej Desai + Claude Opus 4.7 (1M context)
applies_to: v0.8.0
---

# MCP per-request session state (v0.7.1)

Before v0.7.1, `carl_studio.mcp.server` owned a module-level
`_session: dict[str, str]` that held auth-derived state (JWT, tier, user id,
`authenticated_at`). That singleton was process-wide — every client
connecting to the same `carl mcp serve` instance shared one auth bucket,
making multi-tenant deployment unsafe (UAT-049). A single `authenticate`
call leaked into every other in-flight tool invocation.

v0.7.1 inverts the ownership. The canonical shape is
`carl_studio.mcp.session.MCPSession` — an immutable dataclass — and each
`MCPServerConnection` owns one. Tool bodies resolve the active session via
`carl_studio.mcp.session.extract_session`, which prefers the FastMCP
`Context` dependency-injected by the runtime and falls back to the bound
connection's session when a `Context` is unavailable (tests, custom hosts).

## Migration

- **Single-process operators**: no migration required. One
  `MCPServerConnection` means one `MCPSession` — behaviour is identical
  to pre-v0.7.1 except the session is now per-connection-instance rather
  than module-global.
- **Multi-tenant deployments**: now supported. Construct one
  `MCPServerConnection` per client (thread, worker, tenant context). Each
  gets an isolated session; `authenticate` no longer cross-contaminates
  concurrent callers.

## Mutating session state

`MCPSession` is `frozen=True`. Use the `with_` helper:

```python
from carl_studio.mcp.connection import MCPServerConnection

conn = MCPServerConnection(...)
conn.session = conn.session.with_(jwt=jwt, tier="paid", user_id=user_id)
```

Under the hood `with_` is `dataclasses.replace` — the convenience is
preserving the single-import ergonomics on the hot path.

## FastMCP Context DI

When FastMCP binds a `Context` to a tool, `extract_session` prefers it
over the module-bound connection. This is the production path on
authenticated tools — `authenticate`, `get_tier_status`, `run_skill`,
`dispatch_a2a_task`, `sync_data`. The fallback to
`conn._session_override` and `conn.fastmcp.get_context().session`
preserves testability without requiring a live FastMCP request context.

Session lifetime is bounded by
`carl_studio.mcp.session.SESSION_MAX_AGE` (3600s default). Consumers
that care about JWT-level expiry should additionally check
`MCPSession.jwt_expires_at` (POSIX timestamp of the JWT `exp` claim
when known).
