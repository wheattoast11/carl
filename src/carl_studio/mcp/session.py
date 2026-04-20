"""Typed view of the MCP server's per-connection session state.

Prior to v0.7.1 the MCP ``server.py`` module owned a ``_session: dict[str, str]``
global that held auth-derived state (JWT, tier, user id, ``authenticated_at``
epoch seconds). That singleton was process-wide — every client connecting
to the same ``carl mcp serve`` process shared one auth bucket (UAT-049).

H2 inverted the ownership: :class:`MCPSession` is now the canonical shape,
each :class:`~carl_studio.mcp.connection.MCPServerConnection` owns an
instance, and tool bodies resolve the active session via
:func:`extract_session` (which prefers the FastMCP
:class:`~mcp.server.fastmcp.Context` when it is injected and falls back to
the bound connection otherwise).

Design decisions
----------------
* Dataclass, not Pydantic. This primitive is on the hot path and the
  validation surface is one-line.
* ``frozen=True`` — snapshots are immutable. Mutating the session means
  building a new :class:`MCPSession` via :meth:`MCPSession.with_` and
  assigning it back to the connection through its :attr:`session` setter.
* ``authenticated_at`` is a timezone-aware ``datetime`` (UTC) when
  present. The legacy dict carried it as a stringified epoch integer;
  :func:`session_from_dict` / :func:`session_to_dict` preserve that format
  for any caller still on the dict shape (none in-tree).
* ``jwt_expires_at`` is a POSIX timestamp (seconds) when the current JWT
  is known to expire. ``None`` means "no expiry known"; consumers should
  fall back to :data:`SESSION_MAX_AGE` for the staleness check.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - type-only
    from carl_studio.mcp.connection import MCPServerConnection


# ---------------------------------------------------------------------------
# Session staleness
# ---------------------------------------------------------------------------
#
# Sessions older than :data:`SESSION_MAX_AGE` seconds are treated as
# unauthenticated even when a JWT is still present. The constant lives here
# (not in ``server.py``) so tools and connection code can share one source
# of truth.

SESSION_MAX_AGE: float = 3600.0  # 1 hour — re-authenticate after this


@dataclass(frozen=True)
class MCPSession:
    """Immutable snapshot of the MCP server's auth state.

    Fields
    ------
    jwt
        Raw JWT string or empty string when not authenticated. Never
        logged by :meth:`to_dict` callers — consumers treat this as a
        secret.
    tier
        Canonical tier label. ``"free"`` when unauthenticated; ``"paid"``
        after a successful ``authenticate`` with any of
        ``paid`` / ``pro`` / ``enterprise`` in the profile.
    user_id
        Stable user identifier from the carl.camp profile. Empty when
        unauthenticated.
    authenticated_at
        UTC timestamp of the last successful ``authenticate`` call, or
        ``None`` when unauthenticated or when the legacy dict lacks the
        field / holds a malformed value.
    jwt_expires_at
        POSIX timestamp (seconds) of the JWT ``exp`` claim when known,
        otherwise ``None``. Consumers should combine this with
        :data:`SESSION_MAX_AGE` to bound session lifetime independent of
        the upstream issuer's expiry policy.
    """

    jwt: str = ""
    tier: str = "free"
    user_id: str = ""
    authenticated_at: datetime | None = None
    jwt_expires_at: float | None = None

    @property
    def is_authenticated(self) -> bool:
        """True when a non-empty JWT has been accepted by ``authenticate``."""
        return bool(self.jwt)

    def with_(self, **changes: Any) -> MCPSession:
        """Return a new :class:`MCPSession` with the given fields replaced.

        Convenience wrapper over :func:`dataclasses.replace` so callers can
        mutate session state without reaching for a second import. Example::

            conn.session = conn.session.with_(jwt=jwt, tier="paid")
        """
        return replace(self, **changes)


def session_from_dict(raw: dict[str, str]) -> MCPSession:
    """Build an :class:`MCPSession` from the legacy ``_session`` dict.

    Missing keys default to empty strings. An ``authenticated_at`` that
    is missing, empty, or not a base-10 integer degrades cleanly to
    ``None`` — callers should treat that as "unknown auth time" rather
    than failing the whole read.
    """
    jwt = raw.get("jwt", "")
    tier = raw.get("tier", "free") or "free"
    user_id = raw.get("user_id", "")

    at_raw = raw.get("authenticated_at", "")
    authenticated_at: datetime | None
    if at_raw:
        try:
            authenticated_at = datetime.fromtimestamp(int(at_raw), tz=timezone.utc)
        except (ValueError, TypeError, OverflowError, OSError):
            authenticated_at = None
    else:
        authenticated_at = None

    return MCPSession(
        jwt=jwt,
        tier=tier,
        user_id=user_id,
        authenticated_at=authenticated_at,
    )


def session_to_dict(session: MCPSession) -> dict[str, str]:
    """Reverse of :func:`session_from_dict`.

    ``authenticated_at`` is re-encoded as a stringified epoch integer so
    the result is bit-compatible with the legacy module-level ``_session``
    dict. A ``None`` datetime renders as an empty string, matching what
    ``_require_tier`` treats as "never authenticated".
    """
    if session.authenticated_at is not None:
        at_str = str(int(session.authenticated_at.timestamp()))
    else:
        at_str = ""
    return {
        "jwt": session.jwt,
        "tier": session.tier,
        "user_id": session.user_id,
        "authenticated_at": at_str,
    }


def extract_session(conn: "MCPServerConnection | None") -> Any:
    """Best-effort lookup of a ``ServerSession``-like object on the connection.

    This is the canonical helper used by both :mod:`carl_studio.mcp.elicitation`
    and :mod:`carl_studio.mcp.sampling` to reach the active FastMCP
    ``ServerSession`` (the wire-level send/receive seam, **not**
    :class:`MCPSession` which is carl-studio's auth-state snapshot).

    Resolution order:

    1. ``conn._session_override`` — test-harness hook that lets unit tests
       attach a stub ``ServerSession`` without standing up a live FastMCP
       request context.
    2. ``conn.fastmcp.get_context().session`` — the public FastMCP 1.10+
       accessor for the in-flight request's ``ServerSession``.

    Returns ``None`` when no session is resolvable (connection is ``None``,
    FastMCP absent, or we are outside a request context).
    """
    if conn is None:
        return None

    override = getattr(conn, "_session_override", None)
    if override is not None:
        return override

    fastmcp = getattr(conn, "fastmcp", None)
    if fastmcp is None:
        return None

    ctx_getter = getattr(fastmcp, "get_context", None)
    if callable(ctx_getter):
        try:
            ctx = ctx_getter()
        except Exception:
            ctx = None
        if ctx is not None:
            session = getattr(ctx, "session", None)
            if session is not None:
                return session

    return None


__all__ = [
    "MCPSession",
    "SESSION_MAX_AGE",
    "extract_session",
    "session_from_dict",
    "session_to_dict",
]
