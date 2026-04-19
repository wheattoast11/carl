"""Typed view of the MCP server's per-process session state.

The MCP ``server.py`` module owns a ``_session: dict[str, str]`` global that
holds auth-derived state (JWT, tier, user id, ``authenticated_at`` epoch
seconds). That shape is convenient for the existing tool bodies but loses
type information at the boundary and cannot express "no authentication has
happened yet" except by comparing empty strings.

:class:`MCPSession` is the read-only, typed snapshot we want callers to
operate on. Phase 1 only adds the view plus the two bridge helpers
(:func:`session_from_dict`, :func:`session_to_dict`) — the underlying
``_session`` dict is still the source of truth. Phase 3 (T2) will invert
the relationship: a ``SessionStore`` keyed by client id becomes the source
of truth, and the module-level dict goes away.

Design decisions
----------------
* Dataclass, not Pydantic. This primitive is on the hot path and the
  validation surface is one-line.
* ``frozen=True`` — snapshots are handed to callers; mutating them must
  go through :func:`session_to_dict` + the module write on ``server.py``.
* ``authenticated_at`` is a timezone-aware ``datetime`` (UTC) when
  present. The legacy dict carries it as a stringified epoch integer;
  conversion happens at the bridge.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone


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
    """

    jwt: str
    tier: str
    user_id: str
    authenticated_at: datetime | None

    @property
    def is_authenticated(self) -> bool:
        """True when a non-empty JWT has been accepted by ``authenticate``."""
        return bool(self.jwt)


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
    the result is bit-compatible with the module-level ``_session`` dict.
    A ``None`` datetime renders as an empty string, matching what
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


__all__ = [
    "MCPSession",
    "session_from_dict",
    "session_to_dict",
]
