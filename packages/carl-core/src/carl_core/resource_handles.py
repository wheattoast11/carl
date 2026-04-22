"""Resource-handle primitives (v0.17) — specialization of ``Vault[ResourceRef, Any]``.

Long-lived external resources (browser pages, subprocess groups, MCP sessions,
SGLang rollout engines) addressed by opaque :class:`ResourceRef` handles.
Lifecycle inherited from :class:`carl_core.vault.Vault` — the only specialization
is that ``revoke`` runs a caller-supplied ``closer(backend)`` callback so
e.g. a browser page gets ``.close()``'d and a subprocess gets ``.terminate()``'d
without the vault needing to know about either library.

Resolver chain is inherited. Future adapters (e.g. MCP-session resolvers, remote
SGLang endpoints) register resolvers for their kind without touching this file.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from carl_core.errors import ValidationError
from carl_core.vault import Vault, VaultError


__all__ = [
    "ResourceKind",
    "ResourceRef",
    "ResourceVault",
    "ResourceError",
]


ResourceKind = Literal[
    "browser_page",
    "browser_context",
    "subprocess",
    "mcp_session",
    "rollout_engine",
    "remote_agent",
]
_RESOURCE_KINDS: frozenset[str] = frozenset(
    {
        "browser_page",
        "browser_context",
        "subprocess",
        "mcp_session",
        "rollout_engine",
        "remote_agent",
    }
)


class ResourceError(VaultError):
    """Base for ``carl.resource.*`` errors.

    Inherits from :class:`VaultError` so generic ``except VaultError`` catches
    both. Direct ``except ResourceError`` continues to work as before.
    """

    code = "carl.resource"


class ResourceRef(BaseModel):
    """Opaque handle to a long-lived resource held in a :class:`ResourceVault`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    ref_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    kind: ResourceKind = Field(...)
    provider: str = Field(
        ...,
        description="Which backend provides this resource "
        "('playwright', 'subprocess', 'mcp-stdio', 'sglang', 'a2a', ...).",
    )
    uri: str = Field(
        ...,
        description="Human-debuggable reference — URL for pages, 'pid:<n>' "
        "for subprocesses, 'mcp://<server>' for MCP.",
    )
    labels: dict[str, str] = Field(
        default_factory=lambda: {},  # type: dict[str, str]
        description="Free-form tags (e.g. {'role': 'login', 'tenant': 'acme'}). "
        "Informational — not consulted by lifecycle logic.",
    )
    ttl_s: int | None = Field(default=None)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )

    def expired_at(self) -> datetime | None:
        if self.ttl_s is None:
            return None
        return self.created_at + timedelta(seconds=self.ttl_s)

    def is_expired(self, *, now: datetime | None = None) -> bool:
        expires = self.expired_at()
        if expires is None:
            return False
        current = now if now is not None else datetime.now(timezone.utc)
        return current >= expires

    def describe(self) -> dict[str, Any]:
        """Public metadata. Never exposes the backend object."""
        expires = self.expired_at()
        return {
            "ref_id": str(self.ref_id),
            "kind": self.kind,
            "provider": self.provider,
            "uri": self.uri,
            "labels": dict(self.labels),
            "ttl_s": self.ttl_s,
            "created_at": self.created_at.isoformat(),
            "expires_at": expires.isoformat() if expires is not None else None,
        }


class ResourceVault(Vault[ResourceRef, Any]):
    """Privileged vault for long-lived backends with caller-supplied closers."""

    _ref_class = ResourceRef
    _require_privileged_resolve = True
    _error_prefix: ClassVar[str] = "carl.resource"
    _error_class: ClassVar[type[VaultError]] = ResourceError

    def put(
        self,
        backend: Any,
        *,
        kind: ResourceKind,
        provider: str,
        uri: str,
        closer: Any | None = None,
        labels: dict[str, str] | None = None,
        ttl_s: int | None = None,
    ) -> ResourceRef:
        """Register a resource + optional closer callback.

        ``closer(backend) -> None`` (if given) runs at :meth:`revoke` time.
        Closer exceptions are surfaced as ``carl.resource.backend_error``.
        """
        if kind not in _RESOURCE_KINDS:
            raise ValidationError(
                f"invalid resource kind: {kind!r}",
                code=type(self)._err_code("invalid_kind"),
                context={"kind": kind, "valid": sorted(_RESOURCE_KINDS)},
            )
        self._validate_ttl(ttl_s)

        ref = ResourceRef(
            kind=kind,
            provider=provider,
            uri=uri,
            labels=dict(labels or {}),
            ttl_s=ttl_s,
        )
        self.put_value(ref, backend)
        # Store the closer on the entry's extras so revoke() can invoke it.
        if closer is not None:
            with self._lock:
                self._entries[ref.ref_id].extra["closer"] = closer
        return ref

    def revoke(self, ref: ResourceRef) -> bool:
        """Close + invalidate a handle. Runs the registered closer if any.

        Overrides the base to run `extra["closer"](backend)` before marking
        the entry revoked. Closer exceptions become ``carl.resource.backend_error``.
        The base's idempotency semantics are preserved: revoking an
        already-revoked or unknown handle returns False without raising.
        """
        cls = type(self)
        with self._lock:
            entry = self._entries.get(ref.ref_id)
            if entry is None or entry.revoked:
                return False
            closer = entry.extra.get("closer")
            backend = entry.value
            # Mark revoked FIRST so re-entrant closer calls don't loop.
            entry.revoked = True
            entry.resolved_cache = None
            entry.resolved_at = None

        if closer is None:
            return True

        try:
            closer(backend)
        except Exception as exc:
            raise cls._err(
                "backend_error",
                f"resource closer raised: {type(exc).__name__}: {exc}",
                {"exception_type": type(exc).__name__},
                cause=exc,
            ) from exc
        return True
