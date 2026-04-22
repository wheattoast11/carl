"""Resource-handle primitives (v0.16.1 — Stage C-2).

Extends the capability-constrained handle runtime to long-lived external
resources: browser pages, subprocess groups, MCP server connections,
SGLang rollout engines — anything that's stateful, addressable, and
closeable.

Shape mirrors :class:`carl_core.secrets.SecretRef` +
:class:`carl_core.data_handles.DataRef` by design. One grammar across all
three; Carl doesn't need to learn three mental models.

What lives here (carl-core, dependency-free):

* :class:`ResourceRef` — frozen handle with kind / uri / provider /
  pid-or-session-id / ttl metadata.
* :class:`ResourceVault` — thread-safe registry. Stores the opaque
  "backend" object (e.g. a Playwright Page, a subprocess.Popen) behind
  the ref but never serializes it.
* Error codes under ``carl.resource.*``:

  - ``carl.resource.not_found`` — handle ref_id unknown
  - ``carl.resource.revoked`` — explicitly closed
  - ``carl.resource.expired`` — TTL elapsed
  - ``carl.resource.invalid_kind`` — invalid kind
  - ``carl.resource.backend_error`` — backend raised during close

Above in carl_studio: ``BrowserToolkit``, ``SubprocessToolkit``, etc.
consume this vault; they own the optional deps (Playwright, psutil) and
the agent-callable method surfaces.
"""

from __future__ import annotations

import threading
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from carl_core.errors import CARLError, ValidationError


__all__ = [
    "ResourceKind",
    "ResourceRef",
    "ResourceVault",
    "ResourceError",
]


ResourceKind = Literal[
    "browser_page",     # a Playwright / puppeteer page
    "browser_context",  # a Playwright browser context (cookies+storage)
    "subprocess",       # a spawned OS subprocess
    "mcp_session",      # a live MCP stdio / SSE session
    "rollout_engine",   # an SGLang / TRL rollout engine reference
    "remote_agent",     # an A2A remote-agent session handle
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


class ResourceError(CARLError):
    """Base for all ``carl.resource.*`` errors."""

    code = "carl.resource"


class ResourceRef(BaseModel):
    """Opaque handle to a long-lived resource held in a :class:`ResourceVault`.

    The handle carries provider metadata (e.g. ``provider="playwright"``,
    ``provider="subprocess"``) so toolkit layers can dispatch on it, but
    the backend object itself never crosses a tool-call boundary.
    """

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
        default_factory=dict,
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


class _ResourceEntry:
    __slots__ = ("backend", "closer", "revoked")

    def __init__(self, backend: Any, closer: Any) -> None:
        self.backend: Any = backend
        self.closer: Any = closer  # optional callable(backend) -> None
        self.revoked: bool = False


class ResourceVault:
    """Thread-safe KV of ``ResourceRef → backend``.

    The backend is stored opaque. Toolkit layers retrieve the backend via
    :meth:`resolve` (privileged) and dispatch on it. Closing a ref runs
    the caller-supplied ``closer`` (if provided) so e.g. a browser page
    gets ``.close()``'d and a subprocess gets ``.terminate()``'d on
    revoke — without the vault knowing about either library.
    """

    def __init__(self) -> None:
        self._entries: dict[uuid.UUID, _ResourceEntry] = {}
        self._refs: dict[uuid.UUID, ResourceRef] = {}
        self._lock = threading.RLock()

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
        """Register a resource. ``closer`` (if any) runs on :meth:`revoke`.

        ``closer`` takes the backend and returns None; exceptions raised
        during close get swallowed into :class:`ResourceError` with code
        ``carl.resource.backend_error`` so one broken resource doesn't
        kill the vault.
        """
        if kind not in _RESOURCE_KINDS:
            raise ValidationError(
                f"invalid resource kind: {kind!r}",
                code="carl.resource.invalid_kind",
                context={"kind": kind, "valid": sorted(_RESOURCE_KINDS)},
            )
        if ttl_s is not None and ttl_s < 1:
            raise ValidationError(
                f"ttl_s must be a positive int or None, got {ttl_s!r}",
                code="carl.resource.invalid_kind",
                context={"ttl_s": ttl_s},
            )

        ref = ResourceRef(
            kind=kind,
            provider=provider,
            uri=uri,
            labels=dict(labels or {}),
            ttl_s=ttl_s,
        )
        entry = _ResourceEntry(backend=backend, closer=closer)
        with self._lock:
            self._entries[ref.ref_id] = entry
            self._refs[ref.ref_id] = ref
        return ref

    def resolve(self, ref: ResourceRef, *, privileged: bool = False) -> Any:
        """Return the backend object for a ref. Privileged-only.

        Toolkit layers call this with ``privileged=True`` when they're
        about to act against the backend. The agent never gets a backend
        object directly; the toolkit mediates.
        """
        if not privileged:
            raise ResourceError(
                "ResourceVault.resolve() requires privileged=True.",
                code="carl.resource.unauthorized_resolve",
                context={"ref_id": str(ref.ref_id)},
            )
        with self._lock:
            entry = self._entries.get(ref.ref_id)
            current = self._refs.get(ref.ref_id)
            if entry is None or current is None:
                raise ResourceError(
                    f"unknown resource handle: {ref.ref_id}",
                    code="carl.resource.not_found",
                    context={"ref_id": str(ref.ref_id)},
                )
            if entry.revoked:
                raise ResourceError(
                    f"resource handle {ref.ref_id} was revoked",
                    code="carl.resource.revoked",
                    context={"ref_id": str(ref.ref_id)},
                )
            if current.is_expired():
                self._close_entry(entry)
                raise ResourceError(
                    f"resource handle {ref.ref_id} has expired",
                    code="carl.resource.expired",
                    context={"ref_id": str(ref.ref_id)},
                )
            return entry.backend

    def revoke(self, ref: ResourceRef) -> bool:
        """Close + invalidate a handle. Runs the registered closer if any."""
        with self._lock:
            entry = self._entries.get(ref.ref_id)
            if entry is None or entry.revoked:
                return False
            self._close_entry(entry)
        return True

    def exists(self, ref: ResourceRef) -> bool:
        with self._lock:
            entry = self._entries.get(ref.ref_id)
            if entry is None or entry.revoked:
                return False
        return not ref.is_expired()

    def list_refs(self) -> list[ResourceRef]:
        now = datetime.now(timezone.utc)
        with self._lock:
            refs: list[ResourceRef] = []
            for ref_id, ref in self._refs.items():
                entry = self._entries.get(ref_id)
                if entry is None or entry.revoked:
                    continue
                if ref.is_expired(now=now):
                    continue
                refs.append(ref)
            return refs

    def __len__(self) -> int:
        with self._lock:
            return sum(
                1
                for ref_id, ref in self._refs.items()
                if (entry := self._entries.get(ref_id)) is not None
                and not entry.revoked
                and not ref.is_expired()
            )

    def _close_entry(self, entry: _ResourceEntry) -> None:
        """Mark revoked + run closer. Closer exceptions become ResourceError."""
        entry.revoked = True
        if entry.closer is None:
            return
        try:
            entry.closer(entry.backend)
        except Exception as exc:
            raise ResourceError(
                f"resource closer raised: {type(exc).__name__}: {exc}",
                code="carl.resource.backend_error",
                context={"exception_type": type(exc).__name__},
                cause=exc,
            ) from exc
