"""Data-handle primitives (v0.17) — specialization of ``Vault[DataRef, bytes]``.

Extends the unified handle-runtime to arbitrary data payloads: inline bytes,
filesystem paths (lazy-read), streamed iterators (incrementally buffered),
deferred queries, remote URLs, and derived refs. Registry lifecycle is
inherited from :class:`carl_core.vault.Vault`; the rich offset-addressable
read + lazy-hash + stream-buffer semantics stay here because they don't
cleanly map onto the generic ``resolve(ref) -> value`` shape.

Resolver chain is inherited. `query` / `url` / `derived` refs can register
their own resolvers via :meth:`DataVault.register_resolver` — the offset-
read path stays a ``carl.data.backend_unavailable`` for external refs,
while :meth:`resolve` (the generic entry point) honors the resolver chain.

Non-privileged by default. Data is the "agent reads it" case; use
:class:`~carl_core.secrets.SecretVault` for values the agent shouldn't see.
"""

from __future__ import annotations

import hashlib
import mimetypes
import uuid
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from carl_core.errors import ValidationError
from carl_core.vault import Vault, VaultError


__all__ = [
    "DataKind",
    "DataRef",
    "DataVault",
    "DataError",
    "content_type_from_path",
]


DataKind = Literal[
    "bytes",    # inline bytes held in vault memory
    "file",     # path on disk (lazy-read, not slurped at open)
    "stream",   # iterator yielding chunks (consumable, incrementally buffered)
    "query",    # deferred query (URI holds the query; resolver is external)
    "url",      # remote URL (resolver is external)
    "derived",  # produced by a transform of another DataRef
]
_EXTERNAL_KINDS: frozenset[str] = frozenset({"query", "url", "derived"})


class DataError(VaultError):
    """Base for ``carl.data.*`` errors.

    Inherits from :class:`VaultError` so generic ``except VaultError`` catches
    both. Direct ``except DataError`` continues to work as before.
    """

    code = "carl.data"


class DataRef(BaseModel):
    """Opaque handle to a data payload held in a :class:`DataVault`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    ref_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    kind: DataKind = Field(...)
    uri: str = Field(
        ...,
        description="Stable reference: 'carl-data://<kind>/<name>' for "
        "vault-local refs; 'file://...' / 'https://...' for external.",
    )
    size_bytes: int | None = Field(
        default=None,
        description="Payload size if known. None for unsized streams/queries.",
    )
    content_type: str | None = Field(
        default=None,
        description="IANA MIME hint (e.g. 'application/json'). Informational.",
    )
    sha256: str | None = Field(
        default=None,
        description="Full 64-hex sha256. Populated at put time for inline "
        "bytes; lazy for file/stream.",
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
        """Public metadata. Never contains the raw bytes."""
        expires = self.expired_at()
        return {
            "ref_id": str(self.ref_id),
            "kind": self.kind,
            "uri": self.uri,
            "size_bytes": self.size_bytes,
            "content_type": self.content_type,
            "sha256": self.sha256,
            "ttl_s": self.ttl_s,
            "created_at": self.created_at.isoformat(),
            "expires_at": expires.isoformat() if expires is not None else None,
        }


class DataVault(Vault[DataRef, bytes]):
    """Specialization of :class:`Vault` for payloads with offset-addressable reads.

    Three backing modes — bytes (inline), file (lazy-read from path), stream
    (incrementally buffered iterator) — plus external refs (query/url/derived)
    that flow through the resolver chain. TTL / revoke / fingerprint_of /
    exists / list_refs / __len__ inherited from the base.
    """

    _ref_class = DataRef
    _error_prefix: ClassVar[str] = "carl.data"
    _error_class: ClassVar[type[VaultError]] = DataError

    # -- put modes ------------------------------------------------------

    def put_bytes(
        self,
        data: bytes,
        *,
        uri: str | None = None,
        content_type: str | None = None,
        ttl_s: int | None = None,
    ) -> DataRef:
        """Register inline bytes. Fastest path; size + sha256 known at put time."""
        self._validate_ttl(ttl_s)
        raw = bytes(data)
        sha_full, fp = _digest_full(raw)
        ref = DataRef(
            kind="bytes",
            uri=uri or f"carl-data://bytes/{uuid.uuid4().hex[:16]}",
            size_bytes=len(raw),
            content_type=content_type,
            sha256=sha_full,
            ttl_s=ttl_s,
        )
        self.put_value(ref, raw, fingerprint_hex=fp)
        with self._lock:
            self._entries[ref.ref_id].extra["sha256_full"] = sha_full
        return ref

    def open_file(
        self,
        path: str | Path,
        *,
        content_type: str | None = None,
        ttl_s: int | None = None,
    ) -> DataRef:
        """Register a filesystem path. File not read until :meth:`read` is called."""
        cls = type(self)
        self._validate_ttl(ttl_s)
        p = Path(path).expanduser().resolve()
        if not p.is_file():
            raise cls._err(
                "backend_unavailable",
                f"path is not a readable file: {p}",
                {"path": str(p)},
            )
        ref = DataRef(
            kind="file",
            uri=p.as_uri(),
            size_bytes=p.stat().st_size,
            content_type=content_type or content_type_from_path(p),
            sha256=None,  # populated lazily on first full read
            ttl_s=ttl_s,
        )
        self.put_ref_only(ref)
        with self._lock:
            self._entries[ref.ref_id].extra["path"] = p
        return ref

    def open_stream(
        self,
        iterator: Iterator[bytes],
        *,
        uri: str,
        content_type: str | None = None,
        size_hint: int | None = None,
        ttl_s: int | None = None,
    ) -> DataRef:
        """Register a byte-chunk iterator. Consumed on demand + buffered for
        consistent re-reads from offset 0 within the already-buffered portion.
        """
        self._validate_ttl(ttl_s)
        ref = DataRef(
            kind="stream",
            uri=uri,
            size_bytes=size_hint,
            content_type=content_type,
            sha256=None,
            ttl_s=ttl_s,
        )
        self.put_ref_only(ref)
        with self._lock:
            entry = self._entries[ref.ref_id]
            entry.extra["iterator"] = iterator
            entry.extra["buffered"] = bytearray()
        return ref

    def put_external(
        self,
        uri: str,
        *,
        kind: DataKind,
        size_hint: int | None = None,
        content_type: str | None = None,
        sha256: str | None = None,
        ttl_s: int | None = None,
    ) -> DataRef:
        """Register an external ref (``query`` / ``url`` / ``derived``).

        The vault doesn't fetch — :meth:`read` raises ``carl.data.backend_unavailable``
        for these kinds. Callers that want resolver-chain fall-through use
        :meth:`resolve` (inherited from the base) after registering a
        resolver for the ref's kind via :meth:`register_resolver`.
        """
        if kind not in _EXTERNAL_KINDS:
            raise ValidationError(
                f"put_external requires kind in {{query,url,derived}}, got {kind!r}",
                code=type(self)._err_code("invalid_kind"),
                context={"kind": kind},
            )
        self._validate_ttl(ttl_s)
        ref = DataRef(
            kind=kind,
            uri=uri,
            size_bytes=size_hint,
            content_type=content_type,
            sha256=sha256,
            ttl_s=ttl_s,
        )
        self.put_ref_only(ref)
        return ref

    # -- read -----------------------------------------------------------

    def read(
        self,
        ref: DataRef,
        *,
        offset: int = 0,
        length: int | None = None,
    ) -> bytes:
        """Read ``length`` bytes starting at ``offset``. ``length=None`` = to end.

        Dispatches on entry kind: bytes / file / stream. External refs
        (query/url/derived) raise ``carl.data.backend_unavailable`` — use
        :meth:`resolve` with a registered resolver for those.
        """
        cls = type(self)
        if offset < 0:
            raise cls._err("out_of_range", f"offset must be >= 0, got {offset}", {"offset": offset})
        if length is not None and length < 0:
            raise cls._err("out_of_range", f"length must be >= 0 or None, got {length}", {"length": length})

        entry, current = self._lookup_live(ref)

        if current.kind == "bytes":
            payload = entry.value or b""
            return _slice(payload, offset, length)

        if current.kind == "file":
            path = entry.extra.get("path")
            assert isinstance(path, Path)
            with path.open("rb") as fh:
                fh.seek(offset)
                data = fh.read(-1 if length is None else length)
            # Populate sha256 lazily on first full read.
            if offset == 0 and length is None and entry.extra.get("sha256_full") is None:
                sha_full, fp = _digest_full(data)
                entry.extra["sha256_full"] = sha_full
                entry.fingerprint_hex = fp
                with self._lock:
                    self._refs[ref.ref_id] = current.model_copy(update={"sha256": sha_full})
            return data

        if current.kind == "stream":
            return self._read_stream(entry, offset=offset, length=length)

        # query / url / derived — resolver chain via resolve(), not read()
        raise cls._err(
            "backend_unavailable",
            f"ref kind={current.kind!r} has no local bytes; resolve via resolver chain",
            {"ref_id": str(ref.ref_id), "kind": current.kind},
        )

    def _read_stream(
        self,
        entry: Any,
        *,
        offset: int,
        length: int | None,
    ) -> bytes:
        buffered = entry.extra.get("buffered")
        iterator = entry.extra.get("iterator")
        assert isinstance(buffered, bytearray)
        assert iterator is not None
        target_end = (offset + length) if length is not None else None
        with self._lock:
            while target_end is None or len(buffered) < target_end:
                try:
                    chunk = next(iterator)
                except StopIteration:
                    break
                buffered.extend(chunk)
            buf = bytes(buffered)
        return _slice(buf, offset, length)

    def read_all(self, ref: DataRef) -> bytes:
        return self.read(ref, offset=0, length=None)

    # -- metadata overrides --------------------------------------------

    def fingerprint_of(self, ref: DataRef) -> str:
        """12-hex sha256 preview. Forces a full read for lazy-hash refs."""
        entry, _ = self._lookup_live(ref)
        if entry.fingerprint_hex is not None:
            return entry.fingerprint_hex
        data = self.read_all(ref)
        sha_full, fp = _digest_full(data)
        entry.extra["sha256_full"] = sha_full
        entry.fingerprint_hex = fp
        return fp

    def sha256_of(self, ref: DataRef) -> str:
        """Full 64-hex sha256. Forces a full read for lazy-hash refs."""
        entry, _ = self._lookup_live(ref)
        cached = entry.extra.get("sha256_full")
        if isinstance(cached, str):
            return cached
        _ = self.fingerprint_of(ref)
        cached = entry.extra.get("sha256_full")
        assert isinstance(cached, str)
        return cached

    def size_of(self, ref: DataRef) -> int | None:
        """Known size or ``None`` for unsized streams before first read."""
        cls = type(self)
        with self._lock:
            current = self._refs.get(ref.ref_id)
            if current is None:
                raise cls._err(
                    "not_found",
                    f"unknown data handle: {ref.ref_id}",
                    {"ref_id": str(ref.ref_id)},
                )
            return current.size_bytes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def content_type_from_path(path: Path) -> str | None:
    """Minimal MIME guess from suffix. Returns ``None`` when unknown."""
    guess, _ = mimetypes.guess_type(path.as_uri())
    return guess


def _digest_full(data: bytes) -> tuple[str, str]:
    """Return (sha256_64hex, fingerprint_12hex) for ``data``. One pass."""
    sha = hashlib.sha256(data).hexdigest()
    return sha, sha[:12]


def _slice(payload: bytes, offset: int, length: int | None) -> bytes:
    """Bounded slice that tolerates offset past end (returns empty)."""
    if offset >= len(payload):
        return b""
    end = len(payload) if length is None else min(offset + length, len(payload))
    return payload[offset:end]
