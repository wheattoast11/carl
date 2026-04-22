"""Data-handle primitives (v0.16.1 — Stage C).

Extends the capability-constrained handle runtime (``carl_core.secrets``) to
arbitrary data payloads: files, inline bytes, streamed iterators, deferred
queries, and remote URLs. The agent moves :class:`DataRef` handles across
tool-call boundaries; the raw bytes stay behind the :class:`DataVault`.

Design
------

Same shape as :class:`carl_core.secrets.SecretRef` on purpose — callers that
understand one runtime understand the other. The difference is in what the
"value" is:

* ``SecretRef``'s value is a password / key — agents should NEVER read it;
  :meth:`SecretVault.resolve` is privileged-only.
* ``DataRef``'s value is content — the agent frequently needs to read it,
  but still through an explicit :meth:`DataVault.read` call that lands in
  the :class:`~carl_core.interaction.InteractionChain` as a ``DATA_READ``
  step. Reads default to a preview cap so accidental whole-file loads
  stay visible in the audit trail.

What lives here (carl-core, dependency-free):

* :class:`DataRef` — frozen handle with kind/uri/size/sha256/ttl metadata.
* :class:`DataVault` — thread-safe in-memory KV with 3 backing modes
  (inline bytes, file-on-disk, streamed iterator).
* :func:`content_type_from_path` — minimal MIME guessing.
* Error codes under ``carl.data.*``:

  - ``carl.data.invalid_kind`` — invalid kind / invalid ttl
  - ``carl.data.not_found`` — handle ref_id unknown
  - ``carl.data.revoked`` / ``carl.data.expired``
  - ``carl.data.backend_unavailable`` — path unreadable
  - ``carl.data.out_of_range`` — offset/length beyond payload
  - ``carl.data.stream_exhausted`` — stream cursor past end

What lives ABOVE in carl_studio (has stdlib side-effects, optional deps):

* Toolkit-level transforms (gzip / head / tail), publish side-effects,
  JSON / text decoding convenience, agent-callable method surface.
"""

from __future__ import annotations

import mimetypes
import threading
import uuid
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from carl_core.errors import CARLError, ValidationError

__all__ = [
    "DataKind",
    "DataRef",
    "DataVault",
    "DataError",
    "content_type_from_path",
]


# ---------------------------------------------------------------------------
# Kinds
# ---------------------------------------------------------------------------


DataKind = Literal[
    "bytes",    # inline bytes held in vault memory
    "file",     # path on disk (lazy-read, not slurped at open)
    "stream",   # iterator yielding chunks (consumable)
    "query",    # deferred query (URI holds the query; resolver is external)
    "url",      # remote URL (resolver is external)
    "derived",  # produced by a transform of another DataRef
]


# ---------------------------------------------------------------------------
# Error taxonomy
# ---------------------------------------------------------------------------


class DataError(CARLError):
    """Base for all ``carl.data.*`` errors."""

    code = "carl.data"


# ---------------------------------------------------------------------------
# DataRef
# ---------------------------------------------------------------------------


class DataRef(BaseModel):
    """Opaque handle to a data payload held in a :class:`DataVault`.

    The handle travels across tool-call boundaries. Operations that need the
    bytes themselves call :meth:`DataVault.read` — which emits a ``DATA_READ``
    step into the :class:`~carl_core.interaction.InteractionChain`.

    ``size_bytes`` is ``None`` for unsized sources (streams, deferred queries).
    ``sha256`` is populated lazily on first resolve for file-backed refs;
    inline-bytes refs populate it at put time.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    ref_id: uuid.UUID = Field(
        default_factory=uuid.uuid4,
        description="Opaque UUID handle.",
    )
    kind: DataKind = Field(
        ...,
        description="What this handle references (bytes/file/stream/query/url/derived).",
    )
    uri: str = Field(
        ...,
        description="Stable reference string; follows 'carl-data://<kind>/<name>' "
        "for vault-local refs and 'file://...' / 'https://...' for external.",
    )
    size_bytes: int | None = Field(
        default=None,
        description="Payload size if known. None for unsized streams/queries.",
    )
    content_type: str | None = Field(
        default=None,
        description="IANA MIME type hint (e.g. 'application/json', 'text/plain'). "
        "Informational — resolver may override.",
    )
    sha256: str | None = Field(
        default=None,
        description="Full 64-hex sha256 digest of the resolved bytes, when known. "
        "Populated at put time for inline bytes; lazy for files.",
    )
    ttl_s: int | None = Field(
        default=None,
        description="Optional lifetime in seconds. Checked lazily at resolve time.",
    )
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


# ---------------------------------------------------------------------------
# DataVault
# ---------------------------------------------------------------------------


class _DataEntry:
    """Internal KV entry. NEVER exposed to callers. The enclosing class is
    already underscore-prefixed so the fields here use plain names — pyright
    would otherwise flag every vault-level read as reportPrivateUsage."""

    __slots__ = (
        "kind",
        "payload",
        "path",
        "iterator",
        "buffered",
        "sha256_full",
        "fingerprint_hex",
        "revoked",
    )

    def __init__(self, kind: DataKind) -> None:
        self.kind: DataKind = kind
        self.payload: bytes | None = None
        self.path: Path | None = None
        self.iterator: Iterator[bytes] | None = None
        self.buffered: bytearray | None = None
        self.sha256_full: str | None = None
        self.fingerprint_hex: str | None = None
        self.revoked: bool = False


class DataVault:
    """Thread-safe in-memory handle registry for arbitrary data payloads.

    Three put modes (bytes / file / stream) produce refs the agent can pass
    around. Reads are offset+length addressable; streams lazily buffer so a
    re-read after cursor advance returns consistent bytes from position 0.

    The vault never emits chain steps itself — the toolkit layer
    (``carl_studio.handles.data.DataToolkit``) wires the audit callbacks.
    """

    def __init__(self) -> None:
        self._entries: dict[uuid.UUID, _DataEntry] = {}
        self._refs: dict[uuid.UUID, DataRef] = {}
        self._lock = threading.RLock()

    # -- put modes -------------------------------------------------------

    def put_bytes(
        self,
        data: bytes,
        *,
        uri: str | None = None,
        content_type: str | None = None,
        ttl_s: int | None = None,
    ) -> DataRef:
        """Register inline bytes. Fastest path; size + sha256 known at put time."""
        self._check_ttl(ttl_s)
        sha_full, fp = _digest_full(data)
        entry = _DataEntry("bytes")
        entry.payload = bytes(data)
        entry.sha256_full = sha_full
        entry.fingerprint_hex = fp

        ref = DataRef(
            kind="bytes",
            uri=uri or f"carl-data://bytes/{uuid.uuid4().hex[:16]}",
            size_bytes=len(data),
            content_type=content_type,
            sha256=sha_full,
            ttl_s=ttl_s,
        )
        with self._lock:
            self._entries[ref.ref_id] = entry
            self._refs[ref.ref_id] = ref
        return ref

    def open_file(
        self,
        path: str | Path,
        *,
        content_type: str | None = None,
        ttl_s: int | None = None,
    ) -> DataRef:
        """Register a filesystem path. File not read until :meth:`read` is called.

        ``size_bytes`` is taken from ``Path.stat().st_size`` at open time.
        ``sha256`` is populated lazily on first read-all.
        """
        self._check_ttl(ttl_s)
        p = Path(path).expanduser().resolve()
        if not p.is_file():
            raise DataError(
                f"path is not a readable file: {p}",
                code="carl.data.backend_unavailable",
                context={"path": str(p)},
            )
        size = p.stat().st_size
        mime = content_type or content_type_from_path(p)

        entry = _DataEntry("file")
        entry.path = p

        ref = DataRef(
            kind="file",
            uri=p.as_uri(),
            size_bytes=size,
            content_type=mime,
            sha256=None,  # lazy — populated on first full read
            ttl_s=ttl_s,
        )
        with self._lock:
            self._entries[ref.ref_id] = entry
            self._refs[ref.ref_id] = ref
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
        """Register a byte-chunk iterator. Consumed on demand; buffered as consumed.

        Re-reads from offset 0 work as long as the caller does not exceed the
        portion already buffered. ``size_hint`` is copied into ``size_bytes``
        if known; otherwise ``None``.
        """
        self._check_ttl(ttl_s)
        entry = _DataEntry("stream")
        entry.iterator = iterator
        entry.buffered = bytearray()

        ref = DataRef(
            kind="stream",
            uri=uri,
            size_bytes=size_hint,
            content_type=content_type,
            sha256=None,
            ttl_s=ttl_s,
        )
        with self._lock:
            self._entries[ref.ref_id] = entry
            self._refs[ref.ref_id] = ref
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

        The vault does NOT fetch / execute. This is the right primitive for
        "agent gets a pointer the toolkit layer dereferences later."
        """
        if kind not in ("query", "url", "derived"):
            raise ValidationError(
                f"put_external requires kind in {{query,url,derived}}, got {kind!r}",
                code="carl.data.invalid_kind",
                context={"kind": kind},
            )
        self._check_ttl(ttl_s)
        entry = _DataEntry(kind)
        ref = DataRef(
            kind=kind,
            uri=uri,
            size_bytes=size_hint,
            content_type=content_type,
            sha256=sha256,
            ttl_s=ttl_s,
        )
        with self._lock:
            self._entries[ref.ref_id] = entry
            self._refs[ref.ref_id] = ref
        return ref

    # -- read ------------------------------------------------------------

    def read(
        self,
        ref: DataRef,
        *,
        offset: int = 0,
        length: int | None = None,
    ) -> bytes:
        """Read ``length`` bytes starting at ``offset``.

        ``length=None`` reads to end. Raises ``DataError`` if the ref is
        external (``query``/``url``) — those must be resolved by a toolkit-
        layer resolver before bytes are available.
        """
        if offset < 0:
            raise DataError(
                f"offset must be >= 0, got {offset}",
                code="carl.data.out_of_range",
                context={"offset": offset},
            )
        if length is not None and length < 0:
            raise DataError(
                f"length must be >= 0 or None, got {length}",
                code="carl.data.out_of_range",
                context={"length": length},
            )

        entry, current_ref = self._lookup_live(ref)

        if entry.kind == "bytes":
            payload = entry.payload or b""
            return _slice(payload, offset, length)

        if entry.kind == "file":
            assert entry.path is not None
            with entry.path.open("rb") as fh:
                fh.seek(offset)
                data = fh.read(-1 if length is None else length)
            # Populate sha256 lazily on first full read.
            if offset == 0 and length is None and entry.sha256_full is None:
                sha_full, fp = _digest_full(data)
                entry.sha256_full = sha_full
                entry.fingerprint_hex = fp
                # Refresh the stored ref with the now-known sha256 for future describe() calls.
                with self._lock:
                    self._refs[current_ref.ref_id] = current_ref.model_copy(
                        update={"sha256": sha_full}
                    )
            return data

        if entry.kind == "stream":
            return self._read_stream(entry, offset=offset, length=length)

        # query / url / derived — external refs have no bytes locally.
        raise DataError(
            f"ref kind={entry.kind!r} has no local bytes; resolve via toolkit layer",
            code="carl.data.backend_unavailable",
            context={"ref_id": str(ref.ref_id), "kind": entry.kind},
        )

    def _read_stream(
        self,
        entry: _DataEntry,
        *,
        offset: int,
        length: int | None,
    ) -> bytes:
        assert entry.buffered is not None
        assert entry.iterator is not None
        target_end = (offset + length) if length is not None else None
        with self._lock:
            # Advance the iterator until buffered covers target_end, or exhausted.
            while target_end is None or len(entry.buffered) < target_end:
                try:
                    chunk = next(entry.iterator)
                except StopIteration:
                    break
                entry.buffered.extend(chunk)
            buf = bytes(entry.buffered)
        return _slice(buf, offset, length)

    def read_all(self, ref: DataRef) -> bytes:
        """Convenience for ``read(ref, offset=0, length=None)``."""
        return self.read(ref, offset=0, length=None)

    # -- metadata --------------------------------------------------------

    def fingerprint_of(self, ref: DataRef) -> str:
        """12-hex sha256 preview. Forces a full read for lazy-hash refs."""
        entry, _ = self._lookup_live(ref)
        if entry.fingerprint_hex is not None:
            return entry.fingerprint_hex
        data = self.read_all(ref)
        sha_full, fp = _digest_full(data)
        entry.sha256_full = sha_full
        entry.fingerprint_hex = fp
        return fp

    def sha256_of(self, ref: DataRef) -> str:
        """Full 64-hex sha256. Forces a full read for lazy-hash refs."""
        entry, _ = self._lookup_live(ref)
        if entry.sha256_full is not None:
            return entry.sha256_full
        _ = self.fingerprint_of(ref)
        assert entry.sha256_full is not None
        return entry.sha256_full

    def size_of(self, ref: DataRef) -> int | None:
        """Known size or ``None`` for unsized streams before first read."""
        with self._lock:
            current = self._refs.get(ref.ref_id)
            if current is None:
                raise DataError(
                    f"unknown data handle: {ref.ref_id}",
                    code="carl.data.not_found",
                    context={"ref_id": str(ref.ref_id)},
                )
            return current.size_bytes

    # -- lifecycle -------------------------------------------------------

    def revoke(self, ref: DataRef) -> bool:
        """Invalidate a handle. Idempotent. Returns True iff it was live."""
        with self._lock:
            entry = self._entries.get(ref.ref_id)
            if entry is None or entry.revoked:
                return False
            entry.revoked = True
        return True

    def exists(self, ref: DataRef) -> bool:
        with self._lock:
            entry = self._entries.get(ref.ref_id)
            if entry is None or entry.revoked:
                return False
        return not ref.is_expired()

    def list_refs(self) -> list[DataRef]:
        now = datetime.now(timezone.utc)
        with self._lock:
            refs: list[DataRef] = []
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

    # -- internals -------------------------------------------------------

    def _lookup_live(self, ref: DataRef) -> tuple[_DataEntry, DataRef]:
        """Resolve to (entry, current_ref) or raise. Handles expiry self-cleanup."""
        with self._lock:
            entry = self._entries.get(ref.ref_id)
            current = self._refs.get(ref.ref_id)
            if entry is None or current is None:
                raise DataError(
                    f"unknown data handle: {ref.ref_id}",
                    code="carl.data.not_found",
                    context={"ref_id": str(ref.ref_id)},
                )
            if entry.revoked:
                raise DataError(
                    f"data handle {ref.ref_id} was revoked",
                    code="carl.data.revoked",
                    context={"ref_id": str(ref.ref_id)},
                )
            if current.is_expired():
                entry.revoked = True
                raise DataError(
                    f"data handle {ref.ref_id} has expired",
                    code="carl.data.expired",
                    context={
                        "ref_id": str(ref.ref_id),
                        "created_at": current.created_at.isoformat(),
                        "ttl_s": current.ttl_s,
                    },
                )
            return entry, current

    @staticmethod
    def _check_ttl(ttl_s: int | None) -> None:
        if ttl_s is not None and ttl_s < 1:
            raise ValidationError(
                f"ttl_s must be a positive int or None, got {ttl_s!r}",
                code="carl.data.invalid_kind",
                context={"ttl_s": ttl_s},
            )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def content_type_from_path(path: Path) -> str | None:
    """Minimal MIME guess from suffix. Returns ``None`` when unknown.

    Thin wrapper around :mod:`mimetypes` so the toolkit layer doesn't re-roll
    the same lookup. No network; no magic bytes.
    """
    guess, _ = mimetypes.guess_type(path.as_uri())
    return guess


def _digest_full(data: bytes) -> tuple[str, str]:
    """Return (sha256_64hex, fingerprint_12hex) for ``data``.

    Two outputs because upstream tools want either — 64 hex for archival
    identity, 12 hex for logs. Computed in one pass.
    """
    import hashlib

    sha = hashlib.sha256(data).hexdigest()
    return sha, sha[:12]


def _slice(payload: bytes, offset: int, length: int | None) -> bytes:
    """Bounded slice that tolerates offset past end (returns empty)."""
    if offset >= len(payload):
        return b""
    end = len(payload) if length is None else min(offset + length, len(payload))
    return payload[offset:end]
