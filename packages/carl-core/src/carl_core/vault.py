"""Generic handle-vault primitive — `Vault[H, V]` + resolver chain.

The unifying primitive behind `SecretVault`, `DataVault`, `ResourceVault`, and
any future handle kind the agent needs to reference. Every specialization gets
thread-safe UUID-keyed registration, TTL self-revoke, handle-level fingerprinting,
and a pluggable resolver chain for refs whose values live outside the process
(keyring, 1Password CLI, HashiCorp Vault, fernet-file, etc.).

Design principles
-----------------

1. **Refs are the authority.** An `H` handle is an opaque UUID + metadata. Holding
   the handle IS the permission to act on its value. `put → ref`, `ref → value`
   via `resolve`, `revoke(ref)` invalidates. The vault is the capability layer.

2. **Values can live anywhere.** Local bytes (inline), filesystem paths, streamed
   iterators, remote URIs, OS keychain, encrypted files — all expressible via the
   resolver chain. Local storage wins; otherwise a registered resolver for the
   ref's `kind` runs. No resolver + no local value → `carl.vault.not_found`.

3. **Gated resolution.** Secret-like vaults set `_require_privileged_resolve = True`
   so every dereference is visible in code review. Data-like vaults default to
   non-privileged because the normal workflow is "the agent reads the file."

4. **Thread-safe by default.** One `RLock` per vault protects both the entry
   table and the resolver table. Operations are fine-grained — a slow resolver
   doesn't block unrelated puts.

5. **Audit is the caller's responsibility.** The vault DOES NOT emit chain steps;
   the toolkit layer that wraps the vault does. Keeping the vault dependency-free
   lets `carl-core` stay network-free and lets future Terminals-core reuse the
   primitive wholesale.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import (
    Any,
    Callable,
    ClassVar,
    Generic,
    Protocol,
    TypeVar,
    runtime_checkable,
)

from carl_core.errors import CARLError, ValidationError
from carl_core.hashing import fingerprint


__all__ = [
    "HandleRef",
    "Vault",
    "VaultError",
    "Resolver",
]


# ---------------------------------------------------------------------------
# Error taxonomy
# ---------------------------------------------------------------------------


class VaultError(CARLError):
    """Base for ``carl.vault.*`` errors.

    Subclasses (or direct callers) set a specific ``code`` value so CLI / MCP
    clients can branch on the stable taxonomy.
    """

    code = "carl.vault"


# ---------------------------------------------------------------------------
# Protocol + type vars
# ---------------------------------------------------------------------------


@runtime_checkable
class HandleRef(Protocol):
    """Minimal surface every handle type must satisfy.

    `Vault[H, V]` specializations (SecretRef / DataRef / ResourceRef) all meet
    this Protocol. The vault never inspects the handle beyond these members.

    Members are declared as read-only properties so frozen Pydantic models +
    frozen dataclasses satisfy the Protocol under Pyright strict mode. Writable-
    attribute form would fire `reportInvalidTypeArguments` on frozen subtypes.
    """

    @property
    def ref_id(self) -> uuid.UUID: ...

    @property
    def kind(self) -> str: ...

    @property
    def ttl_s(self) -> int | None: ...

    def is_expired(self, *, now: datetime | None = None) -> bool: ...

    def describe(self) -> dict[str, Any]: ...


H = TypeVar("H", bound=HandleRef)
V = TypeVar("V")


# A resolver takes a handle and returns its value bytes/object. Free-form
# signature — resolvers that need more context (env, config) capture it via
# closure or partial application at registration time.
Resolver = Callable[[HandleRef], Any]


# ---------------------------------------------------------------------------
# Internal entry
# ---------------------------------------------------------------------------


def _empty_extra() -> dict[str, Any]:
    return {}


@dataclass
class _VaultEntry(Generic[V]):
    """Internal KV entry. NEVER exposed to callers.

    Specializations may subclass to attach additional metadata (e.g. DataVault
    tracks `path`, `iterator`, `buffered`). The base supplies the minimum:
    value, revocation flag, and an optional fingerprint cache.
    """

    value: V | None = None
    revoked: bool = False
    fingerprint_hex: str | None = None
    # Resolver-populated cache: stores the value produced by a resolver call
    # so subsequent resolves within TTL return the cached bytes. `None` means
    # "not yet resolved remotely." Separate from `value` so `put_ref_only`
    # entries can still cache after first resolve.
    resolved_cache: V | None = None
    resolved_at: datetime | None = None
    extra: dict[str, Any] = field(default_factory=_empty_extra)


# ---------------------------------------------------------------------------
# Vault[H, V]
# ---------------------------------------------------------------------------


class Vault(Generic[H, V]):
    """Thread-safe handle registry + resolver chain.

    Subclass to specialize: set `_ref_class` to the concrete `HandleRef`
    subclass and override `_make_ref` if you need custom fields on the handle.
    Example::

        class SecretVault(Vault[SecretRef, bytes]):
            _ref_class = SecretRef
            _require_privileged_resolve = True
    """

    # Specializations override these class-level attributes.
    _ref_class: ClassVar[type]  # must be set by subclasses; type: type[H]
    _require_privileged_resolve: ClassVar[bool] = False
    _resolver_ttl_s: ClassVar[int | None] = None  # cache duration for resolved values
    # Error-code namespace. Subclasses override to preserve their existing
    # error-code taxonomy — e.g. SecretVault sets "carl.secrets" so raised
    # errors have codes like "carl.secrets.not_found" rather than the
    # generic "carl.vault.not_found".
    _error_prefix: ClassVar[str] = "carl.vault"
    # Exception class raised by the base lifecycle methods. Subclasses override
    # to their own VaultError subclass (e.g. SecretsError) so `pytest.raises`
    # catches the specific subtype AND `carl.<namespace>.*` codes stay stable.
    _error_class: ClassVar[type[VaultError]] = VaultError

    @classmethod
    def _err_code(cls, suffix: str) -> str:
        return f"{cls._error_prefix}.{suffix}"

    @classmethod
    def _err(
        cls, suffix: str, message: str, context: dict[str, Any] | None = None,
        *, cause: BaseException | None = None,
    ) -> VaultError:
        """Build a subclass-typed VaultError with the right code + context."""
        return cls._error_class(
            message,
            code=cls._err_code(suffix),
            context=context or {},
            cause=cause,
        )

    def __init__(self) -> None:
        self._entries: dict[uuid.UUID, _VaultEntry[V]] = {}
        self._refs: dict[uuid.UUID, H] = {}
        self._resolvers: dict[str, Resolver] = {}
        self._lock = threading.RLock()

    # -- put ------------------------------------------------------------

    def _validate_ttl(self, ttl_s: int | None) -> None:
        if ttl_s is not None and ttl_s < 1:
            raise ValidationError(
                f"ttl_s must be a positive int or None, got {ttl_s!r}",
                code=type(self)._err_code("invalid_ttl"),
                context={"ttl_s": ttl_s},
            )

    def _register(self, ref: H, entry: _VaultEntry[V]) -> None:
        """Atomically insert a (ref, entry) pair into the vault."""
        with self._lock:
            self._entries[ref.ref_id] = entry
            self._refs[ref.ref_id] = ref

    def put_value(self, ref: H, value: V, *, fingerprint_hex: str | None = None) -> H:
        """Register a handle whose value lives locally in the vault.

        Most specializations wrap this via a typed `put_bytes` / `put_file` /
        `put` method. The base exposes it so tests and advanced callers can
        stage refs without going through a typed wrapper.
        """
        entry: _VaultEntry[V] = _VaultEntry(
            value=value, fingerprint_hex=fingerprint_hex,
        )
        self._register(ref, entry)
        return ref

    def put_ref_only(self, ref: H) -> H:
        """Register a handle whose value is held externally.

        The ref lands with `value=None`; `resolve` falls through to the registered
        resolver for `ref.kind`. Failure to find a resolver at resolve time
        raises `carl.vault.resolver_unavailable`.
        """
        entry: _VaultEntry[V] = _VaultEntry(value=None)
        self._register(ref, entry)
        return ref

    # -- resolve --------------------------------------------------------

    def resolve(self, ref: H, *, privileged: bool = False) -> V:
        """Dereference a handle.

        Non-privileged callers hit ``<prefix>.unauthorized_resolve`` when the
        subclass sets ``_require_privileged_resolve = True``. This is a
        visibility marker, not a cryptographic guard — its purpose is to make
        every value access visible in code review.
        """
        cls = type(self)
        if cls._require_privileged_resolve and not privileged:
            raise cls._err(
                "unauthorized_resolve",
                f"{cls.__name__}.resolve() requires privileged=True",
                {"ref_id": str(ref.ref_id), "kind": ref.kind},
            )

        entry, current = self._lookup_live(ref)

        # (1) local value wins
        if entry.value is not None:
            return entry.value

        # (2) resolver-cache within TTL
        if (
            entry.resolved_cache is not None
            and entry.resolved_at is not None
            and self._resolver_ttl_s is not None
        ):
            age_s = (datetime.now(timezone.utc) - entry.resolved_at).total_seconds()
            if age_s < self._resolver_ttl_s:
                return entry.resolved_cache

        # (3) fall through to registered resolver
        resolver = self._resolvers.get(current.kind)
        if resolver is None:
            raise cls._err(
                "resolver_unavailable",
                f"no value and no resolver registered for kind={current.kind!r}",
                {"ref_id": str(ref.ref_id), "kind": current.kind},
            )

        try:
            produced = resolver(current)
        except VaultError:
            raise
        except Exception as exc:
            raise cls._err(
                "resolver_failed",
                f"resolver for kind={current.kind!r} raised {type(exc).__name__}: {exc}",
                {"ref_id": str(ref.ref_id), "kind": current.kind},
                cause=exc,
            ) from exc

        with self._lock:
            entry.resolved_cache = produced  # type: ignore[assignment]
            entry.resolved_at = datetime.now(timezone.utc)
        return produced  # type: ignore[return-value]

    # -- metadata -------------------------------------------------------

    def fingerprint_of(self, ref: H) -> str:
        """12-hex preview of the value. Safe to log / show to the agent.

        For local values, returns the cached fingerprint or computes it from
        ``bytes(value)`` if possible. For resolver-backed values, forces a
        resolve + hashes the result. Subclasses may override for non-byte
        values (e.g. file refs compute from path-read).
        """
        cls = type(self)
        with self._lock:
            entry = self._entries.get(ref.ref_id)
            if entry is None:
                raise cls._err(
                    "not_found",
                    f"unknown handle: {ref.ref_id}",
                    {"ref_id": str(ref.ref_id)},
                )
            if entry.fingerprint_hex is not None:
                return entry.fingerprint_hex
            if entry.value is not None and isinstance(entry.value, (bytes, bytearray)):
                fp = fingerprint(bytes(entry.value))
                entry.fingerprint_hex = fp
                return fp
        # Fallback: resolve (privileged to bypass secret-vault gate) and hash.
        value = self.resolve(ref, privileged=True)
        if isinstance(value, (bytes, bytearray)):
            fp = fingerprint(bytes(value))
            with self._lock:
                if (e := self._entries.get(ref.ref_id)) is not None:
                    e.fingerprint_hex = fp
            return fp
        raise cls._err(
            "unsupported_fingerprint",
            f"value for {ref.ref_id} is not byte-like; subclass must override fingerprint_of",
            {"ref_id": str(ref.ref_id), "type": type(value).__name__},
        )

    def exists(self, ref: H) -> bool:
        """Truthy iff the handle is known, not revoked, not expired."""
        with self._lock:
            entry = self._entries.get(ref.ref_id)
            if entry is None or entry.revoked:
                return False
        return not ref.is_expired()

    def list_refs(self) -> list[H]:
        """Enumerate live (non-revoked, non-expired) handles."""
        now = datetime.now(timezone.utc)
        with self._lock:
            return [
                r
                for rid, r in self._refs.items()
                if (e := self._entries.get(rid)) is not None
                and not e.revoked
                and not r.is_expired(now=now)
            ]

    def __len__(self) -> int:
        return len(self.list_refs())

    # -- lifecycle ------------------------------------------------------

    def revoke(self, ref: H) -> bool:
        """Invalidate a handle. Returns True iff the handle existed + was live.

        Idempotent: revoking an already-revoked or unknown handle returns
        False without raising.
        """
        with self._lock:
            entry = self._entries.get(ref.ref_id)
            if entry is None or entry.revoked:
                return False
            entry.revoked = True
            entry.resolved_cache = None
            entry.resolved_at = None
        return True

    # -- resolver chain -------------------------------------------------

    def register_resolver(self, kind: str, resolver: Resolver) -> None:
        """Register a `Resolver` callable for handles of `kind`.

        The resolver fires when `resolve(ref)` is called on a handle whose
        entry has no local value. Only one resolver per kind — registering
        the same kind twice replaces the prior resolver.
        """
        if not callable(resolver):
            raise ValidationError(
                f"resolver for kind={kind!r} must be callable",
                code=type(self)._err_code("invalid_resolver"),
                context={"kind": kind, "type": type(resolver).__name__},
            )
        with self._lock:
            self._resolvers[kind] = resolver

    def register_runtime_resolver(
        self,
        kind: str,
        resolver: Resolver,
        *,
        admin_token: Any | None = None,
    ) -> None:
        """Register a resolver sourced from the private runtime.

        Admin-gate seam for proprietary backends. Functionally identical to
        :meth:`register_resolver`; the separate entry point signals intent
        and accepts an optional ``admin_token`` that this vault verifies
        via a duck-typed ``.verify()`` call (so ``carl-core`` stays
        independent of ``carl_studio.admin``).

        Typical call site — inside the private runtime module::

            from carl_studio.admin import issue_token
            token = issue_token()  # raises if admin gate locked
            vault.register_runtime_resolver(
                "hardware-attested",
                hardware_resolver,
                admin_token=token,
            )

        The real moat enforcement lives in CI: ``carl-studio`` and
        ``carl-core`` must not import ``resonance`` / ``terminals_runtime``
        at module load. This method is documentation + a visibility
        touchpoint — it does not grant access the caller didn't already
        have via :meth:`register_resolver`.
        """
        if admin_token is not None:
            # Duck-typed verify() — AdminToken.verify() raises ImportError on
            # stale / mismatched tokens. We don't type-narrow because
            # carl-core doesn't know AdminToken's type.
            verify = getattr(admin_token, "verify", None)
            if not callable(verify):
                raise ValidationError(
                    "admin_token must expose a callable .verify() method",
                    code=type(self)._err_code("invalid_admin_token"),
                    context={"type": type(admin_token).__name__},
                )
            verify()
        self.register_resolver(kind, resolver)

    def unregister_resolver(self, kind: str) -> bool:
        """Remove a registered resolver. Returns True iff one was registered."""
        with self._lock:
            return self._resolvers.pop(kind, None) is not None

    def registered_kinds(self) -> list[str]:
        """Sorted list of kinds with registered resolvers."""
        with self._lock:
            return sorted(self._resolvers.keys())

    # -- internals ------------------------------------------------------

    def _lookup_live(self, ref: H) -> tuple[_VaultEntry[V], H]:
        """Find the (entry, stored_ref) pair or raise. Handles expiry self-clean."""
        cls = type(self)
        with self._lock:
            entry = self._entries.get(ref.ref_id)
            current = self._refs.get(ref.ref_id)
            if entry is None or current is None:
                raise cls._err(
                    "not_found",
                    f"unknown handle: {ref.ref_id}",
                    {"ref_id": str(ref.ref_id)},
                )
            if entry.revoked:
                raise cls._err(
                    "revoked",
                    f"handle {ref.ref_id} was revoked",
                    {"ref_id": str(ref.ref_id)},
                )
            if current.is_expired():
                entry.revoked = True
                entry.resolved_cache = None
                entry.resolved_at = None
                raise cls._err(
                    "expired",
                    f"handle {ref.ref_id} has expired",
                    {"ref_id": str(ref.ref_id), "ttl_s": current.ttl_s},
                )
            return entry, current
