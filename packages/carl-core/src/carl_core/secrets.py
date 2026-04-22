"""Secret-handle primitives (v0.17) — specialization of ``Vault[SecretRef, bytes]``.

Zero-knowledge default: :class:`SecretVault` requires ``privileged=True`` on every
``resolve()`` call so value accesses are visible in code review. All other
lifecycle (put / revoke / exists / list_refs / fingerprint_of / TTL / resolver
chain) is inherited from :class:`carl_core.vault.Vault`.

Design lineage
--------------
- Object-capability model (Mark Miller, E-rights). Handles, not values.
- HashiCorp Vault response wrapping — single-use tokens that unwrap privileged data.
- 1Password ``op://vault/item/field`` URI shape.

Resolver-chain integration. The generic resolver support inherited from the base
means users can register backends for any :class:`SecretKind`:

    vault = SecretVault()
    vault.register_resolver("env", lambda ref: os.environb[ref.uri.split("://")[1]])
    vault.register_resolver("keychain", keychain_resolver)
    # "1password" kind requires user-installed op CLI — register at admin-gate
    # time from terminals-runtime, never top-level in carl-studio.

See ``docs/v17_vault_resolver_chain.md`` for the full user-facing guide.
"""

from __future__ import annotations

import uuid
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, SecretBytes

from carl_core.errors import CARLError, ValidationError
from carl_core.hashing import fingerprint
from carl_core.vault import Vault, VaultError


__all__ = [
    "SecretKind",
    "SecretRef",
    "SecretVault",
    "SecretsError",
    "seal",
    "unseal",
    "generate_box_keypair",
]


# ---------------------------------------------------------------------------
# Kinds
# ---------------------------------------------------------------------------


SecretKind = Literal[
    "mint",
    "env",
    "keychain",
    "vault",
    "clipboard",
]
_KIND_VALUES: frozenset[str] = frozenset(
    {"mint", "env", "keychain", "vault", "clipboard"},
)


# ---------------------------------------------------------------------------
# Error taxonomy — aliases VaultError so `carl.secrets.*` codes remain stable
# and existing `except SecretsError:` catches work unchanged.
# ---------------------------------------------------------------------------


class SecretsError(VaultError):
    """Base for ``carl.secrets.*`` errors.

    Inherits from :class:`VaultError` so generic ``except VaultError`` catches
    both — but the class identity is preserved for code that specifically
    catches ``SecretsError`` and for the ``CARLError.__mro__`` audit trail.
    """

    code = "carl.secrets"


# ---------------------------------------------------------------------------
# SecretRef
# ---------------------------------------------------------------------------


from datetime import datetime, timedelta, timezone  # noqa: E402 — grouped with pydantic


class SecretRef(BaseModel):
    """Opaque handle to a secret value.

    Frozen + ``extra="forbid"`` so the handle shape is inviolable. The ``uri``
    follows the 1Password ``op://vault/item/field`` shape for external-tooling
    compatibility. ``ref_id`` is a server-opaque UUID — different ``ref_id``s
    for otherwise-identical values defeat accidental-dedup attacks.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    ref_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    kind: SecretKind = Field(...)
    uri: str = Field(
        ...,
        description="Stable reference: 'carl://<kind>/<name>' or 'op://vault/item/field'.",
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
        """Public metadata. Never contains the value."""
        expires = self.expired_at()
        return {
            "ref_id": str(self.ref_id),
            "kind": self.kind,
            "uri": self.uri,
            "ttl_s": self.ttl_s,
            "created_at": self.created_at.isoformat(),
            "expires_at": expires.isoformat() if expires is not None else None,
        }


# ---------------------------------------------------------------------------
# SecretVault — thin Vault[SecretRef, bytes] specialization
# ---------------------------------------------------------------------------


class SecretVault(Vault[SecretRef, bytes]):
    """Zero-knowledge vault for credential handles.

    ``resolve()`` demands ``privileged=True`` — a visibility marker forcing
    every dereference to appear in a code diff. Not a cryptographic guard; its
    job is audit discipline, not enforcement. Combine with the audit-chain
    layer (toolkit-level ``SECRET_RESOLVE`` steps) for full traceability.
    """

    _ref_class = SecretRef
    _require_privileged_resolve = True
    _error_prefix: ClassVar[str] = "carl.secrets"
    _error_class: ClassVar[type[VaultError]] = SecretsError

    def put(
        self,
        value: bytes | str,
        *,
        kind: SecretKind,
        uri: str | None = None,
        ttl_s: int | None = None,
    ) -> SecretRef:
        """Store ``value`` and return a :class:`SecretRef` handle.

        ``value`` is wrapped in :class:`SecretBytes` (redacts in repr) and the
        12-hex sha256 fingerprint is cached so audit paths can show identity
        without reading the bytes back.
        """
        if kind not in _KIND_VALUES:
            raise ValidationError(
                f"invalid secret kind: {kind!r}",
                code=type(self)._err_code("invalid_kind"),
                context={"kind": kind, "valid": sorted(_KIND_VALUES)},
            )
        self._validate_ttl(ttl_s)

        raw = value.encode("utf-8") if isinstance(value, str) else bytes(value)
        fp = fingerprint(raw)
        ref = SecretRef(
            kind=kind,
            uri=uri or f"carl://{kind}/{uuid.uuid4().hex[:16]}",
            ttl_s=ttl_s,
        )
        # Wrap bytes in SecretBytes so any incidental repr/logging shows
        # "<redacted>" instead of the literal value. `put_value` stores
        # `bytes(raw)` as-is — we recover the wrapped form via an extra
        # slot on the entry for parity with pre-v0.17 behavior (the entry's
        # `value` remains accessible as bytes via resolve()).
        self.put_value(ref, raw, fingerprint_hex=fp)
        # Shadow-store the SecretBytes wrapper on the entry's `extra` dict so
        # repr() of internal state doesn't leak the value. Not load-bearing;
        # the raw value still sits on `.value` and is what `resolve()` returns.
        with self._lock:
            entry = self._entries[ref.ref_id]
            entry.extra["redacted_view"] = SecretBytes(raw)
        return ref


# ---------------------------------------------------------------------------
# Sealed-box helpers — pynacl asymmetric one-way encryption
# ---------------------------------------------------------------------------


def seal(pubkey: bytes, value: bytes) -> bytes:
    """Seal ``value`` to ``pubkey`` (raw 32-byte Curve25519) via NaCl SealedBox.

    Returns the sealed ciphertext. Only the holder of the matching private key
    can unseal. Used when the agent mints a secret the user is meant to
    receive without the agent retaining read access to its own output.
    """
    try:
        from nacl.public import PublicKey, SealedBox
    except ImportError as exc:  # pragma: no cover
        raise SecretsError(
            "pynacl is required. Install: pip install 'carl-studio[secrets]'",
            code="carl.secrets.backend_unavailable",
            context={"missing": "pynacl"},
            cause=exc,
        ) from exc
    if len(pubkey) != 32:
        raise ValidationError(
            "pubkey must be exactly 32 bytes (Curve25519)",
            code="carl.secrets.invalid_kind",
            context={"got_len": len(pubkey)},
        )
    return bytes(SealedBox(PublicKey(pubkey)).encrypt(value))


def unseal(privkey: bytes, ciphertext: bytes) -> bytes:
    """Inverse of :func:`seal`. ``privkey`` is 32-byte Curve25519 private key."""
    try:
        from nacl.public import PrivateKey, SealedBox
    except ImportError as exc:  # pragma: no cover
        raise SecretsError(
            "pynacl is required. Install: pip install 'carl-studio[secrets]'",
            code="carl.secrets.backend_unavailable",
            context={"missing": "pynacl"},
            cause=exc,
        ) from exc
    if len(privkey) != 32:
        raise ValidationError(
            "privkey must be exactly 32 bytes (Curve25519)",
            code="carl.secrets.invalid_kind",
            context={"got_len": len(privkey)},
        )
    return bytes(SealedBox(PrivateKey(privkey)).decrypt(ciphertext))


def generate_box_keypair() -> tuple[bytes, bytes]:
    """Generate a fresh Curve25519 (privkey, pubkey) for sealed-box use.

    Both halves are raw 32-byte values. Caller is responsible for storing the
    private half in a SecretVault or OS keychain.
    """
    try:
        from nacl.public import PrivateKey
    except ImportError as exc:  # pragma: no cover
        raise SecretsError(
            "pynacl is required. Install: pip install 'carl-studio[secrets]'",
            code="carl.secrets.backend_unavailable",
            context={"missing": "pynacl"},
            cause=exc,
        ) from exc
    priv = PrivateKey.generate()
    return bytes(priv), bytes(priv.public_key)


# Keep CARLError import live — consumers (e.g. tests) used to import it from
# this module. Delete once call sites migrate.
_ = CARLError
