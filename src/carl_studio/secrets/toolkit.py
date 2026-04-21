"""SecretsToolkit — unified façade for the five agent-callable ops.

Bundles a :class:`SecretVault` + :class:`CryptoRandomMinter` +
:class:`KeychainBackend` + :class:`ClipboardBridge` into a single
object whose methods can be registered directly with the
:class:`~carl_studio.tool_dispatcher.ToolDispatcher`.

Every method takes plain-dict tool arguments and returns a plain dict
(serializable to JSON, safe for inclusion in the
:class:`~carl_core.interaction.InteractionChain`). Handles are
identified by their ``ref_id`` hex UUID in the tool surface — the
toolkit translates to :class:`SecretRef` internally.

The five tool methods (matching the v0.16 design doc):

* :meth:`mint_secret` — mint a fresh random value into the vault
* :meth:`copy_to_clipboard` — scoped TTL-bounded clipboard write
* :meth:`revoke_secret` — explicit handle invalidation
* :meth:`hash_value` — derive a stable fingerprint via the vault
* :meth:`list_secrets` — metadata-only enumeration
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from carl_core.errors import ValidationError
from carl_core.interaction import InteractionChain
from carl_core.secrets import SecretRef, SecretVault, SecretsError

from .clipboard import ClipboardBridge
from .keychain import KeychainBackend
from .minter import CryptoRandomMinter


__all__ = ["SecretsToolkit"]


@dataclass
class SecretsToolkit:
    """Unified entrypoint bundling vault + minter + keychain + clipboard.

    Construct once per CLI session; pass the same instance to the
    :class:`ToolDispatcher` registrations. Every method is safe to call
    from agent-driven tool loops because inputs are dict-shaped and
    outputs never contain value bytes.
    """

    vault: SecretVault
    minter: CryptoRandomMinter
    keychain: KeychainBackend
    clipboard: ClipboardBridge
    chain: InteractionChain | None = None

    @classmethod
    def build(
        cls,
        chain: InteractionChain | None = None,
        *,
        clipboard_default_ttl_s: int = 30,
    ) -> SecretsToolkit:
        """One-call construction: fresh vault + wired-up components.

        The ``chain`` is threaded through to all four components so every
        operation emits a single matching Step.
        """
        vault = SecretVault()
        return cls(
            vault=vault,
            minter=CryptoRandomMinter(vault, chain=chain),
            keychain=KeychainBackend(vault, chain=chain),
            clipboard=ClipboardBridge(
                vault,
                chain=chain,
                default_ttl_s=clipboard_default_ttl_s,
            ),
            chain=chain,
        )

    # -- tool methods (agent-facing) ------------------------------------

    def mint_secret(
        self,
        kind: str = "hex",
        *,
        nbytes: int = 32,
        ttl_s: int | None = None,
    ) -> dict[str, Any]:
        """Mint a new random secret into the vault.

        Args:
            kind: ``"hex"`` | ``"base64"`` | ``"uuid"`` | ``"ed25519_keypair"``
            nbytes: entropy length for hex/base64. Ignored for uuid/ed25519.
            ttl_s: Optional auto-expire TTL in seconds.

        Returns:
            ``{"ref_id", "uri", "fingerprint", "kind", "ttl_s"}`` for
            simple kinds; for ``ed25519_keypair``, ``{"priv_ref_id",
            "pub_ref_id", ..., "kind": "ed25519_keypair"}``.
        """
        if kind == "hex":
            ref = self.minter.mint_hex(nbytes=nbytes, ttl_s=ttl_s)
        elif kind == "base64":
            ref = self.minter.mint_base64(nbytes=nbytes, ttl_s=ttl_s)
        elif kind == "uuid":
            ref = self.minter.mint_uuid(ttl_s=ttl_s)
        elif kind == "ed25519_keypair":
            priv_ref, pub_ref = self.minter.mint_ed25519_keypair(ttl_s=ttl_s)
            return {
                "kind": "ed25519_keypair",
                "priv_ref_id": str(priv_ref.ref_id),
                "pub_ref_id": str(pub_ref.ref_id),
                "priv_uri": priv_ref.uri,
                "pub_uri": pub_ref.uri,
                "priv_fingerprint": self.vault.fingerprint_of(priv_ref),
                "pub_fingerprint": self.vault.fingerprint_of(pub_ref),
                "ttl_s": ttl_s,
            }
        else:
            raise ValidationError(
                f"unknown mint kind: {kind!r}. "
                "Valid: hex | base64 | uuid | ed25519_keypair",
                code="carl.secrets.invalid_kind",
                context={"kind": kind},
            )

        return {
            "ref_id": str(ref.ref_id),
            "uri": ref.uri,
            "fingerprint": self.vault.fingerprint_of(ref),
            "kind": kind,
            "ttl_s": ttl_s,
        }

    def copy_to_clipboard(
        self,
        ref_id: str,
        *,
        ttl_s: int | None = None,
    ) -> dict[str, Any]:
        """Copy a vault-held value to the clipboard with TTL auto-wipe.

        Args:
            ref_id: UUID-hex identifying the handle.
            ttl_s: Override the bridge default (default 30s).

        Returns:
            ``{"fingerprint", "expires_at", "ttl_s"}`` — never the value.
        """
        ref = self._ref_from_id(ref_id)
        return self.clipboard.write_from_ref(ref, ttl_s=ttl_s)

    def revoke_secret(self, ref_id: str) -> dict[str, Any]:
        """Invalidate a handle. Returns ``{"ref_id", "revoked"}``."""
        ref = self._ref_from_id(ref_id)
        revoked = self.vault.revoke(ref)
        if self.chain is not None:
            from carl_core.interaction import ActionType

            self.chain.record(
                ActionType.SECRET_REVOKE,
                name="toolkit.revoke_secret",
                input={"ref_id": ref_id},
                output={"revoked": revoked},
            )
        return {"ref_id": ref_id, "revoked": revoked}

    def hash_value(
        self,
        ref_id: str,
        *,
        algorithm: str = "sha256-12",
    ) -> dict[str, Any]:
        """Return a stable hash/fingerprint of the handle's value.

        The ``sha256-12`` algorithm is the 12-hex CARL canonical; other
        values dispatch through :mod:`hashlib` via the privileged
        resolve path. Returns ``{"ref_id", "algorithm", "fingerprint"}``.
        """
        ref = self._ref_from_id(ref_id)
        if algorithm == "sha256-12":
            return {
                "ref_id": ref_id,
                "algorithm": algorithm,
                "fingerprint": self.vault.fingerprint_of(ref),
            }

        import hashlib

        try:
            h = hashlib.new(algorithm)
        except ValueError as exc:
            raise ValidationError(
                f"unknown hash algorithm: {algorithm!r}",
                code="carl.secrets.invalid_kind",
                context={"algorithm": algorithm},
                cause=exc,
            ) from exc

        value = self.vault.resolve(ref, privileged=True)
        h.update(value)
        digest = h.hexdigest()
        # Emit a SECRET_RESOLVE Step — we did dereference, even though
        # we immediately hashed.
        if self.chain is not None:
            from carl_core.interaction import ActionType

            self.chain.record(
                ActionType.SECRET_RESOLVE,
                name="toolkit.hash_value",
                input={"ref_id": ref_id, "algorithm": algorithm},
                output={
                    "ref_id": ref_id,
                    "algorithm": algorithm,
                    "digest_len": len(digest),
                    "fingerprint": self.vault.fingerprint_of(ref),
                },
            )
        return {
            "ref_id": ref_id,
            "algorithm": algorithm,
            "fingerprint": digest,
        }

    def list_secrets(self) -> dict[str, Any]:
        """Metadata-only enumeration of active handles.

        Returns ``{"count", "refs": [{ref_id, uri, kind, fingerprint,
        ttl_s, created_at, expires_at}, ...]}``. Never value bytes.
        """
        now = datetime.now(timezone.utc)
        refs = self.vault.list_refs()
        entries: list[dict[str, Any]] = []
        for ref in refs:
            d = ref.describe()
            d["fingerprint"] = self.vault.fingerprint_of(ref)
            expires = ref.expired_at()
            if expires is None:
                d["ttl_remaining_s"] = None
            else:
                d["ttl_remaining_s"] = int((expires - now).total_seconds())
            entries.append(d)
        return {"count": len(entries), "refs": entries}

    # -- internals -----------------------------------------------------

    def _ref_from_id(self, ref_id: str) -> SecretRef:
        """Look up a :class:`SecretRef` by its hex-UUID ``ref_id``.

        Raises :class:`SecretsError` ``carl.secrets.not_found`` for
        unknown ids.
        """
        try:
            parsed = uuid.UUID(ref_id)
        except ValueError as exc:
            raise ValidationError(
                f"invalid ref_id: {ref_id!r}",
                code="carl.secrets.invalid_kind",
                context={"ref_id": ref_id},
                cause=exc,
            ) from exc
        for ref in self.vault.list_refs():
            if ref.ref_id == parsed:
                return ref
        raise SecretsError(
            f"unknown secret handle: {ref_id}",
            code="carl.secrets.not_found",
            context={"ref_id": ref_id},
        )
