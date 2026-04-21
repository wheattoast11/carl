"""CryptoRandomMinter — mint fresh secrets directly into a SecretVault.

Every mint emits a ``SECRET_MINT`` Step on the caller-supplied
:class:`~carl_core.interaction.InteractionChain` so the audit trail
captures every value-creation event without the value itself.

The minter never returns the raw value — only the opaque
:class:`~carl_core.secrets.SecretRef` handle. Downstream tooling
resolves through the vault with explicit ``privileged=True``.
"""

from __future__ import annotations

import base64
import secrets as _stdlib_secrets
import uuid
from typing import Literal

from carl_core.interaction import ActionType, InteractionChain
from carl_core.secrets import (
    SecretRef,
    SecretVault,
    generate_box_keypair,
)


__all__ = ["CryptoRandomMinter", "SecretMintKind"]


SecretMintKind = Literal["hex", "base64", "uuid", "ed25519_keypair"]


class CryptoRandomMinter:
    """Factory for cryptographically-random secrets backed by a vault.

    Every method returns a :class:`SecretRef` and writes a ``SECRET_MINT``
    Step to the chain. The raw value exists only inside the vault's
    ``SecretBytes`` wrapper; callers that need the bytes must call
    ``vault.resolve(ref, privileged=True)``.

    Args:
        vault: Where the minted secrets land. Typically a process-local
            :class:`SecretVault`. Re-use one across the lifetime of a CLI
            session; tearing it down drops all minted secrets.
        chain: The :class:`InteractionChain` that receives audit Steps.
            When ``None``, mint operations are silent — use only for
            tests or scripts that truly do not need audit.
    """

    def __init__(
        self,
        vault: SecretVault,
        *,
        chain: InteractionChain | None = None,
    ) -> None:
        self.vault = vault
        self.chain = chain

    # -- factories ------------------------------------------------------

    def mint_hex(
        self,
        nbytes: int = 32,
        *,
        ttl_s: int | None = None,
        uri: str | None = None,
    ) -> SecretRef:
        """Generate ``nbytes`` of cryptographic-random entropy, hex-encoded.

        ``nbytes=32`` → 64 hex chars (256 bits of entropy). Suitable for
        API keys, signing secrets, CSRF tokens.
        """
        if nbytes < 8:
            raise ValueError(f"nbytes must be >= 8, got {nbytes}")
        raw = _stdlib_secrets.token_hex(nbytes).encode("utf-8")
        ref = self.vault.put(raw, kind="mint", uri=uri, ttl_s=ttl_s)
        self._emit_mint(ref, kind_detail="hex", nbytes=nbytes)
        return ref

    def mint_base64(
        self,
        nbytes: int = 32,
        *,
        ttl_s: int | None = None,
        uri: str | None = None,
    ) -> SecretRef:
        """Generate ``nbytes`` of cryptographic-random entropy, base64-encoded.

        ``nbytes=32`` → ~44 chars (256 bits). Equivalent to
        ``openssl rand -base64 32``.
        """
        if nbytes < 8:
            raise ValueError(f"nbytes must be >= 8, got {nbytes}")
        raw = base64.b64encode(_stdlib_secrets.token_bytes(nbytes))
        ref = self.vault.put(raw, kind="mint", uri=uri, ttl_s=ttl_s)
        self._emit_mint(ref, kind_detail="base64", nbytes=nbytes)
        return ref

    def mint_uuid(
        self,
        *,
        ttl_s: int | None = None,
        uri: str | None = None,
    ) -> SecretRef:
        """Generate a fresh UUID4 as the secret value.

        Lower entropy than ``mint_hex``/``mint_base64`` but useful when
        the downstream system expects a UUID-shaped handle.
        """
        raw = str(uuid.uuid4()).encode("utf-8")
        ref = self.vault.put(raw, kind="mint", uri=uri, ttl_s=ttl_s)
        self._emit_mint(ref, kind_detail="uuid", nbytes=36)
        return ref

    def mint_ed25519_keypair(
        self,
        *,
        ttl_s: int | None = None,
        uri_prefix: str | None = None,
    ) -> tuple[SecretRef, SecretRef]:
        """Generate a fresh Curve25519 (privkey, pubkey) pair for sealed boxes.

        Both halves land in the vault as separate :class:`SecretRef`s.
        The private half has ``kind="mint"`` (ephemeral by default —
        callers should transfer it to long-term storage via the
        :class:`KeychainBackend` if needed). The public half is still
        stored as a ``SecretRef`` for consistency even though it's not
        secret — this lets downstream tools use the same ``ref_id``
        handoff pattern regardless of half.

        Requires ``pynacl``.
        """
        priv, pub = generate_box_keypair()
        priv_uri = (
            f"{uri_prefix}/priv" if uri_prefix else None
        )
        pub_uri = f"{uri_prefix}/pub" if uri_prefix else None
        priv_ref = self.vault.put(priv, kind="mint", uri=priv_uri, ttl_s=ttl_s)
        pub_ref = self.vault.put(pub, kind="mint", uri=pub_uri, ttl_s=ttl_s)
        # Two Steps — priv + pub — so the audit captures both halves.
        self._emit_mint(priv_ref, kind_detail="ed25519_privkey", nbytes=32)
        self._emit_mint(pub_ref, kind_detail="ed25519_pubkey", nbytes=32)
        return priv_ref, pub_ref

    # -- audit ----------------------------------------------------------

    def _emit_mint(
        self,
        ref: SecretRef,
        *,
        kind_detail: str,
        nbytes: int,
    ) -> None:
        if self.chain is None:
            return
        fp = self.vault.fingerprint_of(ref)
        self.chain.record(
            ActionType.SECRET_MINT,
            name=f"mint.{kind_detail}",
            input={"kind_detail": kind_detail, "nbytes": nbytes},
            output={
                "ref_id": str(ref.ref_id),
                "uri": ref.uri,
                "fingerprint": fp,
                "ttl_s": ref.ttl_s,
            },
        )
