"""KeychainBackend — OS keychain wrapper with name-only semantics.

Wraps the ``keyring`` library so CARL can persist secrets across CLI
sessions via the native OS keychain (macOS Keychain, Linux Secret
Service, Windows Credential Manager). The agent passes
``(service, account)`` tuples through its tool calls — never the
value — and the privileged backend resolves them.

All operations emit ``SECRET_MINT`` / ``SECRET_RESOLVE`` /
``SECRET_REVOKE`` Steps to the :class:`InteractionChain` so the audit
trail records every keychain touch.

Install: ``pip install 'carl-studio[secrets]'`` pulls ``keyring>=24``.
"""

from __future__ import annotations

from carl_core.errors import CARLError
from carl_core.interaction import ActionType, InteractionChain
from carl_core.secrets import SecretRef, SecretVault


__all__ = ["KeychainBackend", "KeychainError"]


class KeychainError(CARLError):
    """Base for ``carl.keychain.*`` error codes."""

    code = "carl.keychain"


class KeychainBackend:
    """Privileged gateway between ``keyring`` and a process-local vault.

    Usage pattern:

        backend = KeychainBackend(vault, chain=chain)
        # Agent-issued tool calls carry names only.
        ref = backend.load_to_vault("carl-camp", "hf_token")
        # Now ref is a SecretRef — downstream tools operate on it.
        # Agent never sees the value.

    The backend re-exports :func:`keyring.get_password` /
    :func:`keyring.set_password` / :func:`keyring.delete_password` through
    its methods so the lazy import pattern is in one place. If
    ``keyring`` is not installed, calls raise
    :class:`KeychainError` with ``code="carl.keychain.unavailable"``.
    """

    def __init__(
        self,
        vault: SecretVault,
        *,
        chain: InteractionChain | None = None,
    ) -> None:
        self.vault = vault
        self.chain = chain

    # -- availability --------------------------------------------------

    @staticmethod
    def available() -> bool:
        """True iff ``keyring`` is importable and a backend is reachable."""
        try:
            import keyring

            # Probe the default backend — some platforms lack a functional one.
            # get_keyring() raises or returns a backend object; if it works,
            # we trust it.
            _ = keyring.get_keyring()
            return True
        except Exception:
            return False

    # -- privileged operations ----------------------------------------

    def load_to_vault(
        self,
        service: str,
        account: str,
        *,
        ttl_s: int | None = None,
    ) -> SecretRef:
        """Pull the value from the OS keychain and put it in the vault.

        Returns a :class:`SecretRef` the agent can pass around. Raises
        :class:`KeychainError` ``code="carl.keychain.not_found"`` when
        the entry is absent.
        """
        keyring_mod = self._keyring_or_raise()
        value = keyring_mod.get_password(service, account)
        if value is None:
            raise KeychainError(
                f"keychain entry not found: service={service!r} account={account!r}",
                code="carl.keychain.not_found",
                context={"service": service, "account": account},
            )
        ref = self.vault.put(
            value,
            kind="keychain",
            uri=f"carl://keychain/{service}/{account}",
            ttl_s=ttl_s,
        )
        self._emit(
            ActionType.SECRET_RESOLVE,
            name="keychain.load",
            service=service,
            account=account,
            ref=ref,
        )
        return ref

    def store(
        self,
        service: str,
        account: str,
        ref: SecretRef,
    ) -> None:
        """Persist a vault-held value into the OS keychain.

        The vault-held bytes get written via ``keyring.set_password``;
        the agent never sees them. Emits a ``SECRET_MINT``-flavored
        Step (a new persisted entry is being created).
        """
        keyring_mod = self._keyring_or_raise()
        value_bytes = self.vault.resolve(ref, privileged=True)
        try:
            value_str = value_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise KeychainError(
                "keychain values must be UTF-8 strings; cannot store raw bytes",
                code="carl.keychain.unsupported_value",
                context={"ref_id": str(ref.ref_id)},
                cause=exc,
            ) from exc
        keyring_mod.set_password(service, account, value_str)
        self._emit(
            ActionType.SECRET_MINT,
            name="keychain.store",
            service=service,
            account=account,
            ref=ref,
        )

    def delete(self, service: str, account: str) -> bool:
        """Remove the keychain entry. Returns True if something was deleted.

        Idempotent — missing entries return False without raising.
        """
        keyring_mod = self._keyring_or_raise()
        try:
            keyring_mod.delete_password(service, account)
        except Exception:
            return False
        self._emit(
            ActionType.SECRET_REVOKE,
            name="keychain.delete",
            service=service,
            account=account,
            ref=None,
        )
        return True

    def exists(self, service: str, account: str) -> bool:
        """Non-privileged probe — returns True if the entry is present."""
        try:
            keyring_mod = self._keyring_or_raise()
        except KeychainError:
            return False
        try:
            return keyring_mod.get_password(service, account) is not None
        except Exception:
            return False

    # -- helpers -------------------------------------------------------

    @staticmethod
    def _keyring_or_raise():  # type: ignore[no-untyped-def]
        try:
            import keyring

            return keyring
        except ImportError as exc:
            raise KeychainError(
                "keyring is not installed. Install with: "
                "pip install 'carl-studio[secrets]'",
                code="carl.keychain.unavailable",
                cause=exc,
            ) from exc

    def _emit(
        self,
        action: ActionType,
        *,
        name: str,
        service: str,
        account: str,
        ref: SecretRef | None,
    ) -> None:
        if self.chain is None:
            return
        output: dict[str, object] = {"service": service, "account": account}
        if ref is not None:
            output["ref_id"] = str(ref.ref_id)
            output["uri"] = ref.uri
            output["fingerprint"] = self.vault.fingerprint_of(ref)
        self.chain.record(
            action,
            name=name,
            input={"service": service, "account": account},
            output=output,
        )
