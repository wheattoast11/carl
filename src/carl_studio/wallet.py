"""Agent wallet integration.

Wallet addresses are stored unencrypted (they are public); secrets
(private keys, mnemonics) are routed through
:class:`carl_studio.wallet_store.WalletStore`, which encrypts them at rest
with a user-controlled passphrase or an OS keyring master key.

Install wallet support::

    pip install 'carl-studio[wallet]'
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from carl_core.errors import CredentialError

from carl_studio.wallet_store import (
    INSTALL_HINT,
    WalletCorrupted,
    WalletLocked,
    WalletStore,
    keyring_available,
)

logger = logging.getLogger(__name__)

#: Keys used inside :class:`WalletStore` for agent secrets. Public names only —
#: the values themselves are never logged.
WALLET_PRIVATE_KEY = "private_key"
WALLET_MNEMONIC = "mnemonic"
WALLET_ADDRESS = "address"


@dataclass
class WalletInfo:
    """Agent wallet state.

    Only public, non-sensitive fields live here. Secrets live in the encrypted
    :class:`WalletStore` and are never placed on this dataclass.
    """

    address: str
    network: str = "base"
    provider: str = "unknown"
    balance: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict[str, Any])

    def redacted_dict(self) -> dict[str, Any]:
        """Safe JSON projection (no secrets)."""
        return {
            "address": self.address,
            "network": self.network,
            "provider": self.provider,
            "balance": self.balance,
            "metadata": dict(self.metadata),
        }


def _wallet_store_from_env() -> WalletStore | None:
    """Return an *unlocked* store when possible, else None.

    Uses env passphrase first, keyring second. Never prompts.
    """
    try:
        store = WalletStore()
    except Exception as exc:  # pragma: no cover - filesystem failure
        logger.debug("WalletStore init failed: %s", exc)
        return None
    try:
        store.unlock()
    except WalletLocked:
        return None
    except WalletCorrupted as exc:
        logger.warning("Wallet envelope is corrupted: %s", exc)
        return None
    return store


def get_wallet_info() -> WalletInfo | None:
    """Get wallet info from store, camp profile, or local config.

    Resolution order:
      1. Encrypted :class:`WalletStore` (requires passphrase or keyring).
      2. Camp profile metadata.
      3. Local x402 config.
    Returns ``None`` if nothing is configured.
    """
    # 1. Encrypted store — only succeeds when unlockable.
    store = _wallet_store_from_env()
    if store is not None:
        try:
            addr = store.get(WALLET_ADDRESS)
        except WalletLocked:
            addr = None
        if addr:
            return WalletInfo(
                address=addr,
                network="base",
                provider="carl-store",
            )

    try:
        from carl_studio.db import LocalDB

        db = LocalDB()
    except Exception:
        return None

    try:
        jwt = db.get_auth("jwt")
        if not jwt:
            return None

        # 2. Camp profile for wallet address
        try:
            from carl_studio.camp import (
                DEFAULT_CARL_CAMP_SUPABASE_URL,
                fetch_camp_profile,
            )

            supabase_url = db.get_config("supabase_url") or DEFAULT_CARL_CAMP_SUPABASE_URL
            profile = fetch_camp_profile(jwt, supabase_url)
            wallet_addr = profile.metadata.get("wallet_address", "")
            if wallet_addr:
                return WalletInfo(
                    address=wallet_addr,
                    network=profile.metadata.get("x402_chain", "base"),
                    provider="camp",
                )
        except Exception:
            pass

        # 3. Local x402 config
        try:
            from carl_studio.x402 import load_x402_config

            config = load_x402_config(db=db)
            if config.wallet_address:
                return WalletInfo(
                    address=config.wallet_address,
                    network=config.chain,
                    provider="x402-local",
                )
        except Exception:
            pass

    except Exception as exc:
        logger.debug("get_wallet_info failed: %s", exc)

    return None


def check_agentkit_available() -> bool:
    """Check if coinbase-agentkit is installed."""
    try:
        import coinbase_agentkit  # noqa: F401

        return True
    except ImportError:
        return False


def check_encryption_available() -> bool:
    """Check if the ``cryptography`` library is importable."""
    try:
        import cryptography  # type: ignore[import-untyped]  # noqa: F401

        return True
    except ImportError:
        return False


def get_backend_name() -> str:
    """Report the currently-usable encryption backend: fernet / keyring / locked / unsupported."""
    if not check_encryption_available():
        return "unsupported"
    if keyring_available():
        return "keyring"
    store = _wallet_store_from_env()
    if store is not None and not store.is_locked:
        return store.backend
    return "locked"


def create_wallet(passphrase: str | None = None) -> WalletInfo | None:
    """Create a new agent wallet via coinbase-agentkit.

    When ``passphrase`` is provided (or ``CARL_WALLET_PASSPHRASE`` is set, or a
    keyring master key is stored), the resulting secrets are encrypted at rest
    in :class:`WalletStore`. If no key material is available, the address is
    returned but secret persistence is skipped with a warning.

    Requires: ``pip install 'carl-studio[wallet]'``
    """
    try:
        address, wallet_obj = _agentkit_create_wallet()
    except _AgentkitUnavailable:
        logger.info("coinbase-agentkit not installed. Install: %s", INSTALL_HINT)
        return None
    except Exception as exc:
        logger.warning("Wallet creation failed: %s", exc)
        return None

    if not isinstance(address, str):
        logger.warning("Agentkit returned a non-string address; skipping persistence.")
        return None

    # Attempt to persist secrets. Non-fatal — the address itself is useful even
    # without key storage.
    try:
        store = WalletStore()
        store.unlock(passphrase=passphrase)
        store.put(WALLET_ADDRESS, address)
        secret = _agentkit_export_private_key(wallet_obj)
        if secret is not None:
            store.put(WALLET_PRIVATE_KEY, secret)
    except WalletLocked:
        logger.warning(
            "Wallet created but no passphrase supplied — secrets not persisted. "
            "Run: carl camp wallet unlock"
        )
    except CredentialError as exc:
        logger.warning("Wallet secret persistence failed: %s", exc)

    return WalletInfo(
        address=address,
        network="base",
        provider="agentkit",
    )


class _AgentkitUnavailable(RuntimeError):
    """Internal signal that coinbase-agentkit is not installed."""


def _agentkit_create_wallet() -> tuple[Any, Any]:
    """Import agentkit lazily and return ``(address, wallet_obj)``.

    Raises :class:`_AgentkitUnavailable` on ImportError. Any other exception
    propagates to the caller, which maps it to ``None``.
    """
    try:
        mod: Any = __import__("coinbase_agentkit")
    except ImportError as exc:
        raise _AgentkitUnavailable("coinbase-agentkit not installed") from exc

    agentkit_cls: Any = getattr(mod, "AgentKit")
    agentkit_config_cls: Any = getattr(mod, "AgentKitConfig")
    config: Any = agentkit_config_cls(network_id="base-mainnet")
    kit: Any = agentkit_cls(config)
    wallet_obj: Any = kit.wallet
    address: Any = wallet_obj.default_address.address_id
    return address, wallet_obj


def _agentkit_export_private_key(wallet_obj: Any) -> str | None:
    """Best-effort private-key extraction. Returns a str or None."""
    if wallet_obj is None:
        return None
    pk_getter: Any = getattr(wallet_obj, "export", None)
    if pk_getter is None:
        pk_getter = getattr(wallet_obj, "private_key", None)
    if callable(pk_getter):
        try:
            secret: Any = pk_getter()
        except Exception as exc:
            logger.debug("private key export skipped: %s", exc)
            return None
        return secret if isinstance(secret, str) else None
    if isinstance(pk_getter, str):
        return pk_getter
    return None


__all__ = [
    "WALLET_ADDRESS",
    "WALLET_MNEMONIC",
    "WALLET_PRIVATE_KEY",
    "WalletInfo",
    "check_agentkit_available",
    "check_encryption_available",
    "create_wallet",
    "get_backend_name",
    "get_wallet_info",
]
