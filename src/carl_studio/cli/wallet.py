"""Wallet management CLI -- ``carl camp wallet``.

Covers status, creation, passphrase unlock/rotation, and backend introspection.
"""

from __future__ import annotations

import getpass
import json

import typer

wallet_app = typer.Typer(
    name="wallet",
    help="Agent wallet management.",
    no_args_is_help=True,
)


def _prompt_passphrase(label: str = "Wallet passphrase") -> str:
    """Read a passphrase with no echo. Raises typer.Exit on empty."""
    value = getpass.getpass(f"{label}: ")
    if not value:
        typer.echo("Passphrase required.", err=True)
        raise typer.Exit(1)
    return value


@wallet_app.command(name="status")
def wallet_status(
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Show wallet status, balance, and encryption backend."""
    from carl_studio.console import CampConsole
    from carl_studio.wallet import (
        check_agentkit_available,
        check_encryption_available,
        get_backend_name,
        get_wallet_info,
    )

    c = CampConsole()
    info = get_wallet_info()
    backend = get_backend_name()
    locked = backend in {"locked", "unsupported"}

    if json_output:
        if info:
            data: dict[str, object] = info.redacted_dict()
        else:
            data = {"address": None, "network": None, "provider": None, "balance": None}
        data["backend"] = backend
        data["locked"] = locked
        data["encryption_available"] = check_encryption_available()
        typer.echo(json.dumps(data, indent=2))
        raise typer.Exit(0)

    if info:
        c.header("Agent Wallet")
        c.kv("Address", info.address, key_width=14)
        c.kv("Network", info.network, key_width=14)
        c.kv("Provider", info.provider, key_width=14)
        if info.balance:
            c.kv("Balance", info.balance, key_width=14)
        c.kv("Backend", backend, key_width=14)
        c.kv("Locked", "yes" if locked else "no", key_width=14)
    else:
        c.info("No wallet configured.")
        c.blank()
        c.kv("Backend", backend, key_width=14)
        c.kv("Locked", "yes" if locked else "no", key_width=14)
        if not check_encryption_available():
            c.info("Install wallet support: pip install 'carl-studio[wallet]'")
        elif not check_agentkit_available():
            c.info("Install: pip install 'carl-studio[wallet]'")
        else:
            c.info("Create one with: carl camp wallet create")


@wallet_app.command(name="create")
def wallet_create(
    ask_passphrase: bool = typer.Option(
        True,
        "--passphrase/--no-passphrase",
        help="Prompt for a passphrase to encrypt wallet secrets at rest.",
    ),
) -> None:
    """Create a new agent wallet via coinbase-agentkit."""
    from carl_studio.console import CampConsole
    from carl_studio.wallet import create_wallet

    c = CampConsole()
    c.info("Creating agent wallet...")

    passphrase: str | None = None
    if ask_passphrase:
        first = _prompt_passphrase("New wallet passphrase")
        confirm = _prompt_passphrase("Confirm passphrase")
        if first != confirm:
            c.warn("Passphrases did not match.")
            raise typer.Exit(1)
        if len(first) < 8:
            c.warn("Passphrase must be at least 8 characters.")
            raise typer.Exit(1)
        passphrase = first

    info = create_wallet(passphrase=passphrase)
    if info:
        c.ok("Wallet created!")
        c.kv("Address", info.address, key_width=14)
        c.kv("Network", info.network, key_width=14)
        if passphrase is not None:
            c.info("Secrets encrypted at rest (~/.carl/wallet.enc)")
    else:
        c.warn("Wallet creation failed.")
        c.info("Install: pip install 'carl-studio[wallet]'")
        raise typer.Exit(1)


@wallet_app.command(name="unlock")
def wallet_unlock() -> None:
    """Unlock the encrypted wallet store with a passphrase."""
    from carl_studio.console import CampConsole
    from carl_studio.wallet_store import (
        WalletCorrupted,
        WalletLocked,
        WalletStore,
    )

    c = CampConsole()
    passphrase = _prompt_passphrase()

    try:
        store = WalletStore()
        store.unlock(passphrase=passphrase)
    except WalletLocked as exc:
        c.warn(f"Unlock failed: {exc}")
        raise typer.Exit(1) from exc
    except WalletCorrupted as exc:
        c.warn(f"Wallet envelope corrupted: {exc}")
        raise typer.Exit(2) from exc

    c.ok("Wallet unlocked.")
    c.kv("Backend", store.backend, key_width=14)
    c.kv("Keys", str(len(store.keys())), key_width=14)


@wallet_app.command(name="rotate")
def wallet_rotate() -> None:
    """Rotate the wallet passphrase (re-encrypts envelope with new key)."""
    from carl_studio.console import CampConsole
    from carl_studio.wallet_store import (
        WalletCorrupted,
        WalletLocked,
        WalletStore,
    )

    c = CampConsole()
    old = _prompt_passphrase("Current passphrase")
    new_a = _prompt_passphrase("New passphrase")
    new_b = _prompt_passphrase("Confirm new passphrase")
    if new_a != new_b:
        c.warn("New passphrases did not match.")
        raise typer.Exit(1)
    if new_a == old:
        c.warn("New passphrase must differ from the old one.")
        raise typer.Exit(1)
    if len(new_a) < 8:
        c.warn("Passphrase must be at least 8 characters.")
        raise typer.Exit(1)

    try:
        store = WalletStore()
        store.unlock(passphrase=old)
        store.rotate_passphrase(old, new_a)
    except WalletLocked as exc:
        c.warn(f"Rotation failed: {exc}")
        raise typer.Exit(1) from exc
    except WalletCorrupted as exc:
        c.warn(f"Wallet envelope corrupted: {exc}")
        raise typer.Exit(2) from exc

    c.ok("Passphrase rotated.")
