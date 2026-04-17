"""Wallet management CLI -- ``carl camp wallet``.

Agent wallet status, creation, and integration info.
Register in wiring.py via camp_app.add_typer(wallet_app, name='wallet').
"""

from __future__ import annotations

import json

import typer

wallet_app = typer.Typer(
    name="wallet",
    help="Agent wallet management.",
    no_args_is_help=True,
)


@wallet_app.command(name="status")
def wallet_status(
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Show wallet status and balance."""
    from carl_studio.console import get_console
    from carl_studio.wallet import check_agentkit_available, get_wallet_info

    c = get_console()
    info = get_wallet_info()

    if json_output:
        if info:
            data = {
                "address": info.address,
                "network": info.network,
                "provider": info.provider,
                "balance": info.balance,
            }
        else:
            data = {"address": None, "network": None, "provider": None, "balance": None}
        typer.echo(json.dumps(data, indent=2))
        raise typer.Exit(0)

    if info:
        c.header("Agent Wallet")
        c.kv("Address", info.address, key_width=14)
        c.kv("Network", info.network, key_width=14)
        c.kv("Provider", info.provider, key_width=14)
        if info.balance:
            c.kv("Balance", info.balance, key_width=14)
    else:
        c.info("No wallet configured.")
        c.blank()
        if not check_agentkit_available():
            c.info("Install wallet support: pip install 'carl-studio[wallet]'")
        else:
            c.info("Link wallet via carl.camp or use: carl camp wallet create")


@wallet_app.command(name="create")
def wallet_create() -> None:
    """Create a new agent wallet via coinbase-agentkit."""
    from carl_studio.console import get_console
    from carl_studio.wallet import create_wallet

    c = get_console()
    c.info("Creating agent wallet...")

    info = create_wallet()
    if info:
        c.ok("Wallet created!")
        c.kv("Address", info.address, key_width=14)
        c.kv("Network", info.network, key_width=14)
    else:
        c.warn("Wallet creation failed.")
        c.info("Install: pip install 'carl-studio[wallet]'")
        raise typer.Exit(1)
