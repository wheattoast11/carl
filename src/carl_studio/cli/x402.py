"""x402 CLI — ``carl camp x402``.

x402 HTTP payment rail configuration and status.
Register in wiring.py via camp_app.add_typer(x402_app, name='x402').
"""

from __future__ import annotations

import json

import typer

x402_app = typer.Typer(
    name="x402",
    help="x402 HTTP payment rail configuration and status.",
    no_args_is_help=True,
)


@x402_app.command(name="status")
def x402_status(
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Show x402 payment rail status and configuration."""
    from carl_studio.console import get_console
    from carl_studio.x402 import load_x402_config
    from carl_studio.x402_sdk import sdk_available

    c = get_console()
    config = load_x402_config()
    has_sdk = sdk_available()

    if json_output:
        data = config.model_dump()
        data["sdk_available"] = has_sdk
        data["client"] = "x402-sdk" if has_sdk else "urllib-builtin"
        typer.echo(json.dumps(data, indent=2))
        raise typer.Exit(0)

    c.header("x402 Payment Rail")
    c.kv("Enabled", "yes" if config.enabled else "not configured", key_width=20)
    c.kv("Client", "x402-sdk (official)" if has_sdk else "urllib (builtin)", key_width=20)
    c.kv("Wallet", config.wallet_address or "(not set)", key_width=20)
    c.kv("Chain", config.chain, key_width=20)
    c.kv("Token", config.payment_token, key_width=20)
    c.kv("Facilitator", config.facilitator_url or "(not set)", key_width=20)
    c.kv("Auto-approve", f"< ${config.auto_approve_below:.2f}" if config.auto_approve_below else "off", key_width=20)

    if has_sdk:
        c.blank()
        c.ok("Official x402 SDK detected -- auto-intercept and facilitator APIs available.")
    elif not config.wallet_address:
        c.blank()
        c.info("Configure: carl camp x402 configure --wallet <addr> --facilitator <url>")
        c.info("Install SDK: pip install 'carl-studio[x402]'")


@x402_app.command(name="configure")
def x402_configure(
    wallet: str = typer.Option("", "--wallet", help="Wallet address"),
    chain: str = typer.Option("", "--chain", help="Chain identifier (e.g. base, ethereum)"),
    facilitator: str = typer.Option("", "--facilitator", help="Facilitator URL"),
    token: str = typer.Option("", "--token", help="Payment token (e.g. USDC)"),
    auto_approve: float = typer.Option(
        -1.0, "--auto-approve-below", help="Auto-approve threshold in USD (0 to disable)"
    ),
) -> None:
    """Configure x402 payment rail settings."""
    from carl_studio.console import get_console
    from carl_studio.x402 import load_x402_config, save_x402_config

    c = get_console()
    config = load_x402_config()

    if wallet:
        config.wallet_address = wallet
    if chain:
        config.chain = chain
    if facilitator:
        config.facilitator_url = facilitator
    if token:
        config.payment_token = token
    if auto_approve >= 0:
        config.auto_approve_below = auto_approve

    config.enabled = bool(config.wallet_address and config.facilitator_url)
    save_x402_config(config)

    c.ok("x402 configuration saved.")
    c.kv("Enabled", "yes" if config.enabled else "needs wallet + facilitator", key_width=16)
    c.kv("Wallet", config.wallet_address or "(not set)", key_width=16)
    c.kv("Facilitator", config.facilitator_url or "(not set)", key_width=16)


@x402_app.command(name="check")
def x402_check(
    url: str = typer.Argument(..., help="URL to check for x402 capability"),
) -> None:
    """Check if a URL supports x402 payments."""
    from carl_studio.console import get_console
    from carl_studio.x402 import X402Client, load_x402_config

    c = get_console()
    config = load_x402_config()
    client = X402Client(config)

    req = client.check_x402(url)
    if req is None:
        c.info(f"No x402 payment required for {url}")
    else:
        c.ok(f"x402 payment required for {url}")
        c.kv("Amount", req.amount, key_width=14)
        c.kv("Token", req.token, key_width=14)
        c.kv("Chain", req.chain, key_width=14)
        c.kv("Recipient", req.recipient, key_width=14)
        c.kv("Facilitator", req.facilitator, key_width=14)
