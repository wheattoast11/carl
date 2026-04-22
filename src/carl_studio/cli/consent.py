"""Consent CLI — ``carl camp consent``.

Privacy-first consent management. All flags default off.
Register in wiring.py via camp_app.add_typer(consent_app, name='consent').
"""

from __future__ import annotations

import json

import typer

from carl_studio.consent import CONSENT_KEYS, ConsentError, ConsentManager

consent_app = typer.Typer(
    name="consent",
    help="Privacy consent management -- what data you share.",
    no_args_is_help=True,
)


@consent_app.command(name="show")
def consent_show(
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Show current consent preferences."""
    from carl_studio.console import get_console

    mgr = ConsentManager()
    state = mgr.load()

    if json_output:
        typer.echo(json.dumps(state.model_dump(), indent=2))
        raise typer.Exit(0)

    c = get_console()
    c.header("Privacy Consent")
    for key in sorted(CONSENT_KEYS):
        flag = getattr(state, key)
        status = "on" if flag.enabled else "off"
        since = f" (since {flag.changed_at})" if flag.changed_at else " (default)"
        c.kv(key, f"{status}{since}", key_width=24)
    c.blank()
    c.info("Update: carl camp consent update <key> --enable/--disable")
    c.info("Reset all: carl camp consent reset")


@consent_app.command(name="update")
def consent_update(
    key: str = typer.Argument(
        ...,
        help=f"Consent key: {', '.join(sorted(CONSENT_KEYS))}",
    ),
    enable: bool = typer.Option(True, "--enable/--disable", help="Enable or disable"),
) -> None:
    """Update a consent preference."""
    from carl_studio.console import get_console

    c = get_console()
    mgr = ConsentManager()
    try:
        state = mgr.update(key, enable)
    except ConsentError as exc:
        c.error(str(exc))
        raise typer.Exit(1)

    flag = getattr(state, key)
    c.ok(f"{key} = {'on' if flag.enabled else 'off'}")


@consent_app.command(name="reset")
def consent_reset(
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
) -> None:
    """Reset all consents to off (privacy-first defaults)."""
    from carl_studio.console import get_console

    from carl_studio.cli import ui

    c = get_console()
    if not force:
        if not ui.confirm("  Reset all consent flags to off?", default=False):
            raise typer.Exit(0)

    mgr = ConsentManager()
    mgr.all_off()
    c.ok("All consent flags reset to off.")
