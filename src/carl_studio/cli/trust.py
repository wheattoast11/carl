from __future__ import annotations

from pathlib import Path
from typing import Any

import typer

import carl_studio.cli.ui as ui
from carl_studio.project_context import current
from carl_studio.trust import get_trust_registry


trust_app = typer.Typer(
    name="trust",
    help="Manage bare-entry trust prompts for project-aware CLI routing.",
    no_args_is_help=True,
)


def _console() -> Any:
    from carl_studio.console import get_console

    return get_console()


def _detect_project_root() -> Path | None:
    context = current(Path.cwd())
    if context is None:
        return None
    return context.root


@trust_app.command("status")
def status_cmd() -> None:
    """Show current trust settings and acknowledged project root."""
    registry = get_trust_registry()
    state = registry.get()
    c = _console()
    c.header("Trust status")
    c.kv("Enabled", "yes" if state.enabled else "no", key_width=20)
    c.kv(
        "Acknowledged root",
        state.acknowledged_project_root or "(none)",
        key_width=20,
    )
    detected = _detect_project_root()
    c.kv("Current project", str(detected) if detected is not None else "(not in project)", key_width=20)


@trust_app.command("enable")
def enable_cmd() -> None:
    """Enable the bare-entry trust prompt."""
    registry = get_trust_registry()
    state = registry.set_enabled(True)
    c = _console()
    c.ok("Trust pre-check enabled.")
    c.kv(
        "Acknowledged root",
        state.acknowledged_project_root or "(none)",
        key_width=20,
    )


@trust_app.command("disable")
def disable_cmd(
    force: bool = typer.Option(
        False,
        "--force",
        help="Disable without an interactive confirmation prompt.",
    )
) -> None:
    """Disable the bare-entry trust prompt globally."""
    if not force:
        approved = ui.confirm(
            "Disable trust pre-check for future bare `carl` entry?",
            default=False,
        )
        if not approved:
            raise typer.Exit(1)
    registry = get_trust_registry()
    state = registry.set_enabled(False)
    c = _console()
    c.warn("Trust pre-check disabled.")
    c.kv(
        "Acknowledged root",
        state.acknowledged_project_root or "(none)",
        key_width=20,
    )


@trust_app.command("acknowledge")
def acknowledge_cmd(
    path: Path | None = typer.Argument(
        None,
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=False,
        help="Project root to acknowledge. Defaults to the detected current project.",
    )
) -> None:
    """Persist trust for a specific project root.

    Only one acknowledged root is stored at a time — acknowledging a new
    path silently replaces the previous one. The prior root is surfaced
    when this happens so users notice the eviction.
    """
    target = path if path is not None else _detect_project_root()
    if target is None:
        typer.echo("Not inside a CARL project and no path was provided.", err=True)
        raise typer.Exit(2)
    registry = get_trust_registry()
    prior = registry.current_acknowledged_root()
    state = registry.trust_root(target)
    c = _console()
    c.ok("Project acknowledged for bare entry.")
    c.kv("Acknowledged root", state.acknowledged_project_root or "(none)", key_width=20)
    if prior is not None and str(prior) != state.acknowledged_project_root:
        c.info(f"  Replaced previous acknowledged root: {prior}")


@trust_app.command("reset")
def reset_cmd(
    force: bool = typer.Option(
        False,
        "--force",
        help="Reset stored trust state without an interactive confirmation prompt.",
    )
) -> None:
    """Reset trust settings to defaults and clear any acknowledged root."""
    if not force:
        approved = ui.confirm(
            "Reset trust settings and clear the acknowledged project root?",
            default=False,
        )
        if not approved:
            raise typer.Exit(1)
    registry = get_trust_registry()
    state = registry.reset()
    c = _console()
    c.ok("Trust settings reset.")
    c.kv("Enabled", "yes" if state.enabled else "no", key_width=20)
    c.kv("Acknowledged root", "(none)", key_width=20)


__all__ = ["trust_app", "status_cmd", "enable_cmd", "disable_cmd", "acknowledge_cmd", "reset_cmd"]
