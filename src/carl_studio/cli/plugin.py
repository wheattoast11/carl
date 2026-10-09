"""Local CARL plugin installation and source reconciliation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer
from carl_core.errors import CARLError

from carl_studio.console import get_console

plugin_app = typer.Typer(help="Install and maintain the local carl plugin", no_args_is_help=True)

if TYPE_CHECKING:
    from carl_studio.plugin import PluginInstaller


def _installer() -> PluginInstaller:
    from carl_studio.plugin import PluginInstaller, source_root

    return PluginInstaller(source_root())


@plugin_app.command("install")
def install(host: Annotated[list[str] | None, typer.Option("--host")] = None) -> None:
    """Install detected hosts and standalone carl aliases."""
    try:
        result = _installer().install(host)
    except CARLError as exc:
        get_console().error(f"{exc.code}: {exc}")
        raise typer.Exit(1) from exc
    typer.echo(json.dumps(result, indent=2))


@plugin_app.command("update")
def update() -> None:
    """Rebuild the owned plugin projections from current source."""
    installer = _installer()
    try:
        result = installer.install(installer.installed_hosts() or None)
    except CARLError as exc:
        get_console().error(f"{exc.code}: {exc}")
        raise typer.Exit(1) from exc
    typer.echo(json.dumps(result, indent=2))


@plugin_app.command("doctor")
def doctor() -> None:
    """Check installed source, runtime, and skill identities."""
    result = _installer().doctor()
    typer.echo(json.dumps(result, indent=2))
    if not result["healthy"]:
        raise typer.Exit(1)


@plugin_app.command("uninstall")
def uninstall() -> None:
    """Remove matching installer-owned entries and retain edited content."""
    typer.echo(json.dumps(_installer().uninstall(), indent=2))


@plugin_app.command("project")
def project(output: Annotated[Path, typer.Argument()] = Path("build/carl-marketplace")) -> None:
    """Render native plugin files without changing installed host profiles."""
    from carl_studio.plugin import project as render
    from carl_studio.plugin import source_root

    typer.echo(json.dumps(render(source_root(), output.resolve()), indent=2))


@plugin_app.command("migrate")
def migrate(apply: bool = typer.Option(False, "--apply")) -> None:
    """Preview the additive task metadata migration; apply only explicitly."""
    from carl_studio.mcp.tasks import MCPTaskStore

    if not apply:
        typer.echo("ALTER TABLE mcp_tasks ADD COLUMN metadata TEXT NOT NULL DEFAULT '{}';")
        return
    store = MCPTaskStore()
    try:
        typer.echo(json.dumps({"changed": store.migrate_delegation()}))
    finally:
        store.close()
