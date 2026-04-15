"""Startup and readiness commands (`carl start`, `carl doctor`)."""

from __future__ import annotations

import json

import typer

from carl_studio.console import get_console

from .apps import app
from .shared import (
    _build_start_summary,
    _camp_status_label,
    _inventory_rows,
    _project_status_label,
    _render_command_inventory,
)


# ---------------------------------------------------------------------------
# carl doctor
# ---------------------------------------------------------------------------
@app.command(name="doctor")
def doctor(
    json_output: bool = typer.Option(False, "--json", help="Output readiness as JSON"),
) -> None:
    """Audit local readiness for the guided CARL workbench workflow."""
    c = get_console()
    summary = _build_start_summary()
    readiness = summary["readiness"]
    project = summary["project"]
    dependencies = summary["dependencies"]

    doctor_payload = {
        "guided_workbench": readiness["guided_workbench"],
        "blocking_issues": readiness["blocking_issues"],
        "camp": summary["camp"],
        "project": {
            "path": project.get("path"),
            "status": _project_status_label(project),
        },
        "surfaces": {
            "training": dependencies["training"],
            "remote_jobs": readiness["remote_jobs"],
            "live_observe": readiness["live_observe"],
            "diagnose": readiness["diagnose"],
        },
    }

    if json_output:
        typer.echo(json.dumps(doctor_payload, indent=2, default=str))
        raise typer.Exit(0 if readiness["guided_workbench"] else 1)

    c.blank()
    c.header("CARL Doctor")

    table = c.make_table("Check", "Status", "Detail", title="Readiness")
    table.add_row(
        "Guided workbench",
        "ready" if readiness["guided_workbench"] else "needs setup",
        "Project, config, and training deps are aligned"
        if readiness["guided_workbench"]
        else readiness["blocking_issues"][0],
    )
    table.add_row("Project", _project_status_label(project), str(project.get("path") or "(none)"))
    table.add_row(
        "Training deps",
        "ready" if dependencies["training"] else "missing",
        "pip install 'carl-studio[training]'"
        if not dependencies["training"]
        else "local train/eval available",
    )
    table.add_row(
        "HF jobs",
        "ready" if readiness["remote_jobs"] else "optional",
        "remote jobs + push available"
        if readiness["remote_jobs"]
        else "install carl-studio[hf] and run hf auth login",
    )
    table.add_row(
        "Observe live",
        "ready" if readiness["live_observe"] else "optional",
        "textual TUI available" if readiness["live_observe"] else "install carl-studio[tui]",
    )
    table.add_row(
        "Diagnose",
        "ready" if readiness["diagnose"] else "optional",
        "Claude diagnosis available"
        if readiness["diagnose"]
        else "install carl-studio[observe] and export ANTHROPIC_API_KEY",
    )
    table.add_row("Camp", _camp_status_label(summary), "carl camp login")
    c.print(table)
    c.blank()

    if readiness["blocking_issues"]:
        c.print("  [camp.primary]Needs attention[/]")
        for idx, issue in enumerate(readiness["blocking_issues"], start=1):
            c.print(f"    {idx}. {issue}")
        c.blank()
    else:
        c.ok("Local workbench is ready.")
        c.info("Start with: carl train --config carl.yaml")
        c.blank()

    raise typer.Exit(0 if readiness["guided_workbench"] else 1)


# ---------------------------------------------------------------------------
# carl start
# ---------------------------------------------------------------------------
@app.command(name="start")
def start(
    inventory: bool = typer.Option(
        False, "--inventory", help="Show the full installed command inventory"
    ),
    json_output: bool = typer.Option(False, "--json", help="Output readiness as JSON"),
) -> None:
    """Guided onboarding: setup project, check readiness, and discover commands."""
    c = get_console()
    summary = _build_start_summary()
    readiness = summary["readiness"]

    if json_output:
        typer.echo(json.dumps(summary, indent=2, default=str))
        raise typer.Exit(0)

    project = summary["project"]
    dependencies = summary["dependencies"]
    command_tree = summary["command_inventory"]

    c.blank()
    c.header("CARL Start")

    # Interactive Project Setup Handoff
    if not inventory and not project.get("path"):
        c.info("No carl.yaml found in current directory.")
        if typer.confirm("Would you like to initialize a new project now?", default=True):
            from carl_studio.cli.project_data import project_init

            # Execute the interactive wizard
            project_init(
                name="my-carl-project",
                model="",
                method="grpo",
                dataset="",
                output_repo="",
                compute="",
                description="",
                use_case="",
                output="carl.yaml",
                interactive=True,
            )
            # Re-evaluate readiness after initialization
            summary = _build_start_summary()
            readiness = summary["readiness"]
            project = summary["project"]
            dependencies = summary["dependencies"]
            c.blank()
            c.header("Readiness Check")

    c.config_block(
        [
            (
                "guided_workbench",
                "ready" if readiness["guided_workbench"] else "needs setup",
            ),
            ("project", _project_status_label(project)),
            (
                "training_deps",
                "ready" if dependencies["training"] else "missing: carl-studio[training]",
            ),
            (
                "hf_jobs",
                "ready" if readiness["remote_jobs"] else "optional: add hf extra + auth",
            ),
            (
                "diagnose",
                "ready" if readiness["diagnose"] else "optional: add observe extra + Anthropic",
            ),
            ("camp", _camp_status_label(summary)),
        ],
        title="Workflow Readiness",
    )
    c.blank()

    c.config_block(
        [
            ("default_model", summary["defaults"]["model"]),
            ("default_compute", summary["defaults"]["compute"]),
            ("observe_source", summary["defaults"]["observe_source"]),
            ("observe_poll", f"{summary['defaults']['observe_poll']}s"),
            ("trackio_url", summary["defaults"]["trackio_url"] or "(not set)"),
        ],
        title="Defaults",
    )

    dep_table = c.make_table("Surface", "Status", title="Optional Surfaces")
    for name, ready in dependencies.items():
        dep_table.add_row(name, "ready" if ready else "missing")
    c.print(dep_table)
    c.blank()

    if readiness["blocking_issues"]:
        c.print("  [camp.primary]Needs attention[/]")
        for idx, issue in enumerate(readiness["blocking_issues"], start=1):
            c.print(f"    {idx}. {issue}")
        c.blank()

    top_level_commands = len(command_tree)
    grouped_surfaces = sum(1 for subcommands in command_tree.values() if subcommands)
    c.info(f"Installed surface: {top_level_commands} commands, {grouped_surfaces} grouped sub-apps")

    if inventory:
        _render_command_inventory(c, command_tree)
    else:
        for title, commands in _inventory_rows(command_tree):
            c.kv(title, commands, key_width=21)
        c.info("Run 'carl start --inventory' for the full installed command map.")
        c.info("Run 'carl doctor' for a detailed diagnostic audit.")
        c.info("Run 'carl camp account' for managed account, billing, and capability status.")
    c.blank()

    c.print("  [camp.primary]Next steps[/]")
    for idx, step in enumerate(summary["next_steps"], start=1):
        c.print(f"    {idx}. {step}")
    c.blank()
