"""Startup and readiness commands (`carl start`, `carl doctor`)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

import typer

from carl_studio.console import get_console
from carl_studio.sticky import DEFAULT_RECLAIM_MAX_AGE_S

from .apps import app
from .shared import (
    _build_start_summary,
    _camp_status_label,
    _inventory_rows,
    _project_status_label,
    _render_command_inventory,
)

# JRN-005 — Next-steps block is suppressed once the first-run marker is
# older than this (in seconds). 7 days aligns with the rest of the post-
# onboarding grace period surfaces.
_DOCTOR_NEXT_STEPS_MAX_AGE_S: float = 7 * 24 * 60 * 60


def _next_steps_should_show(
    *, verbose: bool, marker_path: Path, now_fn: Callable[[], float] | None = None,
) -> bool:
    """Decide whether to render the Next steps block.

    Shown when ``verbose`` is on OR the first-run marker is younger than
    7 days. Missing marker is treated as "never initialized" — we do not
    show the block in that case; the caller is likely to hit the bigger
    "run carl init" nudge first, which is a better signal.
    """
    if verbose:
        return True
    try:
        if not marker_path.is_file():
            return False
        mtime = marker_path.stat().st_mtime
    except OSError:
        return False
    current = (now_fn or time.time)()
    age = current - mtime
    return 0 <= age <= _DOCTOR_NEXT_STEPS_MAX_AGE_S


# ---------------------------------------------------------------------------
# carl doctor
# ---------------------------------------------------------------------------
@app.command(name="doctor")
def doctor(
    json_output: bool = typer.Option(False, "--json", help="Output readiness as JSON"),
    check_freshness: bool = typer.Option(
        False, "--check-freshness", help="Force dependency freshness check"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Always show the Next steps guidance block"
    ),
) -> None:
    """Audit local readiness for the guided CARL workbench workflow."""
    c = get_console()
    summary = _build_start_summary()
    readiness = summary["readiness"]
    project = summary["project"]
    dependencies = summary["dependencies"]

    # Sticky-note queue — never let the queue subsystem break doctor.
    queue_pending: int | None
    queue_error: str | None = None
    queue_oldest_processing_s: float | None = None
    try:
        from carl_studio.db import LocalDB
        from carl_studio.sticky import StickyQueue

        _queue = StickyQueue(LocalDB())
        _queue_notes = _queue.status(limit=500)
        queue_pending = sum(
            1 for n in _queue_notes if n.status in ("queued", "processing")
        )
        # Surface the oldest ``processing`` row's age so operators can spot
        # a wedge at a glance. Non-None only when at least one row is
        # actually stuck — keeps the happy path noise-free.
        queue_oldest_processing_s = _queue.oldest_processing_age_seconds()
    except Exception as exc:  # noqa: BLE001 — doctor must never crash on subsystem errors
        queue_pending = None
        queue_error = str(exc)

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
        "queue": {
            "pending": queue_pending,
            "error": queue_error,
            "oldest_processing_s": queue_oldest_processing_s,
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
    if queue_error is not None:
        table.add_row("Queue", "error", queue_error)
    elif queue_pending is None:
        table.add_row("Queue", "unavailable", "sticky queue unavailable")
    else:
        # A non-None ``oldest_processing_s`` is the signal that at least
        # one row is in ``processing``. If its age exceeds the shared
        # :data:`DEFAULT_RECLAIM_MAX_AGE_S` threshold we surface a warning-
        # shaped status so operators see something is stuck without having
        # to grep logs. Keeping this bound to the *same* constant the
        # daemon's in-loop reclaim uses eliminates the drift that used to
        # make ``carl doctor`` flag "stuck" for work the daemon had
        # already reclaimed (R2-005).
        queue_status = "empty" if queue_pending == 0 else "ready"
        queue_detail = (
            "no sticky notes pending"
            if queue_pending == 0
            else f"{queue_pending} pending"
        )
        if (
            queue_oldest_processing_s is not None
            and queue_oldest_processing_s > float(DEFAULT_RECLAIM_MAX_AGE_S)
        ):
            queue_status = "stuck"
            queue_detail += (
                f" (oldest processing {queue_oldest_processing_s:.0f}s — "
                "run: carl queue reclaim)"
            )
        elif queue_oldest_processing_s is not None:
            queue_detail += (
                f" (oldest processing {queue_oldest_processing_s:.0f}s)"
            )
        table.add_row("Queue", queue_status, queue_detail)
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

    # Freshness check (silent unless issues found)
    from carl_studio.freshness import needs_check, run_freshness_check

    if check_freshness or needs_check():
        freshness = run_freshness_check(force=check_freshness)
        if freshness.has_issues:
            tone = "red" if freshness.has_errors else "yellow"
            c.print(f"[{tone}]Freshness: {freshness.summary}[/{tone}]")
            for issue in freshness.issues:
                c.print(
                    f"  [dim]- [{issue.severity}] {issue.detail}"
                    f" (fix: {issue.remediation})[/dim]"
                )
            c.blank()

    # JRN-005 — humanize doctor with a short Next steps block. Hidden when
    # there are blocking issues (don't distract from red messages) and
    # gated to recently initialized installs unless --verbose is on.
    if not readiness["blocking_issues"]:
        try:
            from carl_studio.cli.init import FIRST_RUN_MARKER

            marker = FIRST_RUN_MARKER
        except Exception:  # pragma: no cover — defensive
            marker = Path.home() / ".carl" / ".initialized"
        if _next_steps_should_show(verbose=verbose, marker_path=marker):
            c.print("  [camp.primary]Next steps[/]")
            c.kv("carl \"explore my repo\"", "start a conversation")
            c.kv("carl train --help", "train a model")
            c.kv("carl queue add \"text\"", "add a sticky-note")
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

    # First-run nudge: if the user hasn't run `carl init` yet, flag it early
    # so they get one-shot setup instead of piecemeal prompts.
    try:
        from carl_studio.cli.init import _first_run_complete

        if not _first_run_complete() and not inventory:
            c.print("  [camp.accent]New here?[/] Run: [bold]carl init[/] for one-shot setup.")
            c.blank()
    except Exception:
        pass

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

    # Freshness check (silent unless issues found)
    from carl_studio.freshness import needs_check, run_freshness_check

    if needs_check():
        freshness = run_freshness_check()
        if freshness.has_issues:
            tone = "red" if freshness.has_errors else "yellow"
            c.print(f"[{tone}]Freshness: {freshness.summary}[/{tone}]")
            for issue in freshness.issues:
                c.print(
                    f"  [dim]- [{issue.severity}] {issue.detail}"
                    f" (fix: {issue.remediation})[/dim]"
                )
            c.blank()

    c.print("  [camp.primary]Next steps[/]")
    for idx, step in enumerate(summary["next_steps"], start=1):
        c.print(f"    {idx}. {step}")
    c.blank()
