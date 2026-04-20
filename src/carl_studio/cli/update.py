"""carl update — self-updating agentic meta-pipeline CLI (v0.11).

Reports recent git commits, dependency deltas vs PyPI, and a
positive-framed blast-radius summary. Never auto-applies anything —
the user decides after seeing the report.

Respects consent: PyPI calls gated by ``consent_gate("telemetry")``.
"""

from __future__ import annotations

import json as _json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import typer

from carl_studio import __version__ as _carl_version
from carl_studio.update import build_update_report
from carl_studio.update.dep_scan import (
    installed_versions_from_metadata,
    scan_dep_deltas,
)
from carl_studio.update.git_scan import current_head_sha, scan_git_commits


# CARL-adjacent packages we actually track — bounded per Fano V1
# boundedness. Tweak as the dep set evolves.
_DEFAULT_PACKAGES_OF_INTEREST = [
    "carl-studio",
    "carl-core",
    "pydantic",
    "pydantic-settings",
    "typer",
    "rich",
    "anthropic",
    "huggingface-hub",
    "transformers",
    "trl",
    "peft",
]


def _render_report_rich(report: Any, console: Any, detailed: bool) -> None:
    """Pretty-print via CampConsole; degrades to plain typer.echo."""

    try:
        from carl_studio.cli.console import CampConsole  # type: ignore
    except Exception:
        CampConsole = None  # type: ignore

    if console is None and CampConsole is not None:
        console = CampConsole()

    if console is None:
        # Plain fallback
        typer.echo(f"CARL Update · {report.carl_version} · HEAD {report.git_head or '?'}")
        typer.echo(f"Recent commits: {len(report.git_commits)}")
        for c in report.git_commits[:10]:
            typer.echo(f"  {c.sha[:7]} {c.date}  {c.subject}")
        typer.echo(f"Dependency deltas: {len(report.dep_deltas)}")
        for d in report.dep_deltas:
            typer.echo(f"  {d.package:<20} {d.current} → {d.latest}  [{d.severity}]")
        if report.errors:
            typer.echo(f"Errors ({len(report.errors)}):")
            for e in report.errors[:10]:
                typer.echo(f"  • {e}")
        if not report.any_deltas:
            typer.echo("No deltas since last check — up to date.")
        return

    # Rich path (best-effort; falls back on any attribute error)
    try:
        console.section("CARL Update")
        console.print(
            f"[bold cyan]{report.carl_version}[/bold cyan]  "
            f"HEAD [dim]{report.git_head or '?'}[/dim]"
        )

        if report.git_commits:
            console.section("Recent Commits")
            for c in report.git_commits[: 10 if detailed else 5]:
                console.print(f"  [dim]{c.sha[:7]}[/dim] {c.date}  {c.subject}")

        if report.dep_deltas:
            console.section("Dependency Deltas")
            for d in report.dep_deltas:
                console.print(
                    f"  {d.package:<20} {d.current} → {d.latest}  "
                    f"[dim][{d.severity}][/dim]"
                )

        if report.blast_radius:
            console.section("What This Unlocks")
            for b in report.blast_radius[: 10 if detailed else 5]:
                arrow = "→" if b.direction == "positive" else "⚠"
                console.print(f"  {arrow} {b.impact}")

        if report.errors:
            console.section("Errors (non-fatal)")
            for e in report.errors[:5]:
                console.print(f"  [yellow]•[/yellow] {e}")

        if not report.any_deltas:
            console.print("[green]✓[/green] No deltas since last check — up to date.")
    except Exception:
        # If anything in the pretty path fails, emit the plain form.
        typer.echo(report.model_dump_json(indent=2))


def update_cmd(
    detailed: bool = typer.Option(
        False, "--detailed", help="Show full commit/dep lists instead of top-5."
    ),
    summary_only: bool = typer.Option(
        False, "--summary-only", help="Print only the 1-line summary."
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Skip network calls (PyPI); only git + local checks.",
    ),
    as_json: bool = typer.Option(
        False, "--json", help="Emit machine-readable JSON instead of rendered output."
    ),
) -> None:
    """Run the carl-update meta-pipeline and print the report."""

    # Consent gate for the one network call path.
    try:
        from carl_studio.consent import consent_gate

        if not dry_run:
            try:
                consent_gate("telemetry")
            except Exception:
                # Consent denied → continue in offline mode.
                dry_run = True
    except Exception:
        pass  # carl-studio consent module missing? stay permissive.

    repo_path = Path.cwd()
    commits, git_errors = scan_git_commits(repo_path=repo_path)
    head = current_head_sha(repo_path=repo_path)

    dep_deltas: list[Any] = []
    dep_errors: list[str] = []
    if not dry_run:
        installed = installed_versions_from_metadata(_DEFAULT_PACKAGES_OF_INTEREST)
        dep_deltas, dep_errors = scan_dep_deltas(
            installed=installed,
            packages_of_interest=_DEFAULT_PACKAGES_OF_INTEREST,
        )

    report = build_update_report(
        carl_version=_carl_version,
        git_head=head,
        last_check_at=None,  # TODO v0.11.1: read from LocalDB config
        git_commits=commits,
        dep_deltas=dep_deltas,
        errors=git_errors + dep_errors,
    )

    if as_json:
        typer.echo(report.model_dump_json(indent=2))
        return

    if summary_only:
        summary = (
            f"carl {report.carl_version} · {len(report.git_commits)} commits · "
            f"{len(report.dep_deltas)} dep deltas · {len(report.errors)} non-fatal errors"
        )
        typer.echo(summary)
        return

    _render_report_rich(report, console=None, detailed=detailed)
