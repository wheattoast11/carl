"""``carl session [start|resume|list]`` — v0.18 Track D CLI surface.

Public surface of :mod:`carl_studio.cli_session`. Every durable operation
routes through the module-level helpers so tests + alternate front-ends
(REPL, `carl flow`) can reuse the same write path.

Per ``docs/v18_unified_entrypoint_plan.md`` §4 Track D, sync to carl.camp
is a deliberate stub until the platform exposes ``POST /api/sessions``
(see ``docs/platform-parity-reply-2026-04-22.md`` Q4).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import typer

from carl_core.errors import ValidationError


__all__ = ["session_app"]


session_app = typer.Typer(
    name="session",
    help="Manage CLI sessions: start, resume, and list per-project sessions.",
    no_args_is_help=True,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _console() -> Any:
    """Lazy import of CampConsole — keeps --help fast."""
    from carl_studio.console import get_console

    return get_console()


def _resolve_project_root() -> Path:
    """Walk up from cwd to the nearest carl project root.

    Falls back to ``Path.cwd()`` when not inside a project so that
    out-of-project invocations still produce a coherent (empty) listing
    instead of crashing — refusal-with-message is the entry router's job,
    not this helper's.
    """
    from carl_studio.project_context import current as _current

    ctx = _current(Path.cwd())
    return ctx.root if ctx is not None else Path.cwd()


def _parse_metadata(raw: str | None) -> dict[str, Any]:
    """Parse a JSON object string, or return an empty dict.

    Exits with code 2 on invalid JSON so a typo in ``--metadata`` is an
    obvious error rather than a silent empty-dict pass-through.
    """
    if not raw:
        return {}
    try:
        parsed_any: Any = json.loads(raw)
    except json.JSONDecodeError as exc:
        typer.echo(f"invalid --metadata JSON: {exc}", err=True)
        raise typer.Exit(2) from exc
    if not isinstance(parsed_any, dict):
        typer.echo("--metadata must be a JSON object", err=True)
        raise typer.Exit(2)
    return cast(dict[str, Any], parsed_any)


def _short_id(session_id: str) -> str:
    """First 8 chars of a uuid4 — matches carl.camp display convention."""
    return session_id[:8]


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@session_app.command("start")
def start_cmd(
    intent: str | None = typer.Option(
        None,
        "--intent",
        help="Human-readable intent for this session (free-form).",
    ),
    metadata: str | None = typer.Option(
        None,
        "--metadata",
        help="Arbitrary JSON object merged into the session record.",
    ),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON."),
) -> None:
    """Mint a new session and mark it as current for this project."""
    from carl_studio import cli_session

    meta = _parse_metadata(metadata)
    project_root = _resolve_project_root()

    try:
        session = cli_session.start(project_root, intent=intent, metadata=meta)
    except ValidationError as exc:
        typer.echo(f"session start failed: {exc}", err=True)
        raise typer.Exit(1) from exc

    if json_output:
        typer.echo(
            json.dumps(
                {
                    "id": session.id,
                    "short": _short_id(session.id),
                    "project_root": str(session.project_root),
                    "intent": session.intent,
                    "started_at": session.started_at.isoformat(),
                    "status": session.status,
                }
            )
        )
        return

    c = _console()
    c.header("Session started", _short_id(session.id))
    c.kv("ID", session.id, key_width=16)
    c.kv("Project", str(session.project_root), key_width=16)
    if intent:
        c.kv("Intent", intent, key_width=16)
    c.kv("Status", session.status, key_width=16)
    c.info("Use `carl session resume " + _short_id(session.id) + "...` to reattach later.")


@session_app.command("resume")
def resume_cmd(
    session_id: str = typer.Argument(..., help="Session id (full uuid4)."),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON."),
) -> None:
    """Resume an existing session — marks it as current for this project.

    If the session is already completed, a warning is printed but the
    pointer is still updated: inspecting a completed run is a supported
    flow.
    """
    from carl_studio import cli_session

    project_root = _resolve_project_root()

    try:
        session = cli_session.load(session_id, project_root)
    except ValidationError as exc:
        typer.echo(f"session resume failed: {exc}", err=True)
        raise typer.Exit(1) from exc

    try:
        cli_session.set_current(project_root, session.id)
    except ValidationError as exc:
        typer.echo(f"session resume failed: {exc}", err=True)
        raise typer.Exit(1) from exc

    if json_output:
        typer.echo(
            json.dumps(
                {
                    "id": session.id,
                    "short": _short_id(session.id),
                    "status": session.status,
                    "intent": session.intent,
                    "started_at": session.started_at.isoformat(),
                    "resumed": True,
                }
            )
        )
        return

    c = _console()
    c.header("Session resumed", _short_id(session.id))
    c.kv("ID", session.id, key_width=16)
    c.kv("Status", session.status, key_width=16)
    if session.intent:
        c.kv("Intent", session.intent, key_width=16)
    if session.status != "active":
        c.warn(
            f"session is {session.status}; marking current but not reactivating"
        )


@session_app.command("list")
def list_cmd(
    all_sessions: bool = typer.Option(
        False,
        "--all",
        help="Include sessions across every project (v0.18.1+; noop today).",
    ),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON."),
) -> None:
    """List sessions under the current project."""
    from carl_studio import cli_session

    project_root = _resolve_project_root()
    sessions = cli_session.list_sessions(project_root)

    if all_sessions:
        # Global index is a v0.18.1 concern; surface the limitation
        # rather than silently lying about scope.
        typer.echo(
            "# --all is a v0.18.1 surface; listing current-project sessions only.",
        )

    current_session = cli_session.current(project_root)
    current_id = current_session.id if current_session else None

    if json_output:
        payload = [
            {
                "id": s.id,
                "short": _short_id(s.id),
                "intent": s.intent,
                "status": s.status,
                "started_at": s.started_at.isoformat(),
                "completed_at": s.completed_at.isoformat() if s.completed_at else None,
                "is_current": s.id == current_id,
            }
            for s in sessions
        ]
        typer.echo(json.dumps(payload, indent=2))
        return

    c = _console()
    if not sessions:
        c.info("No sessions yet. Start one with `carl session start`.")
        return

    table = c.make_table(
        "",
        "ID",
        "Status",
        "Started",
        "Intent",
        title=f"Sessions in {project_root}",
    )
    for s in sessions:
        marker = "*" if s.id == current_id else ""
        intent_display = (s.intent or "") if s.intent is None or len(s.intent) <= 40 else s.intent[:37] + "..."
        table.add_row(
            marker,
            _short_id(s.id),
            s.status,
            s.started_at.strftime("%Y-%m-%d %H:%M"),
            intent_display,
        )
    c.print(table)
    if current_id:
        c.info(f"Current session: {_short_id(current_id)} (marked with *).")
