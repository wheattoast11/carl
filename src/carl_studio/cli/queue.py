"""``carl queue`` — manage the sticky-note work queue.

User-facing CLI for the ``StickyQueue`` persistence layer (see
``carl_studio.sticky``). These commands let a user inspect and curate the
async work inbox that the CARL heartbeat loop consumes.

Four verbs — deliberately small surface area:

* ``add``    — append a new note with priority.
* ``list``   — render queued / processing / done / archived notes.
* ``status`` — bucket counts by status.
* ``clear``  — archive completed notes (default) or all non-archived notes.
"""

from __future__ import annotations

from typing import Annotated

import typer

from carl_studio.console import get_console
from carl_studio.db import LocalDB
from carl_studio.sticky import StickyQueue, StickyStatus


queue_app = typer.Typer(
    name="queue",
    help="Manage the sticky-note queue (async work inbox for Carl's heartbeat loop).",
    no_args_is_help=True,
)


_VALID_STATUSES: tuple[StickyStatus, ...] = ("queued", "processing", "done", "archived")


def _coerce_status(raw: str | None) -> StickyStatus | None:
    """Validate an optional status filter and return a narrowed literal."""
    if raw is None:
        return None
    if raw not in _VALID_STATUSES:
        raise typer.BadParameter(
            "status must be one of: queued, processing, done, archived"
        )
    # Runtime-narrowed to the Literal; cast is explicit for pyright.
    return raw  # type: ignore[return-value]


@queue_app.command("add")
def queue_add(
    text: Annotated[str, typer.Argument(help="Sticky-note content")],
    priority: Annotated[
        int,
        typer.Option("--priority", "-p", help="1-10, higher runs sooner"),
    ] = 5,
) -> None:
    """Append a new sticky note to the queue."""
    c = get_console()
    content = text.strip()
    if not content:
        c.error("note content must not be empty")
        raise typer.Exit(1)
    if priority < 1 or priority > 10:
        c.error("priority must be in 1..10")
        raise typer.Exit(1)

    try:
        note = StickyQueue(LocalDB()).append(content, priority=priority)
    except ValueError as exc:
        c.error(str(exc))
        raise typer.Exit(1) from exc

    c.ok(f"queued {note.id} (priority={note.priority})")


@queue_app.command("list")
def queue_list(
    limit: Annotated[
        int,
        typer.Option("--limit", "-n", help="Max notes to render"),
    ] = 20,
    status: Annotated[
        str | None,
        typer.Option(
            "--status",
            "-s",
            help="Filter: queued | processing | done | archived",
        ),
    ] = None,
) -> None:
    """Show notes in the queue."""
    c = get_console()
    if limit <= 0:
        c.error("--limit must be a positive integer")
        raise typer.Exit(1)

    status_filter = _coerce_status(status)
    notes = StickyQueue(LocalDB()).status(limit=limit, status=status_filter)
    if not notes:
        c.info("queue empty")
        raise typer.Exit(0)

    table = c.make_table("id", "pri", "status", "created", "content", title="Queue")
    for n in notes:
        # Keep content previews short so the table stays readable in narrow terms.
        preview = n.content if len(n.content) <= 80 else n.content[:77] + "..."
        table.add_row(n.id, str(n.priority), n.status, n.created_at, preview)
    c.print(table)


@queue_app.command("status")
def queue_status() -> None:
    """Show bucket counts by status."""
    c = get_console()
    notes = StickyQueue(LocalDB()).status(limit=500)
    buckets: dict[str, int] = {s: 0 for s in _VALID_STATUSES}
    for n in notes:
        buckets[n.status] = buckets.get(n.status, 0) + 1

    c.header("Queue status")
    for key in _VALID_STATUSES:
        c.kv(key, str(buckets[key]))

    pending = buckets["queued"] + buckets["processing"]
    if pending == 0:
        c.info("No pending work.")
    else:
        c.ok(f"{pending} pending note(s) awaiting Carl.")


@queue_app.command("clear")
def queue_clear(
    done_only: Annotated[
        bool,
        typer.Option(
            "--done/--all",
            help="Archive only 'done' notes (default) vs. all non-archived notes",
        ),
    ] = True,
) -> None:
    """Archive completed notes (safe) or every non-archived note (``--all``)."""
    c = get_console()
    q = StickyQueue(LocalDB())

    if done_only:
        targets = q.status(limit=10_000, status="done")
    else:
        targets = [n for n in q.status(limit=10_000) if n.status != "archived"]

    for n in targets:
        q.archive(n.id)

    label = "done" if done_only else "non-archived"
    c.ok(f"archived {len(targets)} {label} note(s)")
