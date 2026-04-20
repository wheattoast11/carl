"""``carl db`` — local SQLite maintenance for ``~/.carl/carl.db``.

This group exposes the operator-facing DB hygiene verbs: WAL checkpoint,
retention sweep of archived sticky notes, and (opt-in) ``VACUUM``. The
heartbeat daemon runs maintenance automatically every
``CARL_MAINTENANCE_INTERVAL_CYCLES`` completed cycles (default 100), so in
normal operation an operator never needs to invoke these manually — but
they exist for incident response, one-off cleanups, and preflight before
taking a backup.

Verbs:

* ``maintenance`` — run checkpoint + retention (+ optional ``VACUUM``).
"""

from __future__ import annotations

from typing import Annotated

import typer

from carl_studio.console import get_console
from carl_studio.db import DEFAULT_RETENTION_DAYS, LocalDB

from .apps import app

db_app = typer.Typer(
    name="db",
    help="Local SQLite maintenance (~/.carl/carl.db).",
    no_args_is_help=True,
)
app.add_typer(db_app)


@db_app.command("maintenance")
def db_maintenance(
    retention_days: Annotated[
        int,
        typer.Option(
            "--retention-days",
            help="Delete archived sticky notes older than this many days (0 disables)",
        ),
    ] = DEFAULT_RETENTION_DAYS,
    vacuum: Annotated[
        bool,
        typer.Option(
            "--vacuum/--no-vacuum",
            help="Run VACUUM after the checkpoint to reclaim disk (off by default; exclusive lock)",
        ),
    ] = False,
) -> None:
    """Run a WAL checkpoint, retention sweep, and (optionally) VACUUM.

    ``--retention-days`` controls how aggressively archived sticky notes
    are pruned. Values less than ``0`` are rejected. Under heartbeat load
    the ``carl.db-wal`` file grows unbounded without periodic
    ``PRAGMA wal_checkpoint(TRUNCATE)``; this command is the manual
    equivalent of the tick the daemon performs automatically.
    """
    c = get_console()
    if retention_days < 0:
        c.error("--retention-days must be non-negative")
        raise typer.Exit(1)

    try:
        stats = LocalDB().maintenance(
            retention_days=retention_days,
            vacuum=vacuum,
        )
    except ValueError as exc:
        c.error(str(exc))
        raise typer.Exit(1) from exc
    except Exception as exc:  # noqa: BLE001 — surface any DB fault to the user
        c.error(f"maintenance failed: {exc}")
        raise typer.Exit(1) from exc

    c.header("CARL DB Maintenance")
    c.kv("notes_deleted", str(stats["notes_deleted"]), key_width=18)
    checkpoint = stats["wal_checkpoint"]
    # ``WalCheckpointResult`` is a ``TypedDict`` so the item() iteration is
    # strongly typed; the ``or`` fallback kicks in when every slot is
    # ``None`` (WAL inactive, in-memory DB, etc.) which would otherwise
    # render as an empty string.
    checkpoint_view = (
        ", ".join(f"{k}={v}" for k, v in checkpoint.items()) or "(unavailable)"
    )
    c.kv("wal_checkpoint", checkpoint_view, key_width=18)
    c.kv("vacuumed", "yes" if stats["vacuumed"] else "no", key_width=18)
    c.blank()
    c.ok("maintenance complete")
