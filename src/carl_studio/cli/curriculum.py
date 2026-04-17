"""Curriculum CLI -- ``carl curriculum``.

Track your carlito's academic progress through the CARL training curriculum.
Register in cli.py via: ``app.add_typer(curriculum_app, name='curriculum')``
"""
from __future__ import annotations

import typer

from carl_studio.console import get_console
from carl_studio.curriculum import (
    CurriculumPhase,
    CurriculumStore,
    CurriculumTrack,
)

curriculum_app = typer.Typer(
    name="curriculum",
    help="Track your carlito's academic progress",
    no_args_is_help=True,
)


def _get_store() -> CurriculumStore:
    """Get or create the curriculum store."""
    return CurriculumStore()


@curriculum_app.command(name="show")
def curriculum_show(
    model_id: str = typer.Argument("", help="Model ID (blank for most recent)"),
) -> None:
    """Show curriculum status for a carlito."""
    c = get_console()
    store = _get_store()

    try:
        if model_id:
            track = store.load(model_id)
        else:
            track = store.current()

        if track is None:
            c.info("No carlitos enrolled yet. Start with: carl lab curriculum enroll <model_id>")
            return

        c.header("Curriculum Status")
        c.kv("Model", track.model_id)
        c.kv("Phase", track.phase.value)
        c.kv("Version", str(track.version))
        c.kv("Milestones", str(len(track.milestones)))
        c.blank()

        if track.milestones:
            table = c.make_table("Phase", "Event", "Detail", "Timestamp", title="Milestones")
            for ms in track.milestones[-10:]:  # Show last 10
                table.add_row(ms.phase.value, ms.event, ms.detail, ms.timestamp[:19])
            c.print(table)
    finally:
        store.close()


@curriculum_app.command(name="list")
def curriculum_list() -> None:
    """List all enrolled carlitos."""
    c = get_console()
    store = _get_store()

    try:
        tracks = store.list_tracks()

        if not tracks:
            c.info("No carlitos enrolled yet. Start with: carl lab curriculum enroll <model_id>")
            return

        c.header("Enrolled Carlitos")
        table = c.make_table("Model ID", "Phase", "Version", "Milestones", "Last Event")
        for track in tracks:
            s = track.summary()
            table.add_row(
                s["model_id"],
                s["phase"],
                str(s["version"]),
                str(s["milestones"]),
                s["last_event"],
            )
        c.print(table)
    finally:
        store.close()


@curriculum_app.command(name="advance")
def curriculum_advance(
    model_id: str = typer.Argument(..., help="Model ID to advance"),
    phase: str = typer.Argument(..., help="Target phase (drilling, evaluated, graduated, deployed, ttt_active)"),
    event: str = typer.Option("", "--event", "-e", help="Event description"),
    detail: str = typer.Option("", "--detail", "-d", help="Additional detail"),
) -> None:
    """Advance a carlito to the next curriculum phase."""
    c = get_console()
    store = _get_store()

    try:
        # Validate phase
        try:
            target = CurriculumPhase(phase.lower())
        except ValueError:
            valid = ", ".join(p.value for p in CurriculumPhase)
            c.error(f"Invalid phase: {phase!r}. Must be one of: {valid}")
            raise typer.Exit(code=1)

        track = store.load(model_id)
        if track is None:
            c.error(
                f"No carlito found with ID: {model_id!r}. "
                "Enroll first with: carl lab curriculum enroll <model_id>"
            )
            raise typer.Exit(code=1)

        try:
            updated = track.advance(to=target, event=event, detail=detail)
        except ValueError as exc:
            c.error(str(exc))
            raise typer.Exit(code=1)

        store.save(updated)
        c.ok(f"{model_id}: {track.phase.value} -> {target.value}")
        if event:
            c.info(f"Event: {event}")
    finally:
        store.close()


@curriculum_app.command(name="enroll")
def curriculum_enroll(
    model_id: str = typer.Argument(..., help="Model ID to enroll"),
) -> None:
    """Enroll a new carlito in the curriculum."""
    c = get_console()
    store = _get_store()

    try:
        existing = store.load(model_id)
        if existing is not None:
            c.warn(f"Carlito {model_id!r} already enrolled (phase: {existing.phase.value})")
            return

        track = CurriculumTrack(model_id=model_id)
        store.save(track)
        c.ok(f"Enrolled: {model_id}")
        c.kv("Phase", track.phase.value)
    finally:
        store.close()
