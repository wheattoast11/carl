"""Carlito CLI — ``carl carlito``.

Manage your carlitos: list, show, spawn, retire.
Register in wiring.py via: app.add_typer(carlito_app, name='carlito')
"""

from __future__ import annotations

import json

import typer

carlito_app = typer.Typer(
    name="carlito",
    help="Manage carlitos -- small specialized agents you train and spawn.",
    no_args_is_help=True,
)


def _get_registry() -> "CarlitoRegistry":  # noqa: F821
    from carl_studio.carlito import CarlitoRegistry

    return CarlitoRegistry()


@carlito_app.command(name="list")
def carlito_list(
    status: str = typer.Option("", "--status", "-s", help="Filter by status"),
) -> None:
    """List all carlitos."""
    from carl_studio.carlito import CarlitoStatus
    from carl_studio.console import get_console

    c = get_console()
    registry = _get_registry()
    try:
        filter_status = CarlitoStatus(status) if status else None
    except ValueError:
        c.error(f"Unknown status '{status}'. Valid: {', '.join(s.value for s in CarlitoStatus)}")
        raise typer.Exit(1)

    specs = registry.list_all(status=filter_status)
    registry.close()

    if not specs:
        c.info("No carlitos yet. Spawn one with: carl carlito spawn <name>")
        return

    table = c.make_table("Name", "Domain", "Parent Model", "Status", "Skills", title="Carlitos")
    for spec in specs:
        table.add_row(
            spec.name,
            spec.domain or "(general)",
            spec.parent_model,
            spec.status.value,
            ", ".join(spec.skills) or "(none)",
        )
    c.print(table)


@carlito_app.command(name="show")
def carlito_show(
    name: str = typer.Argument(..., help="Carlito name"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Show details of a carlito."""
    from carl_studio.console import get_console

    c = get_console()
    registry = _get_registry()
    spec = registry.load(name)
    registry.close()

    if spec is None:
        c.error(f"Carlito '{name}' not found.")
        raise typer.Exit(1)

    if json_output:
        typer.echo(json.dumps(spec.model_dump(), indent=2, default=str))
        raise typer.Exit(0)

    c.header("Carlito", spec.name)
    c.kv("Parent model", spec.parent_model, key_width=20)
    c.kv("Domain", spec.domain or "(general)", key_width=20)
    c.kv("Persona", spec.persona or "(default)", key_width=20)
    c.kv("Status", spec.status.value, key_width=20)
    c.kv("Skills", ", ".join(spec.skills) or "(none)", key_width=20)
    c.kv("Curriculum ID", spec.curriculum_model_id or "(none)", key_width=20)
    c.kv("Created", spec.created_at, key_width=20)
    c.kv("Updated", spec.updated_at, key_width=20)


@carlito_app.command(name="spawn")
def carlito_spawn(
    name: str = typer.Argument(..., help="Name for the new carlito"),
    model_id: str = typer.Option("", "--model", "-m", help="Curriculum model ID"),
    domain: str = typer.Option("", "--domain", "-d", help="Domain specialization"),
    persona: str = typer.Option("", "--persona", "-p", help="Persona description"),
) -> None:
    """Spawn a carlito from a graduated curriculum track."""
    from carl_studio.carlito import CarlitoRegistry, CarlitoSpawner
    from carl_studio.console import get_console
    from carl_studio.curriculum import CurriculumStore

    c = get_console()
    registry = CarlitoRegistry()

    existing = registry.load(name)
    if existing is not None:
        c.error(f"Carlito '{name}' already exists (status: {existing.status.value}).")
        registry.close()
        raise typer.Exit(1)

    store = CurriculumStore()
    if model_id:
        track = store.load(model_id)
    else:
        track = store.current()

    if track is None:
        c.error("No curriculum track found. Enroll a model first: carl curriculum enroll")
        registry.close()
        store.close()
        raise typer.Exit(1)

    spawner = CarlitoSpawner(registry=registry)
    try:
        spec = spawner.from_graduated_track(
            name=name, track=track, domain=domain, persona=persona
        )
        card = spawner.spawn(spec, track)
    except ValueError as exc:
        c.error(str(exc))
        registry.close()
        store.close()
        raise typer.Exit(1)

    c.ok(f"Spawned carlito '{name}' from {track.model_id}")
    c.kv("Domain", domain or "(general)", key_width=16)
    c.kv("Skills", ", ".join(card.skills) or "(none)", key_width=16)
    c.kv("Status", "deployed", key_width=16)
    c.info("Show details: carl carlito show " + name)

    registry.close()
    store.close()


@carlito_app.command(name="retire")
def carlito_retire(
    name: str = typer.Argument(..., help="Carlito name to retire"),
) -> None:
    """Retire a carlito (set to dormant)."""
    from carl_studio.console import get_console

    c = get_console()
    registry = _get_registry()
    found = registry.retire(name)
    registry.close()

    if found:
        c.ok(f"Carlito '{name}' retired (dormant).")
    else:
        c.error(f"Carlito '{name}' not found.")
        raise typer.Exit(1)
