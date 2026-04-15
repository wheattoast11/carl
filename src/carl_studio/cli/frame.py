"""Frame CLI — ``carl frame``.

Set, show, and clear the analytical WorkFrame that shapes how CARL
structures any problem.
"""

from __future__ import annotations

import json

import typer

frame_app = typer.Typer(
    name="frame",
    help="Set the analytical lens (domain/function/role/objectives) for data ingestion.",
    no_args_is_help=True,
)


@frame_app.command(name="set")
def frame_set(
    domain: str = typer.Option("", "--domain", "-d", help="Subject matter (e.g. saas_sales, pharma)"),
    function: str = typer.Option("", "--function", "-f", help="Analytical lens (e.g. territory_planning, quota_setting)"),
    role: str = typer.Option("", "--role", "-r", help="Decision scope (e.g. analyst, manager, exec)"),
    goal: list[str] = typer.Option([], "--goal", "-g", help="Measurable objective (repeatable)"),
    entity: list[str] = typer.Option([], "--entity", "-e", help="Key entity name (repeatable)"),
    metric: list[str] = typer.Option([], "--metric", "-m", help="KPI to track (repeatable)"),
    constraint: list[str] = typer.Option([], "--constraint", help="Hard limit (repeatable)"),
    context: str = typer.Option("", "--context", help="Free-text background"),
) -> None:
    """Set the active WorkFrame for data ingestion and analysis."""
    from carl_studio.console import get_console
    from carl_studio.frame import WorkFrame

    c = get_console()

    # Merge with existing frame if partial update
    frame = WorkFrame.load()
    updates = {}
    if domain:
        updates["domain"] = domain
    if function:
        updates["function"] = function
    if role:
        updates["role"] = role
    if goal:
        updates["objectives"] = goal
    if entity:
        updates["entities"] = entity
    if metric:
        updates["metrics"] = metric
    if constraint:
        updates["constraints"] = constraint
    if context:
        updates["context"] = context

    if not updates:
        c.warn("No frame dimensions specified. Use --domain, --function, --role, --goal, etc.")
        raise typer.Exit(1)

    frame = frame.model_copy(update=updates)
    path = frame.save()
    c.ok(f"Frame saved to {path}")
    c.kv("Domain", frame.domain or "(not set)", key_width=14)
    c.kv("Function", frame.function or "(not set)", key_width=14)
    c.kv("Role", frame.role or "(not set)", key_width=14)
    if frame.objectives:
        c.kv("Objectives", "; ".join(frame.objectives), key_width=14)
    if frame.entities:
        c.kv("Entities", ", ".join(frame.entities), key_width=14)


@frame_app.command(name="show")
def frame_show(
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Show the active WorkFrame."""
    from carl_studio.console import get_console
    from carl_studio.frame import WorkFrame

    c = get_console()
    frame = WorkFrame.from_project()

    if json_output:
        typer.echo(json.dumps(frame.model_dump(), indent=2))
        raise typer.Exit(0)

    if not frame.active:
        c.info("No frame set. Use: carl frame set --domain <domain> --function <function>")
        return

    c.header("WorkFrame")
    c.kv("Domain", frame.domain or "(not set)", key_width=14)
    c.kv("Function", frame.function or "(not set)", key_width=14)
    c.kv("Role", frame.role or "(not set)", key_width=14)
    if frame.objectives:
        c.kv("Objectives", "; ".join(frame.objectives), key_width=14)
    if frame.constraints:
        c.kv("Constraints", "; ".join(frame.constraints), key_width=14)
    if frame.entities:
        c.kv("Entities", ", ".join(frame.entities), key_width=14)
    if frame.metrics:
        c.kv("Metrics", ", ".join(frame.metrics), key_width=14)
    if frame.context:
        c.blank()
        c.print(f"  {frame.context}")

    c.blank()
    lanes = frame.decompose()
    c.info(f"MECE lanes: {', '.join(lanes)}")


@frame_app.command(name="clear")
def frame_clear() -> None:
    """Clear the active WorkFrame."""
    from carl_studio.console import get_console
    from carl_studio.frame import WorkFrame

    c = get_console()
    WorkFrame().clear()
    c.ok("Frame cleared.")
