"""CLI projections of the Session-owned semantic interface."""

from __future__ import annotations

from pathlib import Path

import typer

from carl_studio.console import get_console

from .apps import lab_app

semantic_app = typer.Typer(help="Local encoder and explicit interpretation feedback.")
lab_app.add_typer(semantic_app, name="semantic")


@semantic_app.command("invoke")
def semantic_invoke(operation: str, request: Path, session_id: str = "semantic") -> None:
    """Invoke the shared Session implementation with durable restart state."""
    import json

    from carl_core.errors import CARLError

    from carl_studio.session import Session, TwinCheckpoint
    from carl_studio.sessions import SessionStore

    if operation not in {"encode_data", "interpret", "interpretation_feedback"}:
        get_console().error("Unknown semantic operation")
        raise typer.Exit(1)
    store = SessionStore()
    state = store.load(session_id)
    session = (
        Session.restore(TwinCheckpoint.model_validate(state["semantic_checkpoint"]))
        if state and "semantic_checkpoint" in state
        else Session(workspace=str(Path.cwd()))
    )
    try:
        result = getattr(session.semantic, operation)(**json.loads(request.read_text()))
        store.save(session_id, {"semantic_checkpoint": session.snapshot().model_dump(mode="json")})
        get_console().info(json.dumps(result))
    except (ValueError, OSError, RuntimeError, CARLError) as exc:
        get_console().error(type(exc).__name__ + ": semantic operation failed")
        raise typer.Exit(1) from exc
    finally:
        session.teardown()


@semantic_app.command("capture")
def semantic_capture(workspace: Path = Path("."), enabled: bool = True) -> None:
    """Grant or revoke local workspace capture independently of training."""
    from carl_studio.consent import ConsentManager

    ConsentManager().capture_grant(workspace=str(workspace.resolve()), enabled=enabled)
    get_console().info("Local workspace capture " + ("enabled" if enabled else "disabled"))
