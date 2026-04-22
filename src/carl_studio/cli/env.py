"""carl env — progressive-disclosure environment-setup CLI (v0.12 MVP)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import typer

from carl_studio.env_setup.questions import next_question
from carl_studio.env_setup.render import (
    render_summary_dict,
    render_training_config_yaml,
)
from carl_studio.env_setup.state import EnvState


_STATE_PATH = Path.home() / ".carl" / "last_env_state.json"
_DEFAULT_OUTPUT = Path("carl.yaml")


def _prompt_user(question: Any) -> str:
    """Render a question via ``cli/ui.py``; return the raw user answer.

    Free-form questions route to ``ui.text``; choice questions route to
    ``ui.select`` (arrow-key UX with a typer-prompt fallback when the
    ``questionary`` extra is missing or stdin is non-TTY). The first
    choice is the default per the CLI UX doctrine.
    """
    from carl_studio.cli import ui

    if question.explainer:
        typer.secho(f"  {question.explainer}", fg="bright_black")

    if question.free_form:
        return ui.text(f"  {question.prompt}").strip()

    choices = [
        ui.Choice(value=c.key, label=c.label)
        for c in question.choices
    ]
    return ui.select(f"  {question.prompt}", choices, default=0).strip()


def env_cmd(
    resume: bool = typer.Option(
        False, "--resume", help="Resume from ~/.carl/last_env_state.json"
    ),
    output: Path = typer.Option(
        _DEFAULT_OUTPUT, "--output", "-o", help="Where to write the generated carl.yaml"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Print the config instead of writing it"
    ),
    as_json: bool = typer.Option(
        False, "--json", help="Emit the state summary as JSON (non-interactive)"
    ),
    auto: bool = typer.Option(
        False, "--auto", help="Non-interactive: require --resume state to be complete"
    ),
) -> None:
    """Build a training-config YAML via a progressive-disclosure wizard."""

    import json as _json

    state = EnvState.from_json_path(_STATE_PATH) if resume else EnvState()

    # Interactive loop — unless --auto, in which case we trust resumed state.
    if not auto:
        while True:
            q = next_question(state)
            if q is None:
                break
            raw = _prompt_user(q)
            try:
                state = q.handle(state, raw)
            except ValueError as exc:
                typer.secho(f"  ⚠ {exc}", fg="yellow")
                continue
            # Persist rolling state so user can Ctrl-C and --resume later
            state.to_json_path(_STATE_PATH)

    if not state.is_complete and not as_json:
        typer.secho(
            "State is not complete; nothing to write. "
            "Re-run without --auto to complete the wizard.",
            fg="red",
        )
        raise typer.Exit(code=1)

    if as_json:
        typer.echo(_json.dumps(render_summary_dict(state), indent=2))
        return

    yaml_text = render_training_config_yaml(state)
    if dry_run:
        typer.echo(yaml_text)
        return

    output.write_text(yaml_text)
    typer.secho(f"✓ wrote {output}", fg="green")
    typer.echo("  Review then run: carl train --config carl.yaml")
