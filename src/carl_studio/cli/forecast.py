"""``carl forecast`` CLI — anticipatory coherence trinity (v0.19 FREE tier).

Reads recent phi values (from --history flag, file, or session), runs
the analytic forecaster, prints the trinity view + early-warning summary.

Pure-numpy; no torch needed. The PAID tier ([anticipatory] extra) adds
fractal multi-band + learned subspace prior.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

forecast_app = typer.Typer(
    help="Anticipatory coherence forecast (v0.19 trinity).",
    no_args_is_help=True,
)


@forecast_app.callback()
def _forecast_root() -> None:  # pyright: ignore[reportUnusedFunction]
    """Force Typer to treat this as a sub-command group."""


def _parse_history(
    history_arg: str | None, history_file: Path | None
) -> list[float]:
    """Parse comma-separated history string OR JSON file containing a list of floats."""
    if history_file is not None:
        try:
            data = json.loads(history_file.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise typer.BadParameter(f"failed to read history file: {exc}") from exc
        if not isinstance(data, list):
            raise typer.BadParameter("history file must contain a JSON list of floats")
        return [float(x) for x in data]  # pyright: ignore[reportUnknownVariableType,reportUnknownArgumentType]
    if history_arg is not None and history_arg.strip():
        try:
            return [float(x.strip()) for x in history_arg.split(",")]
        except ValueError as exc:
            raise typer.BadParameter(f"invalid history value: {exc}") from exc
    return []


@forecast_app.command("show")
def show_cmd(
    horizon: int = typer.Option(4, "--horizon", "-H", help="Steps to forecast."),
    threshold: float = typer.Option(0.5, "--threshold", "-t", help="Early-warning phi threshold."),
    history: Optional[str] = typer.Option(  # noqa: UP007 (typer needs Optional)
        None, "--history", help="Comma-separated phi history values."
    ),
    history_file: Optional[Path] = typer.Option(  # noqa: UP007
        None, "--history-file", help="JSON file containing list of phi values."
    ),
    method: str = typer.Option(
        "linear", "--method", "-m", help="Forecaster: linear or lyapunov."
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Emit JSON instead of human-readable output."
    ),
) -> None:
    """Show anticipatory trinity (past + present + forecast) for given history."""
    from carl_studio.observe.forecast_dashboard import make_trinity_view

    phi_history = _parse_history(history, history_file)
    present_phi = phi_history[-1] if phi_history else 0.5

    trinity = make_trinity_view(
        phi_history=phi_history,
        present_phi=present_phi,
        horizon=horizon,
        method=method,
    )
    warning_step = trinity.early_warning_step(threshold=threshold)

    if json_output:
        payload = trinity.to_dict()
        payload["early_warning_step"] = warning_step
        payload["threshold"] = threshold
        typer.echo(json.dumps(payload, indent=2))
        return

    typer.echo("=" * 60)
    typer.echo("CARL anticipatory coherence trinity")
    typer.echo("=" * 60)
    past = trinity.past_summary
    if past["n"] == 0:
        typer.echo("PAST     | (no history provided)")
    else:
        typer.echo(
            f"PAST     | n={past['n']:3d}  min={past['min']:.3f}  "
            f"mean={past['mean']:.3f}  max={past['max']:.3f}"
        )
    typer.echo(f"PRESENT  | phi={trinity.present_phi:.3f}")
    fc = trinity.forecast
    typer.echo(
        f"FUTURE   | horizon={fc.horizon_steps}  method={fc.method}  "
        f"first_phi={fc.phi_at(0):.3f}  last_phi={fc.phi_at(fc.horizon_steps - 1):.3f}"
    )
    typer.echo(f"DRIFT    | lyapunov={fc.lyapunov_drift():+.4f}")
    if warning_step is None:
        typer.echo(f"WARNING  | none above threshold {threshold:.2f}")
    else:
        typer.echo(
            f"WARNING  | EARLY-WARNING at step {warning_step} (threshold {threshold:.2f})"
        )
    typer.echo("=" * 60)


__all__ = ["forecast_app"]
