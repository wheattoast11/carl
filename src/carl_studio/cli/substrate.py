"""``carl substrate`` CLI — multi-channel substrate health (v0.19 FREE tier).

Samples SubstrateState via SubstrateMonitor, prints per-channel health +
drift derivatives + leading-indicator status. Lazy-imports psutil; if
absent, prints a single ``runtime`` channel and a hint to install it.
"""
from __future__ import annotations

import json
from typing import Optional

import typer

substrate_app = typer.Typer(
    help="Substrate health channels (v0.19 anticipatory coherence).",
    no_args_is_help=True,
)


@substrate_app.callback()
def _substrate_root() -> None:  # pyright: ignore[reportUnusedFunction]
    """Force Typer to treat this as a sub-command group."""


@substrate_app.command("show")
def show_cmd(
    samples: int = typer.Option(
        1, "--samples", "-n", help="Number of consecutive samples to take."
    ),
    interval: float = typer.Option(
        0.0, "--interval", "-i", help="Seconds between samples (default: 0)."
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Emit JSON instead of human-readable output."
    ),
) -> None:
    """Sample current substrate state and print per-channel health."""
    import time

    from carl_studio.observe.substrate_monitor import SubstrateMonitor

    monitor = SubstrateMonitor()
    states: list[Optional[dict[str, object]]] = []
    last_state = None
    for i in range(max(1, samples)):
        if i > 0 and interval > 0:
            time.sleep(interval)
        state = monitor.sample()
        last_state = state
        states.append(state.to_dict())

    if last_state is None:
        typer.echo("ERROR: no substrate sample taken", err=True)
        raise typer.Exit(code=2)

    if json_output:
        typer.echo(
            json.dumps(
                {
                    "psutil_available": monitor.psutil_available,
                    "samples": states,
                },
                indent=2,
            )
        )
        return

    typer.echo("=" * 60)
    typer.echo(
        f"CARL substrate state  (psutil={'yes' if monitor.psutil_available else 'no'})"
    )
    typer.echo("=" * 60)
    if not monitor.psutil_available:
        typer.echo(
            "HINT: install psutil for per-channel monitoring (memory/compute/fd):"
        )
        typer.echo("  pip install psutil")
    typer.echo(f"timestamp     | {last_state.timestamp:.3f}")
    typer.echo(f"overall_psi   | {last_state.overall_psi:.4f}")
    typer.echo("-" * 60)
    typer.echo(f"{'channel':<10}  {'health':>8}  {'drift':>10}  {'accel':>10}  warn")
    typer.echo("-" * 60)
    for name, ch in last_state.channels.items():
        warn = "LEADING" if ch.leading_indicator else "ok"
        typer.echo(
            f"{name:<10}  {ch.health:>8.4f}  {ch.drift_rate:>+10.4f}  "
            f"{ch.drift_acceleration:>+10.4f}  {warn}"
        )
    crit = last_state.critical_channels()
    if crit:
        typer.echo("-" * 60)
        typer.echo(f"CRITICAL channels (leading-indicator fired): {', '.join(crit)}")
    typer.echo("=" * 60)


__all__ = ["substrate_app"]
