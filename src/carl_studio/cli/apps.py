"""CLI application roots for the `carl` command."""

from __future__ import annotations

import typer

from carl_studio import __version__

_APP_HELP = f"""Camp CARL v{__version__} — Coherence-Aware RL  ·  carl.camp

Start here:
  carl start
  carl doctor
  carl train ...

Core workbench:
  project / train / run / observe / eval / infer

Attach platform:
  camp / config / compute / browse
"""

app = typer.Typer(
    name="carl",
    help=_APP_HELP,
    no_args_is_help=False,  # bare `carl` routes to chat via the default callback
)

camp_app = typer.Typer(
    name="camp",
    help="carl.camp account, sync, credits, marketplace, and paid platform surfaces.",
    no_args_is_help=True,
)
app.add_typer(camp_app)

lab_app = typer.Typer(
    name="lab",
    help="Advanced, experimental, and internal CARL surfaces.",
    no_args_is_help=True,
)
app.add_typer(lab_app)

__all__ = ["app", "camp_app", "lab_app"]
