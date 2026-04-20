"""Core utility commands (bundle, compute).

``carl push`` lives in ``carl_studio.cli.push`` since v0.7 where it was
promoted to a first-class verb with an expanded signature (run id
lookup, positional repo argument, etc.). Importing ``push`` from this
module still works via the shim below so downstream scripts that did
``from carl_studio.cli.core import push`` keep resolving.
"""

from __future__ import annotations

import sys

import typer

from carl_studio.console import get_console

from .apps import app

# ---------------------------------------------------------------------------
# carl bundle
# ---------------------------------------------------------------------------
@app.command()
def bundle(
    config: str = typer.Option("carl.yaml", "--config", "-c", help="Config file"),
    output: str = typer.Option("-", "--output", "-o", help="Output file (- for stdout)"),
) -> None:
    """Generate a self-contained HF Jobs training script."""
    import yaml
    from pathlib import Path
    from carl_studio.types.config import TrainingConfig
    from carl_studio.bundler import Bundler

    c = get_console()
    config_path = Path(config)
    if not config_path.exists():
        c.error(f"Config file not found: {config}")
        raise typer.Exit(1)

    with open(config_path) as f:
        raw = yaml.safe_load(f) or {}

    if "base_model" not in raw:
        c.error("base_model required in config")
        raise typer.Exit(1)

    training_config = TrainingConfig(**raw)
    script = Bundler().generate(training_config)

    if output == "-":
        sys.stdout.write(script)
    else:
        Path(output).write_text(script)
        c.ok(f"Bundled script written to {output}")


# ---------------------------------------------------------------------------
# carl compute
# ---------------------------------------------------------------------------
@app.command(name="compute")
def compute_cmd() -> None:
    """List available compute backends."""
    from carl_studio.compute import _BYOK_BACKENDS

    c = get_console()
    c.blank()
    table = c.make_table("Backend", "Type", "Credentials", title="Compute Backends")
    for name in _BYOK_BACKENDS:
        table.add_row(name, "BYOK", "Your API key")
    table.add_row("camp", "Managed", "carl.camp account")
    c.print(table)
    c.blank()



# ---------------------------------------------------------------------------
# carl push — promoted to `carl_studio.cli.push`. The wiring below keeps a
# `push` symbol exported from this module for legacy `from carl_studio.cli.core
# import push` imports. The actual command is registered via `cli/wiring.py`.
# ---------------------------------------------------------------------------
from .push import push_cmd as push  # noqa: E402,F401  — re-export shim

