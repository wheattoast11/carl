"""Core utility commands (bundle, compute, push)."""

from __future__ import annotations

import sys
from pathlib import Path

import typer

from carl_studio.console import get_console

from .apps import app
from .shared import _render_extra_install_hint

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
# carl push
# ---------------------------------------------------------------------------
@app.command()
def push(
    model_path: str = typer.Argument(..., help="Local model/adapter directory to push"),
    repo_id: str = typer.Option(..., "--repo", "-r", help="HuggingFace Hub repo ID"),
    base_model: str = typer.Option("", "--base-model", "-b", help="Base model ID for model card"),
    method: str = typer.Option("grpo", "--method", help="Training method (sft, grpo, dpo)"),
    dataset: str = typer.Option("", "--dataset", "-d", help="Training dataset ID"),
    private: bool = typer.Option(False, "--private", help="Make repo private"),
) -> None:
    """Push model to HuggingFace Hub with CARL coherence metadata."""
    import anyio

    c = get_console()
    artifact_path = Path(model_path)
    if not artifact_path.exists():
        c.error(f"Model path not found: {model_path}")
        raise typer.Exit(1)

    try:
        from carl_studio.hub.models import push_with_metadata
    except ImportError as exc:
        _render_extra_install_hint(c, "hf", "Hub publishing support is not installed.", exc)
        raise typer.Exit(1)

    c.info(f"Publishing artifacts from {model_path} -> {repo_id}")

    try:
        url = anyio.run(
            push_with_metadata,
            model_path,
            repo_id,
            base_model,
            method,
            dataset,
            None,  # coherence_metrics
            private,
        )
    except Exception as exc:
        c.error(f"Publish failed: {exc}")
        raise typer.Exit(1)

    c.ok(f"Published: {url}")
    c.blank()

