"""CARL Studio CLI -- ``carl`` command.

Camp CARL: Coherence-Aware Reinforcement Learning.
Train AI models at summer camp. carl.camp
"""

from __future__ import annotations

import sys
from pathlib import Path

import typer

from carl_studio import __version__
from carl_studio.console import CampConsole, get_console

app = typer.Typer(
    name="carl",
    help=f"Camp CARL v{__version__} — Coherence-Aware RL  ·  carl.camp",
    no_args_is_help=True,
)


def _camp_header() -> None:
    """Print the Camp CARL banner with current theme."""
    c = get_console()
    c.banner(f"v{__version__}")


# ---------------------------------------------------------------------------
# Pipeline event renderer (used by --send-it)
# ---------------------------------------------------------------------------

def _render_pipeline_event(c: CampConsole, event) -> None:
    """Render a SendItPipeline event to the console."""
    stage = event.stage.value
    if stage == "failed":
        c.error(event.message)
    elif stage == "done":
        c.ok(event.message)
    elif event.progress >= 1.0:
        c.ok(f"[{stage}] {event.message}")
    else:
        c.info(f"[{stage}] {event.message}")


# ---------------------------------------------------------------------------
# carl train
# ---------------------------------------------------------------------------
@app.command()
def train(
    config: str = typer.Option("carl.yaml", "--config", "-c", help="Path to carl.yaml config"),
    model: str | None = typer.Option(None, "--model", "-m", help="Model ID (overrides config)"),
    method: str | None = typer.Option(None, "--method", help="Training method: sft, grpo, dpo, kto, orpo"),
    compute: str | None = typer.Option(None, "--compute", help="Compute target: local, l4x1, a10g-large, ..."),
    hardware: str | None = typer.Option(None, "--hardware", help="Hardware flavor (alias for --compute)"),
    max_steps: int | None = typer.Option(None, "--max-steps", help="Maximum training steps"),
    vlm: bool = typer.Option(False, "--vlm", help="Enable VLM mode (Phase 2 vision-language)"),
    gate: bool = typer.Option(False, "--gate", help="Auto phase-transition gating (eval gate between stages)"),
    send_it: bool = typer.Option(False, "--send-it", help="Full pipeline: SFT -> gate -> GRPO -> eval -> push"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what --send-it would do without executing"),
) -> None:
    """Start a CARL training run. Use --send-it for full autonomous pipeline."""
    import yaml
    from pathlib import Path
    from carl_studio.types.config import TrainingConfig

    config_path = Path(config)
    if config_path.exists():
        with open(config_path) as f:
            raw = yaml.safe_load(f) or {}
    else:
        raw = {}

    # CLI overrides
    if model:
        raw["base_model"] = model
    if method:
        raw["method"] = method
    if compute:
        raw["compute_target"] = compute
    if hardware:
        raw.setdefault("compute_target", hardware)
    if max_steps is not None:
        raw["max_steps"] = max_steps
    if vlm:
        raw["vlm_mode"] = True
    if gate:
        raw["eval_gate"] = True

    if "base_model" not in raw:
        get_console().error("--model or base_model in config required")
        raise typer.Exit(1)

    training_config = TrainingConfig(**raw)
    c = get_console()

    # --send-it / --dry-run: full pipeline mode
    if send_it or dry_run:
        from carl_studio.training.pipeline import SendItPipeline

        c.banner(f"v{__version__}")
        c.voice("send_it")

        pipeline = SendItPipeline(training_config, on_event=lambda e: _render_pipeline_event(c, e))

        if dry_run:
            plan = pipeline.plan()
            c.config_block(
                [(k, v) for k, v in plan.config_summary.items()],
                title="Send It -- Dry Run",
            )
            c.blank()
            table = c.make_table("#", "Stage", "Action", title="Pipeline Plan")
            for i, (stage, desc) in enumerate(plan.stages, 1):
                table.add_row(str(i), stage, desc)
            c.print(table)
            c.blank()
            raise typer.Exit(0)

        # Full send
        import anyio

        from rich.progress import Progress, SpinnerColumn, TextColumn
        with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=c.raw) as progress:
            task = progress.add_task("Pipeline running...", total=None)
            run = anyio.run(pipeline.run)
            progress.update(task, completed=True)

        if run.phase.value == "complete":
            c.ok("Pipeline complete")
            c.badge_award("Send It", "full pipeline run")
        else:
            c.error(f"Pipeline ended: {run.phase.value}")
            if run.error_message:
                c.info(run.error_message)
        c.blank()
        raise typer.Exit(0 if run.phase.value == "complete" else 1)

    # Standard single-run mode
    c.banner(f"v{__version__}")
    c.voice("training_start")

    pairs = [
        ("Model", training_config.base_model),
        ("Method", training_config.method.value),
        ("Compute", training_config.compute_target.value),
        ("Steps", str(training_config.max_steps)),
    ]
    if vlm:
        pairs.append(("Mode", "VLM (vision-language)"))
    if gate:
        pairs.append(("Gate", "eval gate enabled"))
    c.config_block(pairs, title="Training Config")
    c.constants()
    c.blank()

    from carl_studio.training.trainer import CARLTrainer
    import anyio

    trainer = CARLTrainer(training_config)
    run = anyio.run(trainer.train)

    if run.hub_job_id:
        c.ok(f"Job submitted: {run.hub_job_id}")
        c.info(f"Monitor: carl status {run.hub_job_id}")
    c.kv("Run ID", run.id)
    c.kv("Phase", run.phase.value)


# ---------------------------------------------------------------------------
# carl status <run_id>
# ---------------------------------------------------------------------------
@app.command()
def status(
    run_id: str = typer.Argument(..., help="HF Jobs run/job ID"),
) -> None:
    """Show job status via HF Jobs API."""
    c = get_console()
    try:
        from huggingface_hub import HfApi
    except ImportError:
        c.error("install carl-studio[hf] for job management")
        raise typer.Exit(1)

    api = HfApi()
    try:
        j = api.inspect_job(job_id=run_id)
    except Exception as exc:
        c.error(f"Inspecting job {run_id}: {exc}")
        raise typer.Exit(1)
    pairs = [("Job", run_id), ("Status", str(j.status.stage))]
    if hasattr(j.status, "message") and j.status.message:
        pairs.append(("Message", j.status.message))
    if hasattr(j, "flavor"):
        pairs.append(("Flavor", str(j.flavor)))
    c.blank()
    c.config_block(pairs)
    c.blank()


# ---------------------------------------------------------------------------
# carl logs <run_id>
# ---------------------------------------------------------------------------
@app.command()
def logs(
    run_id: str = typer.Argument(..., help="HF Jobs run/job ID"),
    tail: int = typer.Option(50, "--tail", "-n", help="Number of log entries to show"),
) -> None:
    """Stream logs from a training run."""
    c = get_console()
    try:
        from huggingface_hub import HfApi
    except ImportError:
        c.error("install carl-studio[hf] for job management")
        raise typer.Exit(1)

    api = HfApi()
    try:
        entries = list(api.fetch_job_logs(job_id=run_id))
    except Exception as exc:
        c.error(f"Fetching logs for {run_id}: {exc}")
        raise typer.Exit(1)

    if not entries:
        c.info("No log entries found.")
        raise typer.Exit(0)

    from rich.syntax import Syntax
    log_text = "\n".join(str(e)[:200] for e in entries[-tail:])
    c.print(Syntax(log_text, "log", theme="monokai", line_numbers=False))


# ---------------------------------------------------------------------------
# carl stop <run_id>
# ---------------------------------------------------------------------------
@app.command()
def stop(
    run_id: str = typer.Argument(..., help="HF Jobs run/job ID to cancel"),
) -> None:
    """Cancel a training run."""
    c = get_console()
    try:
        from huggingface_hub import HfApi
    except ImportError:
        c.error("install carl-studio[hf] for job management")
        raise typer.Exit(1)

    api = HfApi()
    try:
        api.cancel_job(job_id=run_id)
    except Exception as exc:
        c.error(f"Cancelling job {run_id}: {exc}")
        raise typer.Exit(1)

    c.ok(f"Job {run_id} cancelled.")


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
    from carl_studio.types.config import ComputeTarget

    c = get_console()
    c.blank()
    table = c.make_table("Backend", title="Compute Backends")
    for target in ComputeTarget:
        table.add_row(target.value)
    c.print(table)
    c.blank()


# ---------------------------------------------------------------------------
# carl observe — one-shot assessment or live TUI dashboard
# ---------------------------------------------------------------------------
@app.command()
def observe(
    run_id: str = typer.Argument(None, help="Run ID to observe (fetches logs for context)"),
    live: bool = typer.Option(False, "--live", "-l", help="Launch live Textual TUI dashboard"),
    source: str = typer.Option("file", "--source", "-s", help="Data source for --live: file or trackio"),
    path: str = typer.Option("", "--path", "-p", help="Log file path (for --source file)"),
    space: str = typer.Option("wheattoast11-trackio", "--space", help="Trackio space (for --source trackio)"),
    run_name: str = typer.Option("", "--run", help="Trackio run name (for --source trackio)"),
    poll: float = typer.Option(2.0, "--poll", help="Poll interval in seconds (for --live)"),
    history: bool = typer.Option(False, "--history", help="Show all past assessments"),
    api_key: str | None = typer.Option(None, "--api-key", envvar="ANTHROPIC_API_KEY", help="Anthropic API key"),
) -> None:
    """Observe training: one-shot Claude assessment (default) or live TUI (--live)."""
    c = get_console()

    # Live TUI mode
    if live:
        try:
            from carl_studio.observe.app import run_app
        except ImportError:
            c.error("install carl-studio[tui] for live dashboard (textual)")
            raise typer.Exit(1)

        run_app(source=source, path=path, space=space, run=run_name, poll=poll)
        raise typer.Exit(0)

    # One-shot Claude assessment (default)
    import json
    from carl_studio.primitives import CoherenceObserver
    observer = CoherenceObserver(api_key=api_key)

    if history:
        if not observer.history:
            c.info("No assessment history in this session.")
            raise typer.Exit(0)
        for entry in observer.history:
            c.kv("Steps", f"{entry['step_range'][0]}-{entry['step_range'][1]}")
            c.kv("Status", entry['assessment'].get('status', 'unknown'))
            c.info(entry['assessment'].get('diagnosis', ''))
        raise typer.Exit(0)

    assessment = observer.force_observe()

    c.blank()
    c.header("CARL Observer Assessment")
    status = assessment.get('status', 'unknown')
    status_method = {"HEALTHY": c.ok, "PHASE_TRANSITION": c.ok, "WARNING": c.warn, "CRITICAL": c.error}
    status_method.get(status, c.info)(f"Status: {status}")
    c.kv("Diagnosis", assessment.get('diagnosis', 'N/A'), key_width=10)

    signals = assessment.get("signals", [])
    if signals:
        table = c.make_table("Status", "Signal", "Detail", title="Signals")
        for sig in signals:
            icon = {"ok": c.theme.icons.ok, "watch": "~", "alert": c.theme.icons.warn}.get(sig.get("status", ""), "?")
            table.add_row(icon, sig.get('name', ''), sig.get('detail', ''))
        c.print(table)

    recs = assessment.get("recommendations", [])
    if recs:
        c.blank()
        c.print("  [camp.primary]Recommendations:[/]")
        for rec in recs:
            c.info(rec)

    metrics = assessment.get("metrics_summary", {})
    if metrics:
        table = c.make_table("Metric", "Trend", title="Trends")
        for k, v in metrics.items():
            table.add_row(k, str(v))
        c.print(table)
    c.blank()


# ---------------------------------------------------------------------------
# carl push (coming soon)
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
    from carl_studio.hub.models import push_with_metadata

    c = get_console()
    c.info(f"Pushing {model_path} -> {repo_id}")

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
    c.ok(f"Published: {url}")
    c.blank()


# ---------------------------------------------------------------------------
# carl mcp
# ---------------------------------------------------------------------------
@app.command(name="mcp")
def mcp_serve(
    transport: str = typer.Option("stdio", "--transport", "-t", help="Transport: stdio or http"),
    port: int = typer.Option(8100, "--port", "-p", help="HTTP port (if transport=http)"),
) -> None:
    """Start the CARL Studio MCP server."""
    c = get_console()
    try:
        from carl_studio.mcp import mcp as mcp_server
    except ImportError:
        c.error("install carl-studio[mcp] for MCP server")
        raise typer.Exit(1)

    if transport == "stdio":
        mcp_server.run(transport="stdio")
    elif transport == "http":
        mcp_server.run(transport="streamable-http", host="0.0.0.0", port=port)
    else:
        c.error(f"Unknown transport '{transport}'. Use stdio or http.")
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# carl golf
# ---------------------------------------------------------------------------
golf_app = typer.Typer(name="golf", help="Parameter Golf submission tools", no_args_is_help=True)
app.add_typer(golf_app)

GOLF_SCRIPT = Path(__file__).resolve().parent.parent.parent.parent / "zero-rl-pipeline" / "parameter-golf" / "train_gpt.py"
GOLF_CONFIGS_DIR = GOLF_SCRIPT.parent / "configs"


@golf_app.command(name="generate")
def golf_generate(
    output: str = typer.Option("-", "--output", "-o", help="Output file (- for stdout)"),
) -> None:
    """Print or write the Parameter Golf train_gpt.py submission."""
    c = get_console()
    if not GOLF_SCRIPT.exists():
        c.error(f"train_gpt.py not found at {GOLF_SCRIPT}")
        raise typer.Exit(1)
    text = GOLF_SCRIPT.read_text()
    if output == "-":
        sys.stdout.write(text)
    else:
        Path(output).write_text(text)
        c.ok(f"Written to {output} ({len(text)} bytes)")


@golf_app.command(name="configs")
def golf_configs() -> None:
    """List available Parameter Golf variant configs."""
    c = get_console()
    if not GOLF_CONFIGS_DIR.exists():
        c.error(f"Configs dir not found at {GOLF_CONFIGS_DIR}")
        raise typer.Exit(1)
    table = c.make_table("Variant", "Path", title="Parameter Golf Variants")
    for cfg in sorted(GOLF_CONFIGS_DIR.glob("*.sh")):
        table.add_row(cfg.stem, str(cfg))
    c.blank()
    c.print(table)
    c.blank()


# ---------------------------------------------------------------------------
# carl checkpoint — model checkpoint management
# ---------------------------------------------------------------------------

checkpoint_app = typer.Typer(name="checkpoint", help="Model checkpoint management", no_args_is_help=True)
app.add_typer(checkpoint_app)


@checkpoint_app.command(name="list")
def checkpoint_list(
    model: str = typer.Argument(..., help="HuggingFace model repo ID"),
) -> None:
    """List checkpoints and pipeline state for a model."""
    c = get_console()
    try:
        from huggingface_hub import HfApi
    except ImportError:
        c.error("install carl-studio[hf]")
        raise typer.Exit(1)

    import json

    api = HfApi()
    try:
        info = api.model_info(model)
    except Exception as exc:
        c.error(str(exc))
        raise typer.Exit(1)

    files = {s.rfilename: s.size for s in info.siblings}
    c.blank()
    c.kv("Model", model)
    c.kv("Modified", str(info.last_modified))
    c.kv("Files", len(files))

    # Check for pipeline_state.json
    if "pipeline_state.json" in files:
        try:
            from huggingface_hub import hf_hub_download
            state_path = hf_hub_download(model, "pipeline_state.json")
            with open(state_path) as f:
                state = json.load(f)
            c.blank()
            c.config_block([(k, v) for k, v in state.items()], title="Pipeline State")
        except Exception:
            pass

    # Check for adapter
    has_adapter = "adapter_model.safetensors" in files or "adapter_model.bin" in files
    c.kv("LoRA adapter", "yes" if has_adapter else "no")

    # Check for training_args
    if "training_args.bin" in files:
        c.kv("training_args", "yes (resumable)")

    # List related checkpoints
    sft_id = f"{model}-SFT-Gate"
    try:
        api.model_info(sft_id)
        c.ok(f"SFT Gate: {sft_id}")
    except Exception:
        c.info(f"SFT Gate: {sft_id} (not found)")

    c.blank()


@checkpoint_app.command(name="state")
def checkpoint_state(
    model: str = typer.Argument(..., help="HuggingFace model repo ID"),
) -> None:
    """Show pipeline_state.json for a checkpoint."""
    import json

    c = get_console()
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        c.error("install carl-studio[hf]")
        raise typer.Exit(1)

    try:
        state_path = hf_hub_download(model, "pipeline_state.json")
        with open(state_path) as f:
            state = json.load(f)
        from rich.syntax import Syntax
        c.print(Syntax(json.dumps(state, indent=2), "json", theme="monokai"))
    except Exception as exc:
        c.error(f"No pipeline state found: {exc}")
        raise typer.Exit(1)


@checkpoint_app.command(name="resume-info")
def checkpoint_resume_info(
    model: str = typer.Argument(..., help="HuggingFace model repo ID"),
) -> None:
    """Show what would happen on resume for this model."""
    c = get_console()
    try:
        from huggingface_hub import HfApi
    except ImportError:
        c.error("install carl-studio[hf]")
        raise typer.Exit(1)

    api = HfApi()
    sft_id = f"{model}-SFT-Gate"

    c.blank()
    c.header("Resume Analysis")
    c.kv("Model", model)

    # Check SFT checkpoint
    sft_exists = False
    try:
        info = api.model_info(sft_id)
        files = {s.rfilename for s in info.siblings}
        sft_exists = "adapter_model.safetensors" in files
    except Exception:
        pass

    if sft_exists:
        c.ok(f"SFT checkpoint: FOUND at {sft_id}")
        c.info("Resume action: Skip SFT, load checkpoint, start GRPO")
    else:
        c.warn("SFT checkpoint: NOT FOUND")
        c.info("Resume action: Run full pipeline (SFT + GRPO)")

    # Check GRPO checkpoint
    try:
        info = api.model_info(model)
        files = {s.rfilename for s in info.siblings}
        has_adapter = "adapter_model.safetensors" in files
        has_args = "training_args.bin" in files
        if has_adapter:
            c.ok(f"GRPO checkpoint: FOUND at {model}")
            if has_args:
                c.info("Resume action: Load GRPO checkpoint, resume training")
            else:
                c.info("Resume action: Load GRPO checkpoint (no training_args, fresh GRPO)")
        else:
            c.warn("GRPO checkpoint: NOT FOUND (or no adapter)")
    except Exception:
        c.warn("GRPO checkpoint: NOT FOUND")

    c.blank()


# ---------------------------------------------------------------------------
# carl setup — first-run persona selection
# ---------------------------------------------------------------------------

@app.command(name="setup")
def setup() -> None:
    """First-time setup. Pick your camp director and configure your experience."""
    from carl_studio.theme import CampTheme, Persona, save_theme, load_theme, THEME_FILE

    c = get_console()
    c.header("Welcome to Camp CARL", "Coherence-Aware RL  --  carl.camp")
    c.blank()

    if THEME_FILE.exists():
        current = load_theme()
        c.info(f"Current director: {current.persona.value.upper()}")
        if not typer.confirm("  Reconfigure?", default=False):
            raise typer.Exit(0)

    c.print("\n  Pick your camp director:\n")
    c.print('  [camp.primary][1][/] CARL  -- Methodical, precise, dry humor.')
    c.print('  [camp.muted]    "Welcome to Camp CARL."[/]')
    c.blank()
    c.print('  [camp.accent][2][/] CARLI -- Warm, encouraging, high energy.')
    c.print('  [camp.muted]    "Hey! Welcome to camp!"[/]')
    c.blank()

    choice = typer.prompt("  Your pick", default="1")
    persona = Persona.CARLI if choice.strip() in ("2", "carli", "CARLI") else Persona.CARL
    theme = CampTheme.carl() if persona == Persona.CARL else CampTheme.carli()

    c.print("\n  Output style:")
    c.print("  [camp.primary][1][/] Normal   -- balanced spacing")
    c.print("  [camp.primary][2][/] Chill    -- spacious, breathing room")
    c.print("  [camp.primary][3][/] Focused  -- compact, info-dense")
    density_choice = typer.prompt("  Style", default="1")
    theme.density = {"2": "chill", "3": "focused"}.get(density_choice.strip(), "normal")

    save_theme(theme)
    c.blank()
    c.ok(theme.voice.greeting)
    c.info(f"Config saved to {THEME_FILE}")
    c.info("Run 'carl project init' to set up your first project.")
    c.blank()


# ---------------------------------------------------------------------------
# carl project — project configuration management
# ---------------------------------------------------------------------------

project_app = typer.Typer(name="project", help="Project configuration (carl.yaml)", no_args_is_help=True)
app.add_typer(project_app)


@project_app.command(name="init")
def project_init(
    name: str = typer.Option("my-carl-project", "--name", "-n", help="Project name"),
    model: str = typer.Option("", "--model", "-m", help="Base model ID"),
    method: str = typer.Option("grpo", "--method", help="Training method"),
    output: str = typer.Option("carl.yaml", "--output", "-o", help="Output file"),
    interactive: bool = typer.Option(True, "--interactive/--no-interactive", help="Interactive wizard"),
) -> None:
    """Initialize a new carl.yaml project file."""
    from carl_studio.project import CARLProject, save_project

    if interactive and not model:
        model = typer.prompt("Base model ID (e.g. Tesslate/OmniCoder-9B)")
        name = typer.prompt("Project name", default=name)
        method = typer.prompt("Training method (sft/grpo/dpo)", default=method)
        dataset = typer.prompt("Dataset repo (HF ID or local path)", default="")
        output_repo = typer.prompt("Output repo (HF ID)", default=f"wheattoast11/{name}")
        compute = typer.prompt("Compute target (l4x1/l40sx1/a100-largex8/local)", default="l40sx1")
        use_case = typer.prompt("Describe the training goal (optional)", default="")
    else:
        dataset = ""
        output_repo = ""
        compute = "l40sx1"
        use_case = ""

    project = CARLProject(
        name=name,
        base_model=model,
        output_repo=output_repo,
        method=method,
        compute_target=compute,
        dataset_repo=dataset,
        stack={"use_case": use_case} if use_case else {},
    )
    save_project(project, output)
    c = get_console()
    c.ok(f"Project saved to {output}")
    c.config_block([("Model", model), ("Method", method), ("Compute", compute)])
    c.blank()


@project_app.command(name="show")
def project_show(
    config: str = typer.Option("carl.yaml", "--config", "-c", help="Project file"),
) -> None:
    """Display current project configuration."""
    from carl_studio.project import load_project

    c = get_console()
    try:
        proj = load_project(config)
    except FileNotFoundError:
        c.error(f"No project file found at {config}. Run: carl project init")
        raise typer.Exit(1)

    pairs = [
        ("Model", proj.base_model or "(not set)"),
        ("Adapter", proj.adapter or "(none)"),
        ("Method", proj.method),
        ("Compute", f"{proj.compute_target} via {proj.backend}"),
        ("Dataset", proj.dataset_repo or "(not set)"),
        ("Output", proj.output_repo or "(not set)"),
        ("CARL", "enabled" if proj.carl_enabled else "disabled"),
        ("Steps", str(proj.max_steps)),
        ("LR", str(proj.learning_rate)),
    ]
    if proj.stack.use_case:
        pairs.append(("Use case", proj.stack.use_case))
    if proj.stack.tools:
        pairs.append(("Tools", ", ".join(proj.stack.tools)))
    if proj.stack.repos:
        pairs.append(("Repos", ", ".join(proj.stack.repos)))
    c.blank()
    c.config_block(pairs, title=f"CARL Project: {proj.name}")
    c.blank()


@project_app.command(name="set")
def project_set(
    key: str = typer.Argument(..., help="Config key to set (e.g. base_model, method, compute_target)"),
    value: str = typer.Argument(..., help="Value to set"),
    config: str = typer.Option("carl.yaml", "--config", "-c", help="Project file"),
) -> None:
    """Set a project configuration value."""
    from carl_studio.project import load_project, save_project

    c = get_console()
    try:
        proj = load_project(config)
    except FileNotFoundError:
        c.error("No project file found. Run: carl project init")
        raise typer.Exit(1)

    if not hasattr(proj, key):
        c.error(f"Unknown key: {key}")
        c.info(f"Available: {', '.join(proj.model_fields.keys())}")
        raise typer.Exit(1)

    # Type coercion for common fields
    field_info = proj.model_fields.get(key)
    if field_info and field_info.annotation in (int, float, bool):
        if field_info.annotation is bool:
            value = value.lower() in ("true", "1", "yes")
        elif field_info.annotation is int:
            value = int(value)
        elif field_info.annotation is float:
            value = float(value)

    setattr(proj, key, value)
    save_project(proj, config)
    c.ok(f"{key} = {value}")


# ---------------------------------------------------------------------------
# carl browse — HuggingFace Hub discovery
# ---------------------------------------------------------------------------

browse_app = typer.Typer(name="browse", help="Browse HuggingFace Hub for models, datasets, spaces", no_args_is_help=True)
app.add_typer(browse_app)


@browse_app.command(name="models")
def browse_models(
    query: str = typer.Argument("", help="Search query"),
    task: str = typer.Option("", "--task", "-t", help="Filter by task (text-generation, image-text-to-text, ...)"),
    sort: str = typer.Option("trending", "--sort", "-s", help="Sort: trending, downloads, likes, created"),
    limit: int = typer.Option(10, "--limit", "-n", help="Number of results"),
) -> None:
    """Search HuggingFace Hub for models."""
    c = get_console()
    try:
        from huggingface_hub import HfApi
    except ImportError:
        c.error("install carl-studio[hf]")
        raise typer.Exit(1)

    api = HfApi()
    kwargs = {"limit": limit, "sort": sort}
    if query:
        kwargs["search"] = query
    if task:
        kwargs["pipeline_tag"] = task

    models = list(api.list_models(**kwargs))
    if not models:
        c.info("No models found.")
        raise typer.Exit(0)

    table = c.make_table("Model", "Task", "Downloads", "Likes", title=f"HuggingFace Models (top {len(models)})")
    for m in models:
        downloads = getattr(m, "downloads", 0) or 0
        likes = getattr(m, "likes", 0) or 0
        tag = getattr(m, "pipeline_tag", "") or ""
        table.add_row(str(m.id), tag, f"{downloads:,}", f"{likes:,}")
    c.blank()
    c.print(table)
    c.blank()


@browse_app.command(name="datasets")
def browse_datasets(
    query: str = typer.Argument("", help="Search query"),
    task: str = typer.Option("", "--task", "-t", help="Filter by task tag"),
    sort: str = typer.Option("trending", "--sort", "-s", help="Sort: trending, downloads, likes"),
    limit: int = typer.Option(10, "--limit", "-n", help="Number of results"),
) -> None:
    """Search HuggingFace Hub for datasets."""
    c = get_console()
    try:
        from huggingface_hub import HfApi
    except ImportError:
        c.error("install carl-studio[hf]")
        raise typer.Exit(1)

    api = HfApi()
    kwargs = {"limit": limit, "sort": sort}
    if query:
        kwargs["search"] = query
    if task:
        kwargs["tags"] = [task]

    datasets = list(api.list_datasets(**kwargs))
    if not datasets:
        c.info("No datasets found.")
        raise typer.Exit(0)

    table = c.make_table("Dataset", "Downloads", "Likes", title=f"HuggingFace Datasets (top {len(datasets)})")
    for d in datasets:
        downloads = getattr(d, "downloads", 0) or 0
        likes = getattr(d, "likes", 0) or 0
        table.add_row(str(d.id), f"{downloads:,}", f"{likes:,}")
    c.blank()
    c.print(table)
    c.blank()


# ---------------------------------------------------------------------------
# carl data — data generation and validation
# ---------------------------------------------------------------------------

data_app = typer.Typer(name="data", help="Data generation, validation, and statistics", no_args_is_help=True)
app.add_typer(data_app)


@data_app.command(name="validate")
def data_validate(
    path: str = typer.Argument(..., help="Path to JSONL dataset file"),
    gate: float = typer.Option(0.9, "--gate", "-g", help="Quality gate pass rate threshold"),
) -> None:
    """Validate a training dataset with the CARL quality gate."""
    import json as json_mod
    from pathlib import Path as P

    c = get_console()
    p = P(path)
    if not p.exists():
        c.error(f"File not found: {path}")
        raise typer.Exit(1)

    lines = p.read_text().strip().split("\n")
    total = len(lines)
    valid = 0
    issues: list[str] = []

    for i, line in enumerate(lines):
        try:
            obj = json_mod.loads(line)
        except json_mod.JSONDecodeError:
            issues.append(f"Line {i+1}: invalid JSON")
            continue

        # Check required fields
        if "messages" not in obj and "prompt" not in obj and "conversations" not in obj:
            issues.append(f"Line {i+1}: missing messages/prompt/conversations field")
            continue

        # Check non-empty content
        messages = obj.get("messages") or obj.get("conversations") or []
        if isinstance(messages, list) and len(messages) < 2:
            issues.append(f"Line {i+1}: fewer than 2 messages")
            continue

        valid += 1

    pass_rate = valid / max(total, 1)
    passed = pass_rate >= gate

    c.blank()
    c.header("Dataset Validation")
    c.config_block([
        ("File", path),
        ("Total samples", str(total)),
        ("Valid samples", str(valid)),
        ("Pass rate", f"{pass_rate:.1%}"),
        ("Gate threshold", f"{gate:.1%}"),
    ])
    c.gate(passed)

    if issues[:5]:
        c.blank()
        for issue in issues[:5]:
            c.warn(issue)
    c.blank()

    if not passed:
        raise typer.Exit(1)


@data_app.command(name="stats")
def data_stats(
    path: str = typer.Argument(..., help="Path to JSONL dataset file"),
) -> None:
    """Show statistics for a training dataset."""
    import json as json_mod
    from pathlib import Path as P

    c = get_console()
    p = P(path)
    if not p.exists():
        c.error(f"File not found: {path}")
        raise typer.Exit(1)

    lines = p.read_text().strip().split("\n")
    total = len(lines)
    roles: dict[str, int] = {}
    msg_counts: list[int] = []

    for line in lines:
        try:
            obj = json_mod.loads(line)
        except Exception:
            continue

        messages = obj.get("messages") or obj.get("conversations") or []
        msg_counts.append(len(messages))
        for msg in messages:
            role = msg.get("role", "unknown")
            roles[role] = roles.get(role, 0) + 1

    avg_msgs = sum(msg_counts) / max(len(msg_counts), 1)
    max_msgs = max(msg_counts) if msg_counts else 0

    c.blank()
    c.config_block([
        ("File", path),
        ("Samples", str(total)),
        ("Avg messages", f"{avg_msgs:.1f}"),
        ("Max messages", str(max_msgs)),
    ], title="Dataset Statistics")

    table = c.make_table("Role", "Count", title="Role Distribution")
    for role, count in sorted(roles.items()):
        table.add_row(role, f"{count:,}")
    c.print(table)
    c.blank()


# ---------------------------------------------------------------------------
# carl dev — development workflow orchestration
# ---------------------------------------------------------------------------

@app.command(name="dev")
def dev(
    phase: str = typer.Option("all", "--phase", "-p", help="Phase: ground, test, data, scripts, validate, update, all"),
) -> None:
    """Interactive development workflow (gated process).

    Runs the CARL development SOP: ground → test → data → scripts → validate → update.
    Each phase gate-checks before proceeding to the next.
    """
    phases = ["ground", "test", "data", "scripts", "validate", "update"]

    c = get_console()
    if phase != "all" and phase not in phases:
        c.error(f"Unknown phase: {phase}. Choose from: {', '.join(phases)}, all")
        raise typer.Exit(1)

    target_phases = phases if phase == "all" else phases[phases.index(phase):]

    c.blank()
    c.header("CARL Dev -- Gated Workflow")
    c.info(f"Phases: {' -> '.join(target_phases)}")
    c.blank()

    phase_hints = {
        "ground": [
            "Check library versions against PyPI/HF docs",
            "Verify model compatibility with current transformers",
            "Run: pip index versions transformers trl peft",
        ],
        "test": [
            "Run test suite to establish baseline",
            "Run: pytest tests/ -q --tb=short",
        ],
        "data": [
            "Load and validate training datasets",
            "Run: carl data validate data/train.jsonl",
        ],
        "scripts": [
            "Generate or update training scripts",
            "Run: carl bundle --config carl.yaml --output train.py",
        ],
        "validate": [
            "Submit job and run eval gate",
            "Run: carl train --config carl.yaml",
            "Then: carl eval <checkpoint> --threshold 0.5",
        ],
        "update": [
            "Update CLAUDE.md and memory with results",
            "Document what changed and why",
        ],
    }

    for p in target_phases:
        c.print(f"  [camp.primary][{p.upper()}][/]")
        for hint in phase_hints.get(p, []):
            c.print(f"    [camp.muted]{hint}[/]")

        if not typer.confirm("    Gate: proceed?", default=True):
            c.warn(f"Stopped at {p} phase.")
            raise typer.Exit(0)
        c.blank()

    c.ok("All phases complete.")
    c.blank()


# ---------------------------------------------------------------------------
# carl chat — interactive agent REPL
# ---------------------------------------------------------------------------

@app.command(name="chat")
def chat(
    model: str = typer.Option("claude-sonnet-4-20250514", "--model", "-m", help="Claude model for agent"),
    config: str = typer.Option("carl.yaml", "--config", "-c", help="Project config for context"),
    api_key: str | None = typer.Option(None, "--api-key", envvar="ANTHROPIC_API_KEY", help="Anthropic API key"),
) -> None:
    """Interactive CARL agent chat. Discuss training, get recommendations, dispatch runs."""
    c = get_console()
    if not api_key:
        c.error("ANTHROPIC_API_KEY required for carl chat")
        c.info("Set via env var or --api-key")
        raise typer.Exit(1)

    try:
        import anthropic
    except ImportError:
        c.error("install carl-studio[observe] for Anthropic SDK")
        raise typer.Exit(1)

    from carl_studio.primitives.constants import KAPPA, SIGMA

    # Load project context if available
    project_context = ""
    try:
        from carl_studio.project import load_project
        proj = load_project(config)
        project_context = f"""Current project: {proj.name}
Model: {proj.base_model}
Method: {proj.method}
Compute: {proj.compute_target} via {proj.backend}
Dataset: {proj.dataset_repo}
CARL enabled: {proj.carl_enabled}
Use case: {proj.stack.use_case}"""
    except Exception:
        project_context = "No carl.yaml found. Help the user set up a project."

    system_prompt = f"""You are CARL, a coherence-aware training assistant powered by terminals OS (Intuition Labs LLC).

Conservation law: T* = kappa * d, where kappa = {KAPPA:.4f}, sigma = {SIGMA:.4f}, kappa*sigma = 4 bits/dim.

{project_context}

You help users:
1. Configure training runs (model, hardware, data, rewards)
2. Recommend 2-3 options for any training decision
3. Explain CARL coherence metrics (Phi, cloud quality, discontinuity)
4. Dispatch training via carl CLI commands

Keep responses concise. Lead with recommendations, not explanations."""

    client = anthropic.Anthropic(api_key=api_key)
    messages: list[dict] = []

    director = c.theme.persona.value.upper()
    c.blank()
    c.header(f"Campfire Chat with {director}")
    c.kv("Model", model)
    c.info("Type 'quit' or Ctrl+C to exit.")
    c.blank()

    while True:
        try:
            user_input = input("  you> ").strip()
        except (KeyboardInterrupt, EOFError):
            c.blank()
            c.voice("farewell")
            break

        if not user_input or user_input.lower() in ("quit", "exit", "q"):
            c.voice("farewell")
            break

        messages.append({"role": "user", "content": user_input})

        try:
            response = client.messages.create(
                model=model,
                max_tokens=1024,
                system=system_prompt,
                messages=messages,
            )
            assistant_text = ""
            for block in response.content:
                if hasattr(block, "text"):
                    assistant_text += block.text

            messages.append({"role": "assistant", "content": assistant_text})
            c.blank()
            from rich.markdown import Markdown
            c.print(Markdown(f"**carl>** {assistant_text}"))
            c.blank()

        except Exception as e:
            c.error(str(e))


# ---------------------------------------------------------------------------
# carl eval — evaluation gates (delegates to eval module)
# ---------------------------------------------------------------------------

@app.command(name="eval")
def eval_cmd(
    checkpoint: str = typer.Argument(..., help="HF model ID or local checkpoint path"),
    phase: str = typer.Option("auto", "--phase", "-p", help="Eval phase: 1, 2, 2prime, auto"),
    dataset: str = typer.Option("", "--dataset", "-d", help="Eval dataset (HF ID or local path)"),
    threshold: float = typer.Option(0.5, "--threshold", "-t", help="Pass/fail threshold for primary metric"),
    max_samples: int = typer.Option(0, "--max-samples", "-n", help="Max eval samples (0=all)"),
) -> None:
    """Run evaluation gate on a checkpoint. Returns PASS/FAIL."""
    try:
        from carl_studio.eval import EvalConfig, EvalRunner
    except ImportError:
        get_console().error("carl_studio.eval module not available")
        raise typer.Exit(1)

    config = EvalConfig(
        checkpoint=checkpoint,
        phase=phase,
        dataset=dataset if dataset else checkpoint,
        threshold=threshold,
        max_samples=max_samples if max_samples > 0 else None,
    )

    c = get_console()
    c.blank()
    c.header("CARL Eval Gate")
    c.config_block([
        ("Checkpoint", checkpoint),
        ("Phase", phase),
        ("Threshold", str(threshold)),
    ])

    runner = EvalRunner(config)
    report = runner.run()

    c.blank()
    c.kv("Samples", report.n_samples)
    c.kv("Primary", f"{report.primary_metric} = {report.primary_value:.4f}")
    c.gate(report.passed)

    if report.metrics:
        table = c.make_table("Metric", "Value", title="Metrics")
        for k, v in report.metrics.items():
            table.add_row(k, f"{v:.4f}" if isinstance(v, float) else str(v))
        c.print(table)

    if report.coherence:
        table = c.make_table("Probe", "Value", title="Coherence")
        for k, v in report.coherence.items():
            if isinstance(v, float):
                table.add_row(k, f"{v:.4f}")
        c.print(table)

    if report.passed:
        c.badge_award("Eval Gate", f"{report.primary_metric} = {report.primary_value:.4f}")
    c.blank()

    if not report.passed:
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# carl bench — trainability meta-benchmarks
# ---------------------------------------------------------------------------

@app.command(name="bench")
def bench_cmd(
    model_id: str = typer.Argument(..., help="HF model ID to benchmark"),
    suite: str = typer.Option("all", "--suite", "-s", help="Probe suite: all, transition, stability, pressure, adaptation"),
    compare: str = typer.Option("", "--compare", "-c", help="Compare against baseline model"),
) -> None:
    """Measure model trainability with CARL meta-benchmarks (CTI score)."""
    try:
        from carl_studio.bench import BenchConfig, BenchSuite
    except ImportError:
        get_console().error("carl_studio.bench module not available")
        raise typer.Exit(1)

    config = BenchConfig(
        model=model_id,
        suite=suite,
        compare_model=compare if compare else None,
    )

    c = get_console()
    c.blank()
    c.header("CARL Bench -- Trainability Assessment")
    c.config_block([("Model", model_id), ("Suite", suite)])

    bench = BenchSuite(config)
    cti = bench.run()

    table = c.make_table("Probe", "Grade", "Score", "Detail", title="Results")
    for probe_name in ["transition", "stability", "pressure", "adaptation"]:
        result = getattr(cti, probe_name)
        grade_style = "camp.success" if result.grade in ("A", "B") else "camp.warning"
        from rich.text import Text
        grade = Text(result.grade, style=grade_style)
        table.add_row(probe_name, grade, f"{result.score:.2f}", result.detail)
    c.print(table)

    c.blank()
    c.kv("CTI Score", f"{cti.score:.2f}")
    c.kv("Verdict", cti.verdict)
    c.blank()


# ---------------------------------------------------------------------------
# carl align — targeted realignment
# ---------------------------------------------------------------------------

@app.command(name="align")
def align_cmd(
    mode: str = typer.Option(..., "--mode", "-m", help="Alignment mode: patterns, temporal, format"),
    source: str = typer.Option("", "--source", "-s", help="Source material (URL, file, or directory)"),
    model_id: str = typer.Option("", "--model", help="Model to align (HF ID or local path)"),
    quick: bool = typer.Option(False, "--quick", help="Quick mode with sensible defaults"),
    config: str = typer.Option("carl.yaml", "--config", "-c", help="Project config (for model defaults)"),
) -> None:
    """Targeted model realignment (patterns, temporal, or format drift)."""
    c = get_console()
    try:
        from carl_studio.align import AlignConfig, AlignMode, AlignPipeline
    except ImportError:
        c.error("carl_studio.align module not available")
        raise typer.Exit(1)

    # Resolve model from project if not specified
    if not model_id:
        try:
            from carl_studio.project import load_project
            proj = load_project(config)
            model_id = proj.base_model
        except Exception:
            c.error("--model required (no carl.yaml found)")
            raise typer.Exit(1)

    align_config = AlignConfig(
        mode=AlignMode(mode),
        model=model_id,
        source=source if source else None,
        quick=quick,
    )

    c.blank()
    c.header(f"CARL Align -- {mode.title()} Mode")
    c.config_block([("Model", model_id), ("Source", source or "(interactive)")])

    pipeline = AlignPipeline(align_config)
    result = pipeline.run()

    c.blank()
    c.config_block([(k, str(v)) for k, v in result.items()], title="Result")
    c.blank()


# ---------------------------------------------------------------------------
# carl learn — knowledge ingestion
# ---------------------------------------------------------------------------

@app.command(name="learn")
def learn_cmd(
    source: str = typer.Argument(..., help="Source material (URL, file, directory, or HF dataset)"),
    model_id: str = typer.Option("", "--model", "-m", help="Model to teach"),
    depth: str = typer.Option("shallow", "--depth", "-d", help="Depth: shallow (fewer pairs) or deep"),
    quality_threshold: float = typer.Option(0.9, "--gate", "-g", help="Quality gate threshold"),
    output: str = typer.Option("", "--output", "-o", help="Save generated pairs to JSONL"),
    config: str = typer.Option("carl.yaml", "--config", "-c", help="Project config"),
) -> None:
    """Ingest new knowledge from source material into a model."""
    c = get_console()
    try:
        from carl_studio.learn import LearnConfig, LearnPipeline
    except ImportError:
        c.error("carl_studio.learn module not available")
        raise typer.Exit(1)

    # Resolve model from project if not specified
    if not model_id:
        try:
            from carl_studio.project import load_project
            proj = load_project(config)
            model_id = proj.base_model
        except Exception:
            pass  # model is optional for learn (can just generate data)

    learn_config = LearnConfig(
        source=source,
        model=model_id,
        depth=depth,
        quality_threshold=quality_threshold,
    )

    c.blank()
    c.header("CARL Learn -- Knowledge Ingestion")
    c.config_block([("Source", source), ("Depth", depth)])

    pipeline = LearnPipeline(learn_config)
    result = pipeline.run()

    table = c.make_table("Key", "Value", title="Results")
    for k, v in result.items():
        if k == "pairs":
            continue
        table.add_row(k, f"{v:.4f}" if isinstance(v, float) else str(v))
    c.print(table)

    # Save pairs if requested
    if output and "pairs" in result:
        import json as json_mod
        with open(output, "w") as f:
            for pair in result["pairs"]:
                f.write(json_mod.dumps(pair) + "\n")
        c.ok(f"Pairs saved to {output}")
    c.blank()


# ---------------------------------------------------------------------------
# carl paper — paper generation and experimental design
# ---------------------------------------------------------------------------

paper_app = typer.Typer(name="paper", help="Research paper generation and experimental design", no_args_is_help=True)
app.add_typer(paper_app)


@paper_app.command(name="draft")
def paper_draft(
    template: str = typer.Option("unified", "--template", "-t", help="Paper template: unified, carl"),
    results_dir: str = typer.Option("experiments/results", "--results", "-r", help="Experimental results directory"),
    output: str = typer.Option("paper/output/paper.md", "--output", "-o", help="Output file"),
) -> None:
    """Generate a paper draft from experimental results."""
    from carl_studio.paper import PaperGenerator, PaperConfig

    c = get_console()
    config = PaperConfig(template=template, results_dir=results_dir, output_dir=str(Path(output).parent))
    gen = PaperGenerator(config)
    out_path = gen.save_draft(output)
    c.ok(f"Paper draft saved to {out_path}")
    c.kv("Template", template)
    c.blank()


@paper_app.command(name="figures")
def paper_figures(
    results_dir: str = typer.Option("experiments/results", "--results", "-r", help="Experimental results directory"),
    output: str = typer.Option("paper/output/generate_figures.py", "--output", "-o", help="Output script path"),
) -> None:
    """Generate a figure-generation script for the paper."""
    from carl_studio.paper import PaperGenerator, PaperConfig

    c = get_console()
    config = PaperConfig(results_dir=results_dir, output_dir=str(Path(output).parent))
    gen = PaperGenerator(config)
    out_path = gen.save_figures_script(output)
    c.ok(f"Figures script saved to {out_path}")
    c.info(f"Run: python {out_path}")
    c.blank()


@paper_app.command(name="experiments")
def paper_experiments(
    model_id: str = typer.Option("", "--model", "-m", help="Model ID for experiments"),
    output_dir: str = typer.Option("experiments/results", "--output", "-o", help="Results output directory"),
    seed: int = typer.Option(42, "--seed", help="Random seed"),
) -> None:
    """Run experimental design suite and save structured results."""
    from carl_studio.paper import ExperimentSuite

    c = get_console()
    suite = ExperimentSuite(model=model_id, output_dir=output_dir)
    results = suite.run_all(seed=seed)

    table = c.make_table("Experiment", "Verdict", title="CARL Experimental Suite")
    for r in results:
        table.add_row(r.experiment, r.verdict)
    c.blank()
    c.print(table)
    c.ok(f"Results saved to {output_dir}/")
    c.blank()


if __name__ == "__main__":
    app()
