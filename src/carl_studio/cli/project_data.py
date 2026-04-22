"""Project, checkpoint, browse, data, and setup command groups."""

from __future__ import annotations

from pathlib import Path

import typer

from carl_studio.console import get_console

from .apps import app
from .shared import _render_extra_install_hint, _slugify_identifier

# ---------------------------------------------------------------------------
# carl checkpoint — model checkpoint management
# ---------------------------------------------------------------------------

checkpoint_app = typer.Typer(
    name="checkpoint", help="Model checkpoint management", no_args_is_help=True
)
app.add_typer(checkpoint_app)


@checkpoint_app.command(name="list")
def checkpoint_list(
    model: str = typer.Argument(..., help="HuggingFace model repo ID"),
) -> None:
    """List checkpoints and pipeline state for a model."""
    c = get_console()
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        _render_extra_install_hint(c, "hf", "HF Hub support is not installed.", exc)
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
    except ImportError as exc:
        _render_extra_install_hint(c, "hf", "HF Hub support is not installed.", exc)
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
    except ImportError as exc:
        _render_extra_install_hint(c, "hf", "HF Hub support is not installed.", exc)
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


@app.command(name="setup", hidden=True)
def setup() -> None:
    """First-time setup. Pick your camp director and configure your experience."""
    from carl_studio.theme import CampTheme, Persona, save_theme, load_theme, THEME_FILE

    c = get_console()
    c.header("Welcome to Camp CARL", "Coherence-Aware RL  --  carl.camp")
    c.blank()

    from carl_studio.cli import ui

    if THEME_FILE.exists():
        current = load_theme()
        c.info(f"Current director: {current.persona.value.upper()}")
        if not ui.confirm("  Reconfigure?", default=False):
            raise typer.Exit(0)

    persona_choice = ui.select(
        "Pick your camp director",
        [
            ui.Choice(value="carl", label="CARL", hint="Methodical, precise, dry humor"),
            ui.Choice(value="carli", label="CARLI", hint="Warm, encouraging, high energy"),
        ],
        default=0,
    )
    persona = Persona.CARLI if persona_choice == "carli" else Persona.CARL
    theme = CampTheme.carl() if persona == Persona.CARL else CampTheme.carli()

    density_choice = ui.select(
        "Output style",
        [
            ui.Choice(value="normal", label="Normal", hint="balanced spacing"),
            ui.Choice(value="chill", label="Chill", hint="spacious, breathing room"),
            ui.Choice(value="focused", label="Focused", hint="compact, info-dense"),
        ],
        default=0,
    )
    theme.density = density_choice

    save_theme(theme)
    c.blank()
    c.ok(theme.voice.greeting)
    c.info(f"Config saved to {THEME_FILE}")
    c.info("Run 'carl project init' to set up your first project.")
    c.blank()


# ---------------------------------------------------------------------------
# carl project — project configuration management
# ---------------------------------------------------------------------------

project_app = typer.Typer(
    name="project", help="Project configuration (carl.yaml)", no_args_is_help=True
)
app.add_typer(project_app)


@project_app.command(name="init")
def project_init(
    name: str = typer.Option("my-carl-project", "--name", "-n", help="Project name"),
    model: str = typer.Option("", "--model", "-m", help="Base model ID"),
    method: str = typer.Option("grpo", "--method", help="Training method"),
    dataset: str = typer.Option("", "--dataset", "-d", help="Dataset repo ID or local path"),
    output_repo: str = typer.Option("", "--output-repo", help="HF repo for trained model"),
    compute: str = typer.Option("", "--compute", help="Compute target"),
    description: str = typer.Option("", "--description", help="What this project trains"),
    use_case: str = typer.Option("", "--use-case", help="Training goal in plain language"),
    output: str = typer.Option("carl.yaml", "--output", "-o", help="Output file"),
    interactive: bool = typer.Option(
        True, "--interactive/--no-interactive", help="Interactive wizard"
    ),
) -> None:
    """Initialize a new carl.yaml project file."""
    from carl_studio.settings import CARLSettings
    from carl_studio.project import CARLProject, save_project

    settings = CARLSettings.load()
    default_model = model or settings.default_model
    default_compute = compute or settings.default_compute.value

    if interactive:
        from carl_studio.cli import ui

        c = get_console()
        c.blank()
        c.header("CARL Project Init")
        name = ui.text("Project name", default=name)
        description = ui.text("What this project trains", default=description)
        default_repo = output_repo or (
            f"{settings.hub_namespace}/{_slugify_identifier(name)}"
            if settings.hub_namespace
            else ""
        )
        model = ui.text(
            "Base model ID (e.g. your-org/your-model)",
            default=default_model,
        )
        method = ui.select(
            "Training method",
            [
                ui.Choice(value="grpo", label="GRPO", badge="recommended", hint="RL with reward shaping"),
                ui.Choice(value="sft", label="SFT", hint="supervised fine-tuning"),
                ui.Choice(value="dpo", label="DPO", hint="direct preference optimization"),
            ],
            default=0,
        )
        dataset = ui.text("Dataset repo (HF ID or local path)", default=dataset)
        output_repo = ui.text("Output repo (HF ID)", default=default_repo)
        compute = ui.select(
            "Compute target",
            [
                ui.Choice(value="local", label="local", badge="recommended", hint="your own hardware"),
                ui.Choice(value="l4x1", label="l4x1", hint="1× L4 GPU (cheap)"),
                ui.Choice(value="l40sx1", label="l40sx1", hint="1× L40S GPU"),
                ui.Choice(value="a100-largex8", label="a100-largex8", hint="8× A100 (big jobs)"),
            ],
            default=0,
        )
        use_case = ui.text("Describe the training goal (optional)", default=use_case)
    else:
        model = default_model
        compute = default_compute
        if not output_repo and settings.hub_namespace:
            output_repo = f"{settings.hub_namespace}/{_slugify_identifier(name)}"

    project = CARLProject(
        name=name,
        description=description,
        base_model=model,
        output_repo=output_repo,
        method=method,
        compute_target=compute,
        dataset_repo=dataset,
        tracking_url=settings.trackio_url,
        stack={"use_case": use_case} if use_case else {},
    )
    save_project(project, output)
    c = get_console()
    c.ok(f"Project saved to {output}")
    c.config_block(
        [
            ("Model", model),
            ("Method", method),
            ("Compute", compute),
            ("Dataset", dataset or "(set later)"),
            ("Output", output_repo or "(set later)"),
        ],
        title="Project Defaults",
    )
    c.info(f"Train next: carl train --config {Path(output).name}")
    c.info("Inspect readiness: carl doctor")
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
    if proj.description:
        pairs.append(("Description", proj.description))
    if proj.stack.use_case:
        pairs.append(("Use case", proj.stack.use_case))
    if proj.tracking_url:
        pairs.append(("Tracking", proj.tracking_url))
    if proj.stack.tools:
        pairs.append(("Tools", ", ".join(proj.stack.tools)))
    if proj.stack.repos:
        pairs.append(("Repos", ", ".join(proj.stack.repos)))
    c.blank()
    c.config_block(pairs, title=f"CARL Project: {proj.name}")
    c.blank()


@project_app.command(name="set")
def project_set(
    key: str = typer.Argument(
        ..., help="Config key to set (e.g. base_model, method, compute_target)"
    ),
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

browse_app = typer.Typer(
    name="browse", help="Browse HuggingFace Hub for models, datasets, spaces", no_args_is_help=True
)
app.add_typer(browse_app)


@browse_app.command(name="models")
def browse_models(
    query: str = typer.Argument("", help="Search query"),
    task: str = typer.Option(
        "", "--task", "-t", help="Filter by task (text-generation, image-text-to-text, ...)"
    ),
    sort: str = typer.Option(
        "trending", "--sort", "-s", help="Sort: trending, downloads, likes, created"
    ),
    limit: int = typer.Option(10, "--limit", "-n", help="Number of results"),
) -> None:
    """Search HuggingFace Hub for models."""
    c = get_console()
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        _render_extra_install_hint(c, "hf", "HF Hub support is not installed.", exc)
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

    table = c.make_table(
        "Model", "Task", "Downloads", "Likes", title=f"HuggingFace Models (top {len(models)})"
    )
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
    except ImportError as exc:
        _render_extra_install_hint(c, "hf", "HF Hub support is not installed.", exc)
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

    table = c.make_table(
        "Dataset", "Downloads", "Likes", title=f"HuggingFace Datasets (top {len(datasets)})"
    )
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

data_app = typer.Typer(
    name="data", help="Data generation, validation, and statistics", no_args_is_help=True
)
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
            issues.append(f"Line {i + 1}: invalid JSON")
            continue

        # Check required fields
        if "messages" not in obj and "prompt" not in obj and "conversations" not in obj:
            issues.append(f"Line {i + 1}: missing messages/prompt/conversations field")
            continue

        # Check non-empty content
        messages = obj.get("messages") or obj.get("conversations") or []
        if isinstance(messages, list) and len(messages) < 2:
            issues.append(f"Line {i + 1}: fewer than 2 messages")
            continue

        valid += 1

    pass_rate = valid / max(total, 1)
    passed = pass_rate >= gate

    c.blank()
    c.header("Dataset Validation")
    c.config_block(
        [
            ("File", path),
            ("Total samples", str(total)),
            ("Valid samples", str(valid)),
            ("Pass rate", f"{pass_rate:.1%}"),
            ("Gate threshold", f"{gate:.1%}"),
        ]
    )
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
    c.config_block(
        [
            ("File", path),
            ("Samples", str(total)),
            ("Avg messages", f"{avg_msgs:.1f}"),
            ("Max messages", str(max_msgs)),
        ],
        title="Dataset Statistics",
    )

    table = c.make_table("Role", "Count", title="Role Distribution")
    for role, count in sorted(roles.items()):
        table.add_row(role, f"{count:,}")
    c.print(table)
    c.blank()

