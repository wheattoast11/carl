"""Advanced and experimental lab command surfaces."""

from __future__ import annotations

import sys
from pathlib import Path

import typer

from carl_studio.console import get_console

from .apps import app
from .shared import _render_extra_install_hint, _warn_legacy_command_alias


# ---------------------------------------------------------------------------
# carl mcp
# ---------------------------------------------------------------------------
@app.command(name="mcp", hidden=True)
def mcp_serve(
    ctx: typer.Context = typer.Option(None, hidden=True),
    transport: str = typer.Option("stdio", "--transport", "-t", help="Transport: stdio or http"),
    port: int = typer.Option(8100, "--port", "-p", help="HTTP port (if transport=http)"),
) -> None:
    """[experimental] Start the CARL Studio MCP server (9 tools for AI agents)."""
    c = get_console()
    _warn_legacy_command_alias(c, ctx, "carl lab mcp")
    from carl_studio.tier import check_tier, tier_message

    allowed, _, _ = check_tier("mcp.serve")
    if not allowed:
        c.warn(tier_message("mcp.serve") or "MCP server requires CARL Paid.")
        c.info("Upgrade: carl camp upgrade  or  https://carl.camp/pricing")
        raise typer.Exit(1)
    try:
        from carl_studio.mcp import mcp as mcp_server
    except ImportError as exc:
        _render_extra_install_hint(c, "mcp", "MCP server support is not installed.", exc)
        raise typer.Exit(1)

    if transport == "stdio":
        mcp_server.run(transport="stdio")
    elif transport == "http":
        mcp_server.run(transport="streamable-http", host="0.0.0.0", port=port)
    else:
        c.error(f"Unknown transport '{transport}'. Use stdio or http.")
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# carl golf [experimental — requires zero-rl-pipeline as sibling]
# ---------------------------------------------------------------------------
golf_app = typer.Typer(
    name="golf",
    help="[experimental] Parameter Golf tools (requires zero-rl-pipeline repo)",
    no_args_is_help=True,
)
app.add_typer(golf_app, hidden=True)


@golf_app.callback()
def golf_callback(ctx: typer.Context = typer.Option(None, hidden=True)) -> None:
    """Warn when the legacy top-level golf group is used."""
    _warn_legacy_command_alias(get_console(), ctx, "carl lab golf")


_GOLF_BASE = (
    Path(__file__).resolve().parent.parent.parent.parent.parent
    / "zero-rl-pipeline"
    / "parameter-golf"
)


def _golf_check() -> Path:
    """Verify golf dependencies exist. Fail gracefully for PyPI users."""
    script = _GOLF_BASE / "train_gpt.py"
    if not script.exists():
        get_console().error(
            f"Parameter Golf requires the zero-rl-pipeline repo as a sibling directory.\n"
            f"Expected: {script}\n"
            f"This command is for internal development only."
        )
        raise typer.Exit(1)
    return script


@golf_app.command(name="generate")
def golf_generate(
    output: str = typer.Option("-", "--output", "-o", help="Output file (- for stdout)"),
) -> None:
    """Print or write the Parameter Golf train_gpt.py submission."""
    script = _golf_check()
    text = script.read_text()
    if output == "-":
        sys.stdout.write(text)
    else:
        Path(output).write_text(text)
        get_console().ok(f"Written to {output} ({len(text)} bytes)")


@golf_app.command(name="configs")
def golf_configs() -> None:
    """List available Parameter Golf variant configs."""
    script = _golf_check()
    configs_dir = script.parent / "configs"
    if not configs_dir.exists():
        get_console().error(f"Configs dir not found at {configs_dir}")
        raise typer.Exit(1)
    c = get_console()
    table = c.make_table("Variant", "Path", title="Parameter Golf Variants")
    for cfg in sorted(configs_dir.glob("*.sh")):
        table.add_row(cfg.stem, str(cfg))
    c.blank()
    c.print(table)
    c.blank()


# ---------------------------------------------------------------------------
# carl dev — development workflow orchestration
# ---------------------------------------------------------------------------


@app.command(name="dev", hidden=True)
def dev(
    ctx: typer.Context = typer.Option(None, hidden=True),
    phase: str = typer.Option(
        "all", "--phase", "-p", help="Phase: ground, test, data, scripts, validate, update, all"
    ),
) -> None:
    """[experimental] Interactive development workflow (gated process).

    Runs the CARL development SOP: ground → test → data → scripts → validate → update.
    Each phase gate-checks before proceeding to the next.
    """
    phases = ["ground", "test", "data", "scripts", "validate", "update"]

    c = get_console()
    _warn_legacy_command_alias(c, ctx, "carl lab dev")
    if phase != "all" and phase not in phases:
        c.error(f"Unknown phase: {phase}. Choose from: {', '.join(phases)}, all")
        raise typer.Exit(1)

    target_phases = phases if phase == "all" else phases[phases.index(phase) :]

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


@app.command(name="chat", hidden=True)
def chat_repl(
    ctx: typer.Context = typer.Option(None, hidden=True),
    model: str = typer.Option("claude-sonnet-4-6", "--model", "-m", help="Claude model for agent"),
    config: str = typer.Option("carl.yaml", "--config", "-c", help="Project config for context"),
    api_key: str | None = typer.Option(
        None, "--api-key", envvar="ANTHROPIC_API_KEY", help="Anthropic API key"
    ),
) -> None:
    """[legacy] Simple REPL chat. For the full agentic loop use: carl chat."""
    c = get_console()
    _warn_legacy_command_alias(c, ctx, "carl lab chat")
    if not api_key:
        c.error("ANTHROPIC_API_KEY required for carl chat")
        c.info("Set via env var or --api-key")
        raise typer.Exit(1)

    try:
        import anthropic
    except ImportError as exc:
        _render_extra_install_hint(
            c,
            "observe",
            "Anthropic SDK support is not installed.",
            exc,
            quoted=True,
        )
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
# carl eval — replaced by eval_cmd at top of file (Phase 2' multi-turn)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# carl bench — trainability meta-benchmarks
# ---------------------------------------------------------------------------


@app.command(name="bench", hidden=True)
def bench_cmd(
    ctx: typer.Context = typer.Option(None, hidden=True),
    model_id: str = typer.Argument(..., help="HF model ID to benchmark"),
    suite: str = typer.Option(
        "all", "--suite", "-s", help="Probe suite: all, transition, stability, pressure, adaptation"
    ),
    compare: str = typer.Option("", "--compare", "-c", help="Compare against baseline model"),
) -> None:
    """Measure model trainability with CARL meta-benchmarks (CTI score)."""
    _warn_legacy_command_alias(get_console(), ctx, "carl lab bench")
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


@app.command(name="align", hidden=True)
def align_cmd(
    ctx: typer.Context = typer.Option(None, hidden=True),
    mode: str = typer.Option(
        ..., "--mode", "-m", help="Alignment mode: patterns, temporal, format"
    ),
    source: str = typer.Option(
        "", "--source", "-s", help="Source material (URL, file, or directory)"
    ),
    model_id: str = typer.Option("", "--model", help="Model to align (HF ID or local path)"),
    quick: bool = typer.Option(False, "--quick", help="Quick mode with sensible defaults"),
    config: str = typer.Option(
        "carl.yaml", "--config", "-c", help="Project config (for model defaults)"
    ),
) -> None:
    """Targeted model realignment (patterns, temporal, or format drift)."""
    c = get_console()
    _warn_legacy_command_alias(c, ctx, "carl lab align")
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


@app.command(name="learn", hidden=True)
def learn_cmd(
    ctx: typer.Context = typer.Option(None, hidden=True),
    source: str = typer.Argument(..., help="Source material (URL, file, directory, or HF dataset)"),
    model_id: str = typer.Option("", "--model", "-m", help="Model to teach"),
    depth: str = typer.Option(
        "shallow", "--depth", "-d", help="Depth: shallow (fewer pairs) or deep"
    ),
    quality_threshold: float = typer.Option(0.9, "--gate", "-g", help="Quality gate threshold"),
    output: str = typer.Option("", "--output", "-o", help="Save generated pairs to JSONL"),
    config: str = typer.Option("carl.yaml", "--config", "-c", help="Project config"),
    synthesize: bool = typer.Option(
        False, "--synthesize", "-s", help="Synthesize graded RL training samples from codebase"
    ),
    count: int = typer.Option(
        10, "--count", help="Target number of valid samples (synthesize mode)"
    ),
    kit: str = typer.Option(
        "", "--kit", "-k", help="Kit ID (coding-agent, tool-specialist, reasoning, mcp-agent)"
    ),
    recipe: str = typer.Option(
        "", "--recipe", "-r", help="Recipe YAML file for multi-course training"
    ),
    frame: str = typer.Option(
        "", "--frame", help="WorkFrame: 'current' for active frame, or path to frame YAML"
    ),
) -> None:
    """Ingest new knowledge from source material into a model."""
    c = get_console()
    _warn_legacy_command_alias(c, ctx, "carl lab learn")
    try:
        from carl_studio.learn import LearnConfig, LearnPipeline
    except ImportError:
        c.error("carl_studio.learn module not available")
        raise typer.Exit(1)

    # Natural language detection
    from carl_studio.learn.ingest import SourceIngester, SourceType

    try:
        detected = SourceIngester._detect_type(source)
    except ValueError:
        detected = None

    if detected == SourceType.NATURAL:
        try:
            from carl_studio.learn.planner import interpret_natural

            c.header("CARL Learn", "Interpreting...")
            plan = interpret_natural(source)
            c.config_block(
                [
                    ("Sources", ", ".join(plan.sources) if plan.sources else "(none)"),
                    ("Kit", plan.kit),
                    ("Action", plan.action),
                    ("Samples", str(plan.count)),
                ],
                title="Interpreted Plan",
            )
            c.info(plan.explanation)
            if not typer.confirm("Proceed?"):
                raise typer.Exit(0)
            # Apply interpreted plan to local vars
            source = plan.sources[0] if plan.sources else source
            kit = plan.kit or kit
            count = plan.count or count
            if plan.action in ("synthesize", "train"):
                synthesize = True
        except RuntimeError as e:
            c.error(f"No LLM available for interpretation: {e}")
            c.info("Set ANTHROPIC_API_KEY, OPENROUTER_API_KEY, or OPENAI_API_KEY")
            raise typer.Exit(1)

    if synthesize:
        from carl_studio.learn.synthesize import SynthesizeConfig, SynthesizePipeline

        synth_config = SynthesizeConfig(source=source, count=count, output=output)
        c.header("CARL Learn", "Synthesize Mode")
        result = SynthesizePipeline(synth_config, c).run()
        c.blank()
        if kit:
            c.info(
                f"Kit '{kit}' selected — use with: carl train --kit {kit} --dataset {result.output_path}"
            )
        raise typer.Exit(0)

    if recipe:
        from carl_studio.data.recipe import load_recipe

        r = load_recipe(recipe)
        c.header("CARL Learn", f"Recipe: {r.name}")
        c.config_block(
            [
                ("Base", r.base),
                ("Courses", str(len(r.courses))),
                ("Source", r.source or source),
            ]
        )
        for i, course in enumerate(r.courses):
            c.kv(f"Course {i + 1}", f"kit={course.kit}, steps={course.steps}")
        c.info("Recipe loaded. Run: carl train --recipe " + recipe)
        raise typer.Exit(0)

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

    if kit:
        from carl_studio.data.kits import KitRegistry

        k = KitRegistry().get(kit)
        c.kv("Kit", f"{k.name} — {k.description}")

    # Resolve WorkFrame if specified
    frame_prefix = ""
    if frame:
        from carl_studio.frame import WorkFrame

        if frame == "current":
            wf = WorkFrame.from_project(config)
        else:
            wf = WorkFrame.load(frame)
        if wf.active:
            frame_prefix = wf.attention_query()
            c.kv("Frame", f"{wf.domain}/{wf.function}/{wf.role}")

    pipeline = LearnPipeline(learn_config)
    result = pipeline.run(context_prefix=frame_prefix)

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

paper_app = typer.Typer(
    name="paper", help="Research paper generation and experimental design", no_args_is_help=True
)
app.add_typer(paper_app, hidden=True)


@paper_app.callback()
def paper_callback(ctx: typer.Context = typer.Option(None, hidden=True)) -> None:
    """Warn when the legacy top-level paper group is used."""
    _warn_legacy_command_alias(get_console(), ctx, "carl lab paper")


@paper_app.command(name="draft")
def paper_draft(
    template: str = typer.Option(
        "unified", "--template", "-t", help="Paper template: unified, carl"
    ),
    results_dir: str = typer.Option(
        "experiments/results", "--results", "-r", help="Experimental results directory"
    ),
    output: str = typer.Option("paper/output/paper.md", "--output", "-o", help="Output file"),
) -> None:
    """Generate a paper draft from experimental results."""
    from carl_studio.paper import PaperGenerator, PaperConfig

    c = get_console()
    config = PaperConfig(
        template=template, results_dir=results_dir, output_dir=str(Path(output).parent)
    )
    gen = PaperGenerator(config)
    out_path = gen.save_draft(output)
    c.ok(f"Paper draft saved to {out_path}")
    c.kv("Template", template)
    c.blank()


@paper_app.command(name="figures")
def paper_figures(
    results_dir: str = typer.Option(
        "experiments/results", "--results", "-r", help="Experimental results directory"
    ),
    output: str = typer.Option(
        "paper/output/generate_figures.py", "--output", "-o", help="Output script path"
    ),
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
    output_dir: str = typer.Option(
        "experiments/results", "--output", "-o", help="Results output directory"
    ),
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


# ---------------------------------------------------------------------------
# carl admin — hardware-gated admin unlock
# ---------------------------------------------------------------------------

admin_app = typer.Typer(name="admin", help="Admin unlock (hardware-gated)", no_args_is_help=True)
app.add_typer(admin_app, hidden=True)


@admin_app.callback()
def admin_callback(ctx: typer.Context = typer.Option(None, hidden=True)) -> None:
    """Warn when the legacy top-level admin group is used."""
    _warn_legacy_command_alias(get_console(), ctx, "carl lab admin")


@admin_app.command(name="unlock")
def admin_unlock() -> None:
    """Generate and write the admin key for this machine.

    Requires CARL_ADMIN_SECRET in environment.
    Run once per machine. The key is hardware-specific.
    """
    c = get_console()
    try:
        from carl_studio.admin import write_admin_key

        path = write_admin_key()
        c.ok(f"Admin key written to {path}")
        c.info("Admin mode active on this machine.")
    except RuntimeError as exc:
        c.error(str(exc))
        raise typer.Exit(1)
    except Exception as exc:
        c.error(f"Failed: {exc}")
        raise typer.Exit(1)


@admin_app.command(name="status")
def admin_status_cmd() -> None:
    """Show admin unlock status for this machine."""
    from carl_studio.admin import admin_status

    c = get_console()
    info = admin_status()
    c.blank()
    pairs = [(k.replace("_", " ").title(), v) for k, v in info.items()]
    status_label = info["status"]
    style = "camp.success" if status_label == "UNLOCKED" else "camp.muted"
    c.print(f"  [{style}]Admin: {status_label}[/]")
    c.config_block([(k, v) for k, v in pairs if k != "Status"], title="Admin Details")
    c.blank()


@admin_app.command(name="clear")
def admin_clear() -> None:
    """Remove the admin key from this machine (lock admin mode)."""
    from carl_studio.admin import clear_admin_key, _ADMIN_KEY_PATH

    c = get_console()
    if not _ADMIN_KEY_PATH.exists():
        c.info("Admin key not present. Already locked.")
        raise typer.Exit(0)
    if not typer.confirm("  Remove admin key and lock admin mode?", default=False):
        raise typer.Exit(0)
    clear_admin_key()
    c.ok("Admin key removed. Admin mode locked.")
