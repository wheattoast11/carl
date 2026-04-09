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
# carl eval
# ---------------------------------------------------------------------------
@app.command(name="eval")
def eval_cmd(
    adapter: str = typer.Option(
        "wheattoast11/OmniCoder-9B-Zero-Phase2Prime",
        "--adapter", "-a",
        help="HF adapter/checkpoint ID to evaluate",
    ),
    base_model: str = typer.Option(
        "Tesslate/OmniCoder-9B",
        "--base-model", "-b",
        help="Base model ID (for Phase 2' adapter merging)",
    ),
    sft_adapter: str | None = typer.Option(
        None,
        "--sft-adapter",
        help="SFT adapter to merge before GRPO adapter",
    ),
    dataset: str = typer.Option(
        "wheattoast11/zero-rl-tool-calling-data",
        "--dataset", "-d",
        help="HF dataset ID or local path",
    ),
    data_files: str | None = typer.Option(
        None,
        "--data-files",
        help="Data files pattern (e.g. 'eval.jsonl')",
    ),
    phase: str = typer.Option(
        "auto",
        "--phase", "-p",
        help="Eval phase: 1, 2, 2prime, auto",
    ),
    threshold: float = typer.Option(
        0.30,
        "--threshold", "-t",
        help="Pass threshold for primary metric",
    ),
    max_samples: int | None = typer.Option(
        None,
        "--max-samples", "-n",
        help="Cap number of eval samples",
    ),
    max_turns: int = typer.Option(
        10,
        "--max-turns",
        help="Max multi-turn loops (Phase 2')",
    ),
    remote: bool = typer.Option(
        False,
        "--remote",
        help="Submit as HF Job instead of running locally",
    ),
    hardware: str = typer.Option(
        "l40sx1",
        "--hardware",
        help="Hardware flavor for remote eval",
    ),
    job_id: str | None = typer.Option(
        None,
        "--monitor",
        help="Monitor an existing eval job ID",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output results as JSON",
    ),
) -> None:
    """Evaluate a checkpoint with CARL eval gate.

    Examples:

      carl eval --adapter wheattoast11/OmniCoder-9B-Zero-Phase2Prime

      carl eval --phase 2prime --remote

      carl eval --monitor <job-id>
    """
    from carl_studio.eval.runner import EvalConfig, EvalRunner, EvalGate

    c = get_console()

    # -- Monitor existing job --
    if job_id:
        c.info(f"Monitoring eval job: {job_id}")
        from carl_studio.eval.runner import poll_eval_results

        report = poll_eval_results(job_id, poll_interval=15.0)
        if report is None:
            c.error("Eval job failed or timed out")
            raise typer.Exit(1)
        _render_eval_report(c, report, json_output)
        raise typer.Exit(0 if report.passed else 1)

    # -- Build config --
    eval_config = EvalConfig(
        checkpoint=adapter,
        base_model=base_model,
        sft_adapter=sft_adapter,
        dataset=dataset,
        data_files=data_files or ("eval.jsonl" if phase in ("2prime", "auto") else None),
        phase=phase,
        threshold=threshold,
        max_samples=max_samples,
        max_turns=max_turns,
    )

    _camp_header()

    if remote:
        # -- Submit remote eval --
        from carl_studio.eval.runner import submit_eval_job

        c.info(f"Submitting eval job on {hardware}...")
        c.config_block([
            ("Adapter", adapter),
            ("Base model", base_model),
            ("Phase", phase),
            ("Hardware", hardware),
            ("Max samples", str(max_samples or "all")),
        ], title="Eval Config")
        c.blank()

        try:
            submitted_job_id = submit_eval_job(eval_config, hardware=hardware)
        except Exception as e:
            c.error(f"Submission failed: {e}")
            raise typer.Exit(1)

        c.ok(f"Eval job submitted: {submitted_job_id}")
        c.info(f"Monitor: carl eval --monitor {submitted_job_id}")
        c.blank()
        raise typer.Exit(0)

    # -- Local eval --
    c.config_block([
        ("Adapter", adapter),
        ("Base model", base_model),
        ("Phase", eval_config.phase),
        ("Dataset", dataset),
        ("Threshold", f"{threshold:.0%}"),
        ("Max samples", str(max_samples or "all")),
    ], title="Eval Config")
    c.blank()
    c.info("Running eval (this may take a while)...")

    try:
        runner = EvalRunner(eval_config)
        report = runner.run()
    except Exception as e:
        c.error(f"Eval failed: {e}")
        raise typer.Exit(1)

    _render_eval_report(c, report, json_output)
    raise typer.Exit(0 if report.passed else 1)


def _render_eval_report(c: "CampConsole", report: "EvalReport", json_output: bool = False) -> None:
    """Render an EvalReport to the console."""
    import json as json_mod

    if json_output:
        output = {
            "checkpoint": report.checkpoint,
            "phase": report.phase,
            "n_samples": report.n_samples,
            "metrics": report.metrics,
            "primary_metric": report.primary_metric,
            "primary_value": report.primary_value,
            "threshold": report.threshold,
            "passed": report.passed,
        }
        if report.coherence:
            output["coherence"] = report.coherence
        from rich.syntax import Syntax
        c.print(Syntax(json_mod.dumps(output, indent=2), "json", theme="monokai"))
        return

    c.blank()
    c.header(f"Eval Results -- Phase {report.phase}")
    c.blank()

    # Metrics table
    table = c.make_table("Metric", "Value", title="Metrics")
    for k, v in report.metrics.items():
        if isinstance(v, float):
            if k.endswith("_rate") or k == "task_completion" or k == "tool_format_compliance" or k == "failure_rate":
                display = f"{v:.2%}"
            else:
                display = f"{v:.2f}"
        else:
            display = str(v)
        table.add_row(k, display)
    c.print(table)

    # Coherence
    if report.coherence:
        c.blank()
        coh_table = c.make_table("Metric", "Value", title="CARL Coherence")
        for k, v in report.coherence.items():
            coh_table.add_row(k, f"{v:.4f}")
        c.print(coh_table)

    # Gate
    c.blank()
    c.gate(
        report.passed,
        detail=f"{report.primary_metric}={report.primary_value:.2%} (threshold={report.threshold:.0%})",
    )
    c.kv("Samples", str(report.n_samples))
    c.kv("Checkpoint", report.checkpoint)
    c.blank()


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
# carl observe — zero-config rich training observation
# ---------------------------------------------------------------------------


def _sparkline(values: list[float], width: int = 40) -> str:
    """Render a list of floats as a Unicode sparkline string."""
    if not values:
        return ""
    chars = " \u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588"
    lo = min(values)
    hi = max(values)
    span = hi - lo if hi > lo else 1.0
    out = []
    # Subsample if wider than width
    step_size = max(1, len(values) // width)
    sampled = values[::step_size][:width]
    for v in sampled:
        idx = int((v - lo) / span * (len(chars) - 1))
        out.append(chars[idx])
    return "".join(out)


def _trend_arrow(values: list[float]) -> str:
    """Compute trend direction from first-half vs second-half means."""
    if len(values) < 4:
        return "~"
    half = len(values) // 2
    first = sum(values[:half]) / half
    second = sum(values[half:]) / (len(values) - half)
    diff = second - first
    if abs(diff) < 0.005:
        return "~"
    return "+" if diff > 0 else "-"


def _phase_state(phi_values: list[float], defect_densities: list[float]) -> str:
    """Determine phase state from Phi trend and defect dynamics."""
    if len(phi_values) < 4:
        return "insufficient data"
    half = len(phi_values) // 2
    phi_first = sum(phi_values[:half]) / half
    phi_second = sum(phi_values[half:]) / (len(phi_values) - half)
    phi_delta = phi_second - phi_first

    if defect_densities and len(defect_densities) >= 4:
        dd_first = sum(defect_densities[:half]) / half
        dd_second = sum(defect_densities[half:]) / (len(defect_densities) - half)
        dd_delta = dd_second - dd_first
    else:
        dd_delta = 0.0

    if phi_delta > 0.02 and dd_delta < -0.01:
        return "crystallizing"
    elif phi_delta < -0.02 and dd_delta > 0.01:
        return "melting"
    elif phi_delta > 0.02:
        return "ordering"
    elif phi_delta < -0.02:
        return "disordering"
    else:
        return "stable"


def _health_assessment(
    phi_mean: float,
    phi_trend: str,
    entropy_std: float,
    phase: str,
    frac_zero_reward: float,
    lyapunov_proxy: float,
) -> tuple[str, str]:
    """Compute health status (GREEN/YELLOW/RED) and a one-line reason."""
    reasons: list[str] = []
    severity = 0  # 0=green, 1=yellow, 2=red

    if frac_zero_reward > 0.5:
        reasons.append(f"reward starvation ({frac_zero_reward:.0%} zero-reward)")
        severity = max(severity, 2)
    elif frac_zero_reward > 0.2:
        reasons.append(f"reward sparse ({frac_zero_reward:.0%} zero-reward)")
        severity = max(severity, 1)

    if phi_mean < 0.1:
        reasons.append(f"Phi very low ({phi_mean:.3f})")
        severity = max(severity, 1)

    if phase == "melting":
        reasons.append("coherence destabilizing")
        severity = max(severity, 1)

    if lyapunov_proxy > 0.5:
        reasons.append(f"high instability (Lyapunov proxy {lyapunov_proxy:.3f})")
        severity = max(severity, 1)
    elif lyapunov_proxy < -0.3:
        reasons.append("convergent dynamics")

    if phase == "crystallizing":
        reasons.append("coherence transition in progress")

    labels = {0: "GREEN", 1: "YELLOW", 2: "RED"}
    label = labels[severity]
    reason = "; ".join(reasons) if reasons else "training dynamics healthy"
    return label, reason


def _load_frames(
    url: str | None,
    file: str | None,
    source: str,
) -> list:
    """Load ObserveFrames from the specified source. Returns (frames, source_desc)."""
    from carl_studio.observe.data_source import FileSource, TrackioSource

    if file:
        src = FileSource(file)
        frames = src.poll()
        return frames, f"file: {file}"

    if url:
        # Parse Trackio URL: https://<space>.hf.space/
        # Extract space name from URL
        import re
        match = re.match(r"https?://([^.]+(?:\.[^.]+)*?)\.hf\.space/?", url)
        if match:
            space = match.group(1)
        else:
            space = url.rstrip("/").split("/")[-1] if "/" in url else url

        src = TrackioSource(space=space)
        frames = src.poll()
        return frames, f"trackio: {space}"

    if source == "trackio":
        src = TrackioSource()
        frames = src.poll()
        return frames, "trackio: wheattoast11-trackio"

    return [], "no source"


def _render_observe_report(c: "CampConsole", frames: list, source_desc: str) -> dict:
    """Render the rich one-shot observe report. Returns computed metrics dict."""
    from carl_studio.primitives.constants import KAPPA, SIGMA

    if not frames:
        c.blank()
        c.header("CARL Observer")
        c.warn("No data found.")
        c.info(f"Source: {source_desc}")
        c.blank()
        c.print("  [camp.muted]Usage:[/]")
        c.print("    carl observe --url https://wheattoast11-trackio.hf.space/")
        c.print("    carl observe --file logs/train.jsonl")
        c.print("    carl observe --live --url https://trackio.hf.space/")
        c.blank()
        return {}

    # Extract time series
    steps = [f.step for f in frames]
    phis = [f.phi for f in frames]
    losses = [f.loss for f in frames]
    reward_means = [f.reward_mean for f in frames]

    # Aggregate per-reward series
    all_reward_keys: set[str] = set()
    for f in frames:
        all_reward_keys.update(f.rewards.keys())
    reward_series: dict[str, list[float]] = {k: [] for k in sorted(all_reward_keys)}
    for f in frames:
        for k in reward_series:
            reward_series[k].append(f.rewards.get(k, 0.0))

    # Compute entropy stats (from phi as proxy when entropy not directly available)
    # In the data model, entropy comes via trace frames
    entropies: list[float] = []
    for f in frames:
        if hasattr(f, "trace_carl_reward") and f.trace_carl_reward > 0:
            entropies.append(f.trace_carl_reward)

    # Defect densities from reward data or direct fields
    defect_densities: list[float] = []
    for f in frames:
        if hasattr(f, "rewards") and f.rewards:
            # Approximate from reward signals
            dd = f.rewards.get("reward_discontinuity", f.rewards.get("reward_carl", 0.0))
            defect_densities.append(dd)

    # Compute derived metrics
    phi_mean = sum(phis) / len(phis) if phis else 0.0
    phi_std = (sum((p - phi_mean) ** 2 for p in phis) / len(phis)) ** 0.5 if len(phis) > 1 else 0.0
    phi_min = min(phis) if phis else 0.0
    phi_max = max(phis) if phis else 0.0
    phi_trend = _trend_arrow(phis)

    loss_mean = sum(losses) / len(losses) if losses else 0.0
    reward_mean = sum(reward_means) / len(reward_means) if reward_means else 0.0

    # Entropy from phi distribution (proxy)
    if len(phis) > 1:
        import math
        # Use phi variance as entropy proxy
        entropy_mean_val = phi_std
        entropy_std_val = (sum((abs(phis[i] - phis[i - 1]) - phi_std) ** 2
                               for i in range(1, len(phis))) / (len(phis) - 1)) ** 0.5
    else:
        entropy_mean_val = 0.0
        entropy_std_val = 0.0

    # Phase state
    phase = _phase_state(phis, defect_densities)

    # Cloud quality from rewards
    cloud_vals = [f.rewards.get("reward_carl", f.rewards.get("reward_cloud", 0.0))
                  for f in frames if f.rewards]
    cloud_mean = sum(cloud_vals) / len(cloud_vals) if cloud_vals else 0.0

    # Discontinuity events: count frames where phi jumps exceed threshold
    discontinuity_threshold = 0.03
    discontinuity_events = 0
    for i in range(1, len(phis)):
        if abs(phis[i] - phis[i - 1]) > discontinuity_threshold:
            discontinuity_events += 1

    # Lyapunov proxy: average absolute delta-phi (stability indicator)
    if len(phis) > 1:
        deltas = [abs(phis[i] - phis[i - 1]) for i in range(1, len(phis))]
        lyapunov_proxy = sum(deltas) / len(deltas)
    else:
        lyapunov_proxy = 0.0

    # Conservation check: kappa * sigma should equal 4
    conservation_product = KAPPA * SIGMA
    conservation_error = abs(conservation_product - 4.0)

    # Fraction of zero-reward steps
    zero_reward_steps = sum(1 for r in reward_means if abs(r) < 1e-6)
    frac_zero_reward = zero_reward_steps / len(reward_means) if reward_means else 0.0

    # Health assessment
    health_label, health_reason = _health_assessment(
        phi_mean, phi_trend, entropy_std_val, phase, frac_zero_reward, lyapunov_proxy,
    )

    # ---- Render ----
    c.blank()
    c.header("CARL Observer")
    c.print(f"  [camp.muted]Source: {source_desc}  |  {len(frames)} steps  |  "
            f"range: {steps[0]}-{steps[-1]}[/]")
    c.blank()

    # Health badge
    health_style = {
        "GREEN": "camp.success",
        "YELLOW": "camp.accent",
        "RED": "camp.warning",
    }.get(health_label, "camp.muted")
    c.print(f"  [{health_style}]Health: {health_label}[/]  {health_reason}")
    c.blank()

    # Phi trajectory sparkline
    spark = _sparkline(phis, width=50)
    c.print(f"  [camp.primary]Phi trajectory[/]  ({phi_trend})")
    c.print(f"  {spark}")
    c.print(f"  [camp.muted][{phi_min:.4f} {'.' * 20} {phi_max:.4f}][/]")
    c.blank()

    # Metrics table
    table = c.make_table("Metric", "Value", "Trend", title="Coherence Metrics")
    table.add_row("Phi mean", f"{phi_mean:.4f}", phi_trend)
    table.add_row("Phi std", f"{phi_std:.4f}", "")
    table.add_row("Entropy (proxy)", f"{entropy_mean_val:.4f}", "")
    table.add_row("Entropy std", f"{entropy_std_val:.4f}", "")
    table.add_row("Loss mean", f"{loss_mean:.4f}", _trend_arrow(losses))
    table.add_row("Reward mean", f"{reward_mean:.4f}", _trend_arrow(reward_means))
    table.add_row("Cloud quality", f"{cloud_mean:.4f}", _trend_arrow(cloud_vals) if cloud_vals else "~")
    c.print(table)
    c.blank()

    # Phase and dynamics
    phase_table = c.make_table("Signal", "Value", title="Dynamics")
    phase_table.add_row("Phase state", phase)
    phase_table.add_row("Discontinuity events", f"{discontinuity_events} / {len(frames) - 1} steps")
    phase_table.add_row("Lyapunov proxy", f"{lyapunov_proxy:.4f}")
    phase_table.add_row("Conservation (kappa*sigma)", f"{conservation_product:.4f} (error: {conservation_error:.2e})")
    phase_table.add_row("Zero-reward fraction", f"{frac_zero_reward:.1%}")
    c.print(phase_table)
    c.blank()

    # Per-reward sparklines (if available)
    if reward_series:
        c.print("  [camp.primary]Reward channels[/]")
        for rk, rv in reward_series.items():
            short_name = rk.replace("reward_", "")
            spark_r = _sparkline(rv, width=30)
            rmean = sum(rv) / len(rv) if rv else 0.0
            c.print(f"  {short_name:<20s} {spark_r}  mean={rmean:.3f}")
        c.blank()

    # Loss sparkline
    if losses and any(l > 0 for l in losses):
        c.print(f"  [camp.primary]Loss trajectory[/]  ({_trend_arrow(losses)})")
        c.print(f"  {_sparkline(losses, width=50)}")
        c.blank()

    # Constants reminder
    c.constants()
    c.blank()

    return {
        "phi_mean": phi_mean,
        "phi_std": phi_std,
        "phi_trend": phi_trend,
        "loss_mean": loss_mean,
        "reward_mean": reward_mean,
        "phase": phase,
        "cloud_quality": cloud_mean,
        "discontinuity_events": discontinuity_events,
        "lyapunov_proxy": lyapunov_proxy,
        "health": health_label,
        "health_reason": health_reason,
        "n_frames": len(frames),
        "step_range": [steps[0], steps[-1]],
    }


def _render_diagnose(c: "CampConsole", frames: list, api_key: str | None) -> None:
    """Run Claude-powered diagnosis on the loaded frames."""
    import os

    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        c.blank()
        c.warn("--diagnose requires ANTHROPIC_API_KEY")
        c.info("Set the environment variable or pass --api-key <key>")
        c.info("Without it, carl observe still gives full local metrics above.")
        c.blank()
        return

    if not frames:
        c.warn("No data to diagnose.")
        return

    from carl_studio.primitives import CoherenceObserver, CoherenceProbe, CoherenceSnapshot

    c.blank()
    c.print("  [camp.primary]Claude-powered analysis[/]  (--diagnose)")
    c.rule()

    # Build synthetic CoherenceSnapshots from ObserveFrames for the observer
    snapshots: list[CoherenceSnapshot] = []
    for f in frames:
        snap = CoherenceSnapshot(
            step=f.step,
            n_tokens=f.trace_n_tokens if hasattr(f, "trace_n_tokens") and f.trace_n_tokens > 0 else 512,
            phi_mean=f.phi,
            phi_std=0.0,
            phi_trajectory=[f.phi],
            n_defects=0,
            n_crystallizations=f.trace_crystallizations if hasattr(f, "trace_crystallizations") else 0,
            n_meltings=f.trace_meltings if hasattr(f, "trace_meltings") else 0,
            defect_density=0.0,
            cloud_quality_mean=f.rewards.get("reward_carl", f.rewards.get("reward_cloud", 0.0)) if f.rewards else 0.0,
            scale_coherence={},
            entropy_mean=0.0,
            entropy_std=0.0,
            surprisal_mean=0.0,
            surprisal_std=0.0,
            top_k_mass=0.0,
        )
        snapshots.append(snap)

    observer = CoherenceObserver(
        api_key=key,
        observe_every=1,  # We'll force-observe anyway
        window_size=len(snapshots),
    )

    # Load all snapshots into buffer
    for snap in snapshots:
        observer._buffer.append(snap)
    if len(observer._buffer) > observer.window_size:
        observer._buffer = observer._buffer[-observer.window_size:]

    # Force the Claude call
    assessment = observer.force_observe()

    # Render assessment
    status = assessment.get("status", "unknown")
    status_style = {
        "HEALTHY": "camp.success",
        "PHASE_TRANSITION": "camp.success",
        "WARNING": "camp.accent",
        "CRITICAL": "camp.warning",
    }.get(status, "camp.muted")

    c.print(f"  [{status_style}]Status: {status}[/]")
    c.kv("Diagnosis", assessment.get("diagnosis", "N/A"), key_width=10)
    c.blank()

    signals = assessment.get("signals", [])
    if signals:
        table = c.make_table("Status", "Signal", "Detail", title="Signals")
        for sig in signals:
            sig_status = sig.get("status", "")
            icon = {
                "ok": c.theme.icons.ok,
                "watch": "~",
                "alert": c.theme.icons.warn,
            }.get(sig_status, "?")
            table.add_row(icon, sig.get("name", ""), sig.get("detail", ""))
        c.print(table)
        c.blank()

    recs = assessment.get("recommendations", [])
    if recs:
        c.print("  [camp.primary]Recommendations[/]")
        for rec in recs:
            c.info(rec)
        c.blank()

    metrics_summary = assessment.get("metrics_summary", {})
    if metrics_summary:
        table = c.make_table("Metric", "Trend", title="Claude Assessment Trends")
        for k, v in metrics_summary.items():
            table.add_row(k.replace("_", " "), str(v))
        c.print(table)
        c.blank()


@app.command()
def observe(
    url: str | None = typer.Option(None, "--url", "-u", help="Trackio space URL (e.g. https://wheattoast11-trackio.hf.space/)"),
    file: str | None = typer.Option(None, "--file", "-f", help="Local JSONL log file path"),
    live: bool = typer.Option(False, "--live", "-l", help="Launch real-time Textual TUI dashboard"),
    source: str = typer.Option("auto", "--source", "-s", help="Data source: trackio, file, or auto"),
    diagnose: bool = typer.Option(False, "--diagnose", "-d", help="Enable Claude-powered analysis (requires ANTHROPIC_API_KEY)"),
    api_key: str | None = typer.Option(None, "--api-key", envvar="ANTHROPIC_API_KEY", help="Anthropic API key for --diagnose"),
    poll: float = typer.Option(2.0, "--poll", help="Poll interval in seconds (for --live)"),
    run_name: str = typer.Option("", "--run", help="Trackio run name"),
    space: str = typer.Option("wheattoast11-trackio", "--space", help="Trackio space name (if not using --url)"),
) -> None:
    """Observe training coherence dynamics. Rich defaults, zero config.

    \b
    One-shot (default):
      carl observe --url https://wheattoast11-trackio.hf.space/
      carl observe --file logs/train.jsonl

    \b
    Live TUI:
      carl observe --live --url https://trackio.hf.space/

    \b
    Claude-powered analysis:
      carl observe --diagnose --url https://trackio.hf.space/
    """
    c = get_console()

    # Resolve source type
    resolved_source = source
    if resolved_source == "auto":
        if file:
            resolved_source = "file"
        elif url:
            resolved_source = "trackio"
        else:
            resolved_source = "trackio"

    # ---- Live TUI mode ----
    if live:
        try:
            from carl_studio.observe.app import run_app
        except ImportError:
            c.error("install carl-studio[tui] for live dashboard (textual)")
            raise typer.Exit(1)

        # Resolve path/space from --url or --file for the TUI
        tui_source = resolved_source
        tui_path = file or ""
        tui_space = space

        if url and resolved_source == "trackio":
            import re
            match = re.match(r"https?://([^.]+(?:\.[^.]+)*?)\.hf\.space/?", url)
            if match:
                tui_space = match.group(1)

        run_app(
            source=tui_source,
            path=tui_path,
            space=tui_space,
            run=run_name,
            poll=poll,
        )
        raise typer.Exit(0)

    # ---- One-shot rich report ----
    frames, source_desc = _load_frames(url=url, file=file, source=resolved_source)
    _render_observe_report(c, frames, source_desc)

    # ---- Optional Claude diagnosis ----
    if diagnose:
        _render_diagnose(c, frames, api_key)


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
    """[experimental] Start the CARL Studio MCP server (9 tools for AI agents)."""
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
# carl golf [experimental — requires zero-rl-pipeline as sibling]
# ---------------------------------------------------------------------------
golf_app = typer.Typer(name="golf", help="[experimental] Parameter Golf tools (requires zero-rl-pipeline repo)", no_args_is_help=True)
app.add_typer(golf_app)

_GOLF_BASE = Path(__file__).resolve().parent.parent.parent.parent / "zero-rl-pipeline" / "parameter-golf"


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
    """[experimental] Interactive development workflow (gated process).

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
# carl eval — replaced by eval_cmd at top of file (Phase 2' multi-turn)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# carl config — settings management
# ---------------------------------------------------------------------------

config_app = typer.Typer(name="config", help="User settings and tier management", no_args_is_help=True)
app.add_typer(config_app)


@config_app.command(name="show")
def config_show(
    unmask: bool = typer.Option(False, "--unmask", help="Show full credential values"),
) -> None:
    """Display current settings (credentials masked by default)."""
    from carl_studio.settings import CARLSettings, GLOBAL_CONFIG
    from carl_studio.tier import FEATURE_TIERS, Tier

    c = get_console()
    settings = CARLSettings.load()
    effective = settings.get_effective_tier()
    display = settings.display_dict(mask_secrets=not unmask)

    c.blank()
    c.header("CARL Settings")

    # Tier info with auto-elevation indicator
    tier_label = settings.tier.value.title()
    if effective != settings.tier:
        tier_label += f" -> {effective.value.title()} (auto-elevated)"
    c.kv("Tier", tier_label, key_width=20)
    c.kv("Preset", display["preset"], key_width=20)
    c.blank()

    # Core settings
    pairs = [
        ("default_model", display["default_model"]),
        ("default_compute", display["default_compute"]),
        ("hub_namespace", display["hub_namespace"]),
        ("naming_prefix", display["naming_prefix"]),
        ("log_level", display["log_level"]),
        ("trackio_url", display["trackio_url"]),
    ]
    c.config_block(pairs, title="Defaults")

    # Credentials
    cred_pairs = [
        ("hf_token", display["hf_token"]),
        ("anthropic_api_key", display["anthropic_api_key"]),
    ]
    c.config_block(cred_pairs, title="Credentials")

    # Observe defaults
    obs_pairs = [
        ("entropy", display["observe.entropy"]),
        ("phi", display["observe.phi"]),
        ("sparkline", display["observe.sparkline"]),
        ("poll_interval", display["observe.poll_interval"]),
        ("source", display["observe.source"]),
    ]
    c.config_block(obs_pairs, title="Observe Defaults")

    # Config file locations
    c.blank()
    c.info(f"Global config: {GLOBAL_CONFIG}")
    local = settings.model_config.get("env_prefix", "CARL_")
    c.info(f"Env prefix: {local}")

    # Feature access
    c.blank()
    gated_features = sorted(
        ((f, t) for f, t in FEATURE_TIERS.items() if t > Tier.FREE),
        key=lambda x: (x[1].value, x[0]),
    )
    if gated_features:
        table = c.make_table("Feature", "Required Tier", "Access", title="Gated Features")
        for feat, required in gated_features:
            allowed = effective >= required
            icon = c.theme.icons.ok if allowed else c.theme.icons.fail
            table.add_row(feat, required.value.title(), icon)
        c.print(table)

    c.blank()


@config_app.command(name="set")
def config_set(
    key: str = typer.Argument(..., help="Setting key (e.g. tier, default_model, log_level)"),
    value: str = typer.Argument(..., help="Value to set"),
) -> None:
    """Set a configuration value. Saves to ~/.carl/config.yaml."""
    from carl_studio.settings import CARLSettings, set_field

    c = get_console()
    settings = CARLSettings.load()

    try:
        settings = set_field(settings, key, value)
    except ValueError as exc:
        c.error(str(exc))
        raise typer.Exit(1)

    path = settings.save()
    c.ok(f"{key} = {value}")
    c.info(f"Saved to {path}")


@config_app.command(name="reset")
def config_reset(
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
) -> None:
    """Reset all settings to defaults."""
    from carl_studio.settings import reset_settings, GLOBAL_CONFIG

    c = get_console()
    if not force:
        if not GLOBAL_CONFIG.is_file():
            c.info("No config file to reset.")
            raise typer.Exit(0)
        if not typer.confirm("  Reset all settings to defaults?", default=False):
            raise typer.Exit(0)

    reset_settings()
    c.ok("Settings reset to defaults.")
    c.info(f"Removed {GLOBAL_CONFIG}")


@config_app.command(name="init")
def config_init(
    preset: str = typer.Option("", "--preset", "-p", help="Start with a preset: research, production, quick"),
    interactive: bool = typer.Option(True, "--interactive/--no-interactive", help="Interactive setup"),
) -> None:
    """Create ~/.carl/config.yaml with optional interactive prompts."""
    from carl_studio.settings import CARLSettings, GLOBAL_CONFIG, Preset, ObserveDefaults
    from carl_studio.tier import Tier

    c = get_console()

    if GLOBAL_CONFIG.is_file() and interactive:
        c.info(f"Config already exists at {GLOBAL_CONFIG}")
        if not typer.confirm("  Overwrite?", default=False):
            raise typer.Exit(0)

    settings = CARLSettings()

    if preset:
        try:
            settings.preset = Preset(preset.lower())
        except ValueError:
            c.error(f"Unknown preset '{preset}'. Use: research, production, quick")
            raise typer.Exit(1)

    if interactive:
        c.blank()
        c.header("CARL Config Setup")
        c.blank()

        # Tier
        c.print("  Subscription tier:")
        c.print("  [camp.primary][1][/] Free       -- observe, basic eval, bench")
        c.print("  [camp.accent][2][/]  Pro        -- train, align, learn, push, full eval")
        c.print("  [camp.warning][3][/] Enterprise -- MCP, custom envs, orchestration")
        tier_choice = typer.prompt("  Tier", default="1")
        tier_map = {"1": Tier.FREE, "2": Tier.PRO, "3": Tier.ENTERPRISE,
                    "free": Tier.FREE, "pro": Tier.PRO, "enterprise": Tier.ENTERPRISE}
        settings.tier = tier_map.get(tier_choice.strip().lower(), Tier.FREE)

        # Preset
        c.blank()
        c.print("  Configuration preset:")
        c.print("  [camp.primary][1][/] Research   -- verbose observe, debug logging, all metrics")
        c.print("  [camp.primary][2][/] Production -- minimal logging, auto-push, eval gating")
        c.print("  [camp.primary][3][/] Quick      -- fast defaults, L4 compute, 20 steps max")
        c.print("  [camp.primary][4][/] Custom     -- configure everything manually")
        preset_choice = typer.prompt("  Preset", default="4")
        preset_map = {"1": Preset.RESEARCH, "2": Preset.PRODUCTION, "3": Preset.QUICK, "4": Preset.CUSTOM,
                      "research": Preset.RESEARCH, "production": Preset.PRODUCTION,
                      "quick": Preset.QUICK, "custom": Preset.CUSTOM}
        settings.preset = preset_map.get(preset_choice.strip().lower(), Preset.CUSTOM)

        # Model
        c.blank()
        settings.default_model = typer.prompt(
            "  Default base model",
            default=settings.default_model,
        )

        # Hub namespace
        detected_ns = settings.hub_namespace
        if detected_ns:
            c.info(f"Detected HF namespace: {detected_ns}")
        settings.hub_namespace = typer.prompt(
            "  Hub namespace",
            default=detected_ns or "wheattoast11",
        )

        # Naming prefix
        settings.naming_prefix = typer.prompt(
            "  Naming prefix (for runs/repos)",
            default=settings.naming_prefix,
        )

        # Trackio
        c.blank()
        trackio = typer.prompt("  Trackio dashboard URL (blank to skip)", default="")
        if trackio:
            settings.trackio_url = trackio

    # Apply preset after interactive to merge
    settings = settings.model_validate(settings.model_dump())

    path = settings.save()
    c.blank()
    c.ok(f"Config saved to {path}")

    # Show summary
    display = settings.display_dict()
    pairs = [(k, v) for k, v in display.items()
             if k not in ("hf_token", "anthropic_api_key")]
    c.config_block(pairs, title="Your Settings")
    c.blank()
    c.info("Credentials are auto-detected from environment and HF hub.")
    c.info("Run 'carl config show' to see your full configuration.")
    c.blank()


@config_app.command(name="path")
def config_path() -> None:
    """Show config file locations."""
    from carl_studio.settings import GLOBAL_CONFIG, CARL_HOME, _find_local_config

    c = get_console()
    c.blank()
    c.kv("Home", str(CARL_HOME), key_width=14)
    c.kv("Global config", str(GLOBAL_CONFIG), key_width=14)
    c.kv("Global exists", "yes" if GLOBAL_CONFIG.is_file() else "no", key_width=14)

    local = _find_local_config()
    if local:
        c.kv("Local config", str(local), key_width=14)
    else:
        c.kv("Local config", "(none found)", key_width=14)
    c.blank()


@config_app.command(name="preset")
def config_preset(
    name: str = typer.Argument(..., help="Preset name: research, production, quick"),
) -> None:
    """Apply a configuration preset."""
    from carl_studio.settings import CARLSettings, Preset

    c = get_console()
    try:
        preset = Preset(name.lower())
    except ValueError:
        c.error(f"Unknown preset '{name}'. Use: research, production, quick")
        raise typer.Exit(1)

    if preset == Preset.CUSTOM:
        c.error("'custom' is not a preset. Use 'carl config set' to customize.")
        raise typer.Exit(1)

    settings = CARLSettings.load()
    settings.preset = preset
    settings = settings.model_validate(settings.model_dump())  # Re-trigger preset application
    path = settings.save()

    c.ok(f"Applied preset: {name}")
    display = settings.display_dict()
    c.config_block(
        [(k, v) for k, v in display.items()
         if k in ("default_compute", "log_level", "observe.entropy", "observe.phi",
                   "observe.sparkline", "observe.poll_interval")],
        title=f"Preset: {name}",
    )
    c.info(f"Saved to {path}")
    c.blank()


if __name__ == "__main__":
    app()
