"""Training, eval, and run lifecycle commands."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import typer
from pydantic import ValidationError

from carl_studio import __version__
from carl_studio.console import CampConsole, get_console

from .apps import app
from .shared import (
    _camp_header,
    _lookup_local_run,
    _normalize_remote_status,
    _normalize_training_config,
    _now_iso,
    _persist_training_run,
    _render_extra_install_hint,
    _render_local_run_record,
    _render_pipeline_event,
    _render_training_config_error,
    _resolve_run_reference,
    _run_remote_id,
    _safe_json_object,
)

if TYPE_CHECKING:
    from carl_studio.eval.runner import EvalReport

# ---------------------------------------------------------------------------
# carl run
# ---------------------------------------------------------------------------
run_app = typer.Typer(
    name="run",
    help="Inspect and manage local run history plus remote job IDs.",
    no_args_is_help=True,
)
app.add_typer(run_app)


@run_app.command(name="list")
def run_list(
    limit: int = typer.Option(10, "--limit", "-n", help="Maximum runs to show"),
    status_filter: str = typer.Option("", "--status", help="Filter by local status"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """List locally tracked training and pipeline runs."""
    from carl_studio.db import LocalDB

    c = get_console()
    runs = LocalDB().list_runs(limit=limit, status=status_filter or None)

    if json_output:
        typer.echo(json.dumps(runs, indent=2, default=str))
        raise typer.Exit(0)

    c.blank()
    c.header("CARL Runs")

    if not runs:
        c.info("No local runs tracked yet.")
        c.info("Start one with: carl train --config carl.yaml")
        c.blank()
        raise typer.Exit(0)

    table = c.make_table("Run ID", "Mode", "Status", "Model", "Remote", "Created")
    for run_record in runs:
        remote_id = _run_remote_id(run_record)
        created = str(run_record.get("created_at", ""))[:19]
        table.add_row(
            str(run_record.get("id", "")),
            str(run_record.get("mode", "")),
            str(run_record.get("status", "")),
            str(run_record.get("model_id", "")),
            remote_id or "-",
            created or "-",
        )
    c.print(table)
    c.blank()


@run_app.command(name="show")
def run_show(
    run_id: str = typer.Argument(..., help="Local run ID"),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Show the locally tracked details for a run."""
    local_run = _lookup_local_run(run_id)
    c = get_console()
    if local_run is None:
        c.error(f"Run not found: {run_id}")
        c.info("Use 'carl run list' to see stored runs.")
        raise typer.Exit(1)

    if json_output:
        typer.echo(json.dumps(local_run, indent=2, default=str))
        raise typer.Exit(0)

    _render_local_run_record(c, local_run, title="Stored Run")
    remote_id = _run_remote_id(local_run)
    if remote_id:
        c.info(f"Remote status: carl run status {run_id}")
        c.info(f"Remote logs: carl run logs {run_id}")
    c.blank()


@run_app.command(name="status")
def run_status_cmd(
    run_id: str = typer.Argument(..., help="Local run ID or remote job ID"),
) -> None:
    """Show remote status, resolving a local run ID when possible."""
    status(run_id)


@run_app.command(name="logs")
def run_logs_cmd(
    run_id: str = typer.Argument(..., help="Local run ID or remote job ID"),
    tail: int = typer.Option(50, "--tail", "-n", help="Number of log entries to show"),
) -> None:
    """Show remote logs, resolving a local run ID when possible."""
    logs(run_id, tail=tail)


@run_app.command(name="stop")
def run_stop_cmd(
    run_id: str = typer.Argument(..., help="Local run ID or remote job ID"),
) -> None:
    """Cancel a remote job, resolving a local run ID when possible."""
    stop(run_id)


# ---------------------------------------------------------------------------
# carl eval
# ---------------------------------------------------------------------------
@app.command(name="eval")
def eval_cmd(
    adapter: str = typer.Option(
        "wheattoast11/OmniCoder-9B-Zero-Phase2Prime",
        "--adapter",
        "-a",
        help="HF adapter/checkpoint ID to evaluate",
    ),
    base_model: str = typer.Option(
        "Tesslate/OmniCoder-9B",
        "--base-model",
        "-b",
        help="Base model ID (for Phase 2' adapter merging)",
    ),
    sft_adapter: str | None = typer.Option(
        None,
        "--sft-adapter",
        help="SFT adapter to merge before GRPO adapter",
    ),
    dataset: str = typer.Option(
        "wheattoast11/zero-rl-tool-calling-data",
        "--dataset",
        "-d",
        help="HF dataset ID or local path",
    ),
    data_files: str | None = typer.Option(
        None,
        "--data-files",
        help="Data files pattern (e.g. 'eval.jsonl')",
    ),
    phase: str = typer.Option(
        "auto",
        "--phase",
        "-p",
        help="Eval phase: 1, 2, 2prime, auto",
    ),
    threshold: float = typer.Option(
        0.30,
        "--threshold",
        "-t",
        help="Pass threshold for primary metric",
    ),
    max_samples: int | None = typer.Option(
        None,
        "--max-samples",
        "-n",
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
    from carl_studio.eval.runner import EvalConfig, EvalRunner

    c = get_console()

    # -- Monitor existing job --
    if job_id:
        c.info(f"Monitoring eval job: {job_id}")
        from carl_studio.eval.runner import poll_eval_results

        try:
            report = poll_eval_results(job_id, poll_interval=15.0)
        except ImportError as exc:
            _render_extra_install_hint(c, "hf", "HF Jobs support is not installed.", exc)
            raise typer.Exit(1)
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
        c.config_block(
            [
                ("Adapter", adapter),
                ("Base model", base_model),
                ("Phase", phase),
                ("Hardware", hardware),
                ("Max samples", str(max_samples or "all")),
            ],
            title="Eval Config",
        )
        c.blank()

        try:
            submitted_job_id = submit_eval_job(eval_config, hardware=hardware)
        except ImportError as exc:
            _render_extra_install_hint(c, "hf", "HF Jobs support is not installed.", exc)
            raise typer.Exit(1)
        except Exception as e:
            c.error(f"Submission failed: {e}")
            raise typer.Exit(1)

        c.ok(f"Eval job submitted: {submitted_job_id}")
        c.info(f"Monitor: carl eval --monitor {submitted_job_id}")
        c.blank()
        raise typer.Exit(0)

    # -- Local eval --
    c.config_block(
        [
            ("Adapter", adapter),
            ("Base model", base_model),
            ("Phase", eval_config.phase),
            ("Dataset", dataset),
            ("Threshold", f"{threshold:.0%}"),
            ("Max samples", str(max_samples or "all")),
        ],
        title="Eval Config",
    )
    c.blank()
    c.info("Running eval (this may take a while)...")

    try:
        runner = EvalRunner(eval_config)
        report = runner.run()
    except ImportError as exc:
        _render_extra_install_hint(c, "training", "Eval dependencies are not installed.", exc)
        raise typer.Exit(1)
    except Exception as e:
        c.error(f"Eval failed: {e}")
        raise typer.Exit(1)

    _render_eval_report(c, report, json_output)
    raise typer.Exit(0 if report.passed else 1)


def _render_eval_report(c: CampConsole, report: EvalReport, json_output: bool = False) -> None:
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
            if (
                k.endswith("_rate")
                or k == "task_completion"
                or k == "task_completion_rate"
                or k == "tool_format_compliance"
                or k == "failure_rate"
            ):
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
    name: str | None = typer.Option(
        None, "--name", help="Run name (defaults from model/output repo)"
    ),
    model: str | None = typer.Option(None, "--model", "-m", help="Model ID (overrides config)"),
    dataset: str | None = typer.Option(
        None, "--dataset", "-d", help="Dataset repo ID or local path"
    ),
    output_repo: str | None = typer.Option(
        None, "--output-repo", "-o", help="HF repo to push checkpoints to"
    ),
    method: str | None = typer.Option(
        None, "--method", help="Training method: sft, grpo, dpo, kto, orpo"
    ),
    compute: str | None = typer.Option(
        None, "--compute", help="Compute target: local, l4x1, a10g-large, a100-large, ..."
    ),
    hardware: str | None = typer.Option(
        None, "--hardware", help="Hardware flavor (alias for --compute)"
    ),
    max_steps: int | None = typer.Option(None, "--max-steps", help="Maximum training steps"),
    vlm: bool = typer.Option(False, "--vlm", help="Enable VLM mode (Phase 2 vision-language)"),
    gate: bool = typer.Option(
        False, "--gate", help="Auto phase-transition gating (eval gate between stages)"
    ),
    send_it: bool = typer.Option(
        False, "--send-it", help="Full pipeline: SFT -> gate -> GRPO -> eval -> push"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show what --send-it would do without executing"
    ),
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
    if name:
        raw["run_name"] = name
    if model:
        raw["base_model"] = model
    if dataset:
        raw["dataset_repo"] = dataset
    if output_repo:
        raw["output_repo"] = output_repo
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

    c = get_console()
    raw = _normalize_training_config(raw)
    try:
        training_config = TrainingConfig(**raw)
    except ValidationError as exc:
        _render_training_config_error(c, raw, exc)
        raise typer.Exit(1)

    # --send-it / --dry-run: full pipeline mode
    if send_it or dry_run:
        if send_it:
            from carl_studio.tier import check_tier, tier_message

            allowed, _, _ = check_tier("train.send_it")
            if not allowed:
                c.warn(tier_message("train.send_it") or "--send-it requires CARL Paid.")
                c.info("Upgrade: carl login --upgrade  or  https://carl.camp/pricing")
                raise typer.Exit(1)
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

        try:
            _persist_training_run(training_config, run, mode="pipeline")
        except Exception as exc:
            c.warn(f"Local run history not updated: {exc}")

        if run.phase.value == "complete":
            c.ok("Pipeline complete")
            c.badge_award("Send It", "full pipeline run")
        else:
            c.error(f"Pipeline ended: {run.phase.value}")
            if run.error_message:
                c.info(run.error_message)
        c.info(f"Run details: carl run show {run.id}")
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

    try:
        from carl_studio.training.trainer import CARLTrainer
        import anyio
    except ImportError as exc:
        _render_extra_install_hint(c, "training", "Training dependencies are not installed.", exc)
        raise typer.Exit(1)

    trainer = CARLTrainer(training_config)
    try:
        run = anyio.run(trainer.train)
    except ImportError as exc:
        _render_extra_install_hint(c, "training", "Training dependencies are not installed.", exc)
        raise typer.Exit(1)

    try:
        _persist_training_run(training_config, run, mode=f"train:{training_config.method.value}")
    except Exception as exc:
        c.warn(f"Local run history not updated: {exc}")

    if run.hub_job_id:
        c.ok(f"Job submitted: {run.hub_job_id}")
        c.info(f"Monitor remote job: carl status {run.hub_job_id}")
        c.info(f"Monitor via local run: carl run status {run.id}")
    c.kv("Run ID", run.id)
    c.kv("Phase", run.phase.value)
    c.info(f"Run details: carl run show {run.id}")


# ---------------------------------------------------------------------------
# carl status <run_id>
# ---------------------------------------------------------------------------
@app.command(hidden=True)
def status(
    run_id: str = typer.Argument(..., help="Local run ID or HF Jobs run/job ID"),
) -> None:
    """Show job status via HF Jobs API or a stored local run ID."""
    c = get_console()
    local_run, remote_job_id = _resolve_run_reference(run_id)

    if local_run is not None and not remote_job_id:
        _render_local_run_record(c, local_run, title="Local Run Status")
        c.info("This run does not have a remote job ID yet.")
        raise typer.Exit(0)

    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        _render_extra_install_hint(c, "hf", "HF Jobs support is not installed.", exc)
        raise typer.Exit(1)

    api = HfApi()
    try:
        j = api.inspect_job(job_id=remote_job_id)
    except Exception as exc:
        c.error(f"Inspecting job {remote_job_id}: {exc}")
        raise typer.Exit(1)

    status_value = str(j.status.stage)
    if local_run is not None:
        from carl_studio.db import LocalDB

        result = _safe_json_object(local_run.get("result"))
        result.update(
            {
                "hub_job_id": remote_job_id,
                "remote_status": status_value,
                "remote_message": getattr(j.status, "message", "") or "",
            }
        )
        updates: dict[str, Any] = {
            "status": _normalize_remote_status(status_value),
            "remote_id": remote_job_id,
            "result": result,
        }
        if status_value in {"completed", "failed", "error", "canceled"}:
            updates["completed_at"] = _now_iso()
        LocalDB().update_run(local_run["id"], updates)

    pairs = []
    if local_run is not None:
        pairs.append(("Run", str(local_run["id"])))
    pairs.extend([("Job", remote_job_id), ("Status", status_value)])
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
@app.command(hidden=True)
def logs(
    run_id: str = typer.Argument(..., help="Local run ID or HF Jobs run/job ID"),
    tail: int = typer.Option(50, "--tail", "-n", help="Number of log entries to show"),
) -> None:
    """Stream logs from a training run."""
    c = get_console()
    local_run, remote_job_id = _resolve_run_reference(run_id)

    if local_run is not None and not remote_job_id:
        _render_local_run_record(c, local_run, title="Local Run")
        c.info("No remote logs available yet for this run.")
        raise typer.Exit(0)

    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        _render_extra_install_hint(c, "hf", "HF Jobs support is not installed.", exc)
        raise typer.Exit(1)

    api = HfApi()
    try:
        entries = list(api.fetch_job_logs(job_id=remote_job_id))
    except Exception as exc:
        c.error(f"Fetching logs for {remote_job_id}: {exc}")
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
@app.command(hidden=True)
def stop(
    run_id: str = typer.Argument(..., help="Local run ID or HF Jobs run/job ID to cancel"),
) -> None:
    """Cancel a training run."""
    c = get_console()
    local_run, remote_job_id = _resolve_run_reference(run_id)

    if local_run is not None and not remote_job_id:
        _render_local_run_record(c, local_run, title="Local Run")
        c.info("This run has no remote job ID to cancel.")
        raise typer.Exit(0)

    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        _render_extra_install_hint(c, "hf", "HF Jobs support is not installed.", exc)
        raise typer.Exit(1)

    api = HfApi()
    try:
        api.cancel_job(job_id=remote_job_id)
    except Exception as exc:
        c.error(f"Cancelling job {remote_job_id}: {exc}")
        raise typer.Exit(1)

    if local_run is not None:
        from carl_studio.db import LocalDB

        result = _safe_json_object(local_run.get("result"))
        result.update({"hub_job_id": remote_job_id, "remote_status": "canceled"})
        LocalDB().update_run(
            local_run["id"],
            {
                "status": "canceled",
                "remote_id": remote_job_id,
                "completed_at": _now_iso(),
                "result": result,
            },
        )

    c.ok(f"Job {remote_job_id} cancelled.")

