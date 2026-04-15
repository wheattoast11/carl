"""Observe command and coherence report rendering."""

from __future__ import annotations

import typer

from carl_studio.console import CampConsole, get_console

from .apps import app
from .shared import _print_observe_usage, _render_extra_install_hint

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
    space: str,
    project: str,
    run_name: str,
) -> tuple[list, str]:
    """Load ObserveFrames from the specified source. Returns (frames, source_desc)."""
    from carl_studio.observe.data_source import FileSource, TrackioSource, normalize_trackio_space

    if file:
        src = FileSource(file)
        frames = src.poll()
        return frames, f"file: {file}"

    resolved_space = normalize_trackio_space(url or space)
    if url or source == "trackio":
        src = TrackioSource(space=resolved_space, project=project, run=run_name)
        frames = src.poll()
        source_desc = f"trackio: {resolved_space}"
        if src.resolved_project:
            source_desc += f"  |  project: {src.resolved_project}"
        if src.resolved_run:
            source_desc += f"  |  run: {src.resolved_run}"
        return frames, source_desc

    return [], "no source"


def _render_observe_report(c: "CampConsole", frames: list, source_desc: str) -> dict:
    """Render the rich one-shot observe report. Returns computed metrics dict."""
    from carl_studio.primitives.constants import KAPPA, SIGMA

    if not frames:
        c.blank()
        c.header("CARL Observer")
        c.warn("No data found.")
        source_parts = source_desc.split("  |  ")
        if source_parts:
            c.info(f"Source: {source_parts[0]}")
            for part in source_parts[1:]:
                c.info(part)
        else:
            c.info(f"Source: {source_desc}")
        _print_observe_usage(c)
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
        # Use phi variance as entropy proxy
        entropy_mean_val = phi_std
        entropy_std_val = (
            sum((abs(phis[i] - phis[i - 1]) - phi_std) ** 2 for i in range(1, len(phis)))
            / (len(phis) - 1)
        ) ** 0.5
    else:
        entropy_mean_val = 0.0
        entropy_std_val = 0.0

    # Phase state
    phase = _phase_state(phis, defect_densities)

    # Cloud quality from rewards
    cloud_vals = [
        f.rewards.get("reward_carl", f.rewards.get("reward_cloud", 0.0))
        for f in frames
        if f.rewards
    ]
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
        phi_mean,
        phi_trend,
        entropy_std_val,
        phase,
        frac_zero_reward,
        lyapunov_proxy,
    )

    # ---- Render ----
    c.blank()
    c.header("CARL Observer")
    c.print(
        f"  [camp.muted]Source: {source_desc}  |  {len(frames)} steps  |  "
        f"range: {steps[0]}-{steps[-1]}[/]"
    )
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
    table.add_row(
        "Cloud quality", f"{cloud_mean:.4f}", _trend_arrow(cloud_vals) if cloud_vals else "~"
    )
    c.print(table)
    c.blank()

    # Phase and dynamics
    phase_table = c.make_table("Signal", "Value", title="Dynamics")
    phase_table.add_row("Phase state", phase)
    phase_table.add_row("Discontinuity events", f"{discontinuity_events} / {len(frames) - 1} steps")
    phase_table.add_row("Lyapunov proxy", f"{lyapunov_proxy:.4f}")
    phase_table.add_row(
        "Conservation (kappa*sigma)",
        f"{conservation_product:.4f} (error: {conservation_error:.2e})",
    )
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
    if losses and any(loss_value > 0 for loss_value in losses):
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

    try:
        import anthropic  # noqa: F401
    except ImportError as exc:
        _render_extra_install_hint(
            c,
            "observe",
            "Claude-powered diagnosis dependencies are not installed.",
            exc,
            quoted=True,
        )
        c.blank()
        return

    if not frames:
        c.warn("No data to diagnose.")
        return

    from carl_studio.primitives import CoherenceObserver, CoherenceSnapshot

    c.blank()
    c.print("  [camp.primary]Claude-powered analysis[/]  (--diagnose)")
    c.rule()

    # Build synthetic CoherenceSnapshots from ObserveFrames for the observer
    snapshots: list[CoherenceSnapshot] = []
    for f in frames:
        snap = CoherenceSnapshot(
            step=f.step,
            n_tokens=f.trace_n_tokens
            if hasattr(f, "trace_n_tokens") and f.trace_n_tokens > 0
            else 512,
            phi_mean=f.phi,
            phi_std=0.0,
            phi_trajectory=[f.phi],
            n_defects=0,
            n_crystallizations=f.trace_crystallizations
            if hasattr(f, "trace_crystallizations")
            else 0,
            n_meltings=f.trace_meltings if hasattr(f, "trace_meltings") else 0,
            defect_density=0.0,
            cloud_quality_mean=f.rewards.get("reward_carl", f.rewards.get("reward_cloud", 0.0))
            if f.rewards
            else 0.0,
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
        observer._buffer = observer._buffer[-observer.window_size :]

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
    url: str | None = typer.Option(
        None, "--url", "-u", help="Trackio space URL (e.g. https://wheattoast11-trackio.hf.space/)"
    ),
    file: str | None = typer.Option(None, "--file", "-f", help="Local JSONL log file path"),
    live: bool = typer.Option(False, "--live", "-l", help="Launch real-time Textual TUI dashboard"),
    source: str = typer.Option(
        "auto", "--source", "-s", help="Data source: trackio, file, or auto"
    ),
    diagnose: bool = typer.Option(
        False,
        "--diagnose",
        "-d",
        help="Enable Claude-powered analysis (requires ANTHROPIC_API_KEY)",
    ),
    api_key: str | None = typer.Option(
        None, "--api-key", envvar="ANTHROPIC_API_KEY", help="Anthropic API key for --diagnose"
    ),
    poll: float | None = typer.Option(None, "--poll", help="Poll interval in seconds (for --live)"),
    project: str = typer.Option(
        "", "--project", help="Trackio project name (auto-detected when only one exists)"
    ),
    run_name: str = typer.Option("", "--run", help="Trackio run name"),
    space: str = typer.Option(
        "wheattoast11-trackio", "--space", help="Trackio space name (if not using --url)"
    ),
) -> None:
    """Observe training coherence dynamics. Rich defaults, zero config.

    \b
    One-shot (default):
      carl observe --url https://wheattoast11-trackio.hf.space/ --run my-run
      carl observe --file logs/train.jsonl

    \b
    Live TUI:
      carl observe --live --url https://wheattoast11-trackio.hf.space/ --run my-run

    \b
    Claude-powered analysis:
      carl observe --diagnose --url https://wheattoast11-trackio.hf.space/ --run my-run
    """
    c = get_console()
    from carl_studio.settings import CARLSettings

    settings = CARLSettings.load()
    effective_url = url or (settings.trackio_url if not file else None)
    poll_interval = poll if poll is not None else settings.observe_defaults.default_poll_interval

    if effective_url and file:
        c.error("Use either --url or --file, not both.")
        raise typer.Exit(1)

    # Resolve source type
    resolved_source = source
    if resolved_source == "auto":
        if settings.observe_defaults.default_source == "trackio":
            resolved_source = "trackio"
        if file:
            resolved_source = "file"
        elif effective_url:
            resolved_source = "trackio"
        elif resolved_source == "auto":
            resolved_source = "trackio"

    # ---- Live TUI mode ----
    if live:
        try:
            from carl_studio.observe.app import run_app
        except ImportError as exc:
            _render_extra_install_hint(c, "tui", "Live dashboard support is not installed.", exc)
            raise typer.Exit(1)

        # Resolve path/space from --url or --file for the TUI
        tui_source = resolved_source
        tui_path = file or ""
        tui_space = space

        if resolved_source == "trackio":
            from carl_studio.observe.data_source import TrackioError, normalize_trackio_space

            try:
                tui_space = normalize_trackio_space(effective_url or space)
            except TrackioError as exc:
                c.error(str(exc))
                _print_observe_usage(c)
                raise typer.Exit(1)

        run_app(
            source=tui_source,
            path=tui_path,
            space=tui_space,
            project=project,
            run=run_name,
            poll=poll_interval,
        )
        raise typer.Exit(0)

    # ---- One-shot rich report ----
    try:
        frames, source_desc = _load_frames(
            url=effective_url,
            file=file,
            source=resolved_source,
            space=space,
            project=project,
            run_name=run_name,
        )
    except Exception as exc:
        c.blank()
        c.header("CARL Observer")
        c.error(str(exc))
        _print_observe_usage(c)
        raise typer.Exit(1)
    _render_observe_report(c, frames, source_desc)

    # ---- Optional Claude diagnosis ----
    if diagnose:
        _render_diagnose(c, frames, api_key)


