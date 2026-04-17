"""Textual TUI app for ``carl observe --live``.

Three-panel layout:
  Left:         Phi trajectory sparkline (updates per step)
  Right-top:    Reward breakdown + loss table (live-updating)
  Right-bottom: Latest completion samples (scrollable log)

Phase transition markers appear as highlighted rows when |delta_Phi| > 0.03.

Requires ``carl-studio[tui]`` (textual >= 1.0, textual-plotext >= 0.3).

Launch:
    carl observe --live --source file --path training_log.jsonl
    carl observe --live --source trackio --run my-run
    textual serve carl_studio.observe.app:ObserveApp  # browser mode
"""

from __future__ import annotations

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.widgets import Footer, Header, RichLog, Static

from carl_studio.observe.data_source import FileSource, ObserveFrame, TrackioSource
from carl_studio.theme import load_theme


class PhiChart(Static):
    """ASCII sparkline of Phi trajectory + per-token field visualization."""

    SPARK_CHARS = " ▁▂▃▄▅▆▇█"
    values: reactive[list[float]] = reactive(list, recompose=False)

    def __init__(self, max_points: int = 60, **kwargs) -> None:
        super().__init__(**kwargs)
        self._max = max_points
        self._values: list[float] = []
        self._last_phi_field: str = ""
        self._last_entropy_field: str = ""
        self._last_defect_map: str = ""
        self._last_trace_tokens: int = 0

    def push(
        self, phi: float, transition: bool = False, frame: "ObserveFrame | None" = None
    ) -> None:
        self._values.append(phi)
        if len(self._values) > self._max:
            self._values = self._values[-self._max :]

        # Update trace-level fields if available
        if frame is not None and frame.phi_sparkline:
            self._last_phi_field = frame.phi_sparkline
            self._last_entropy_field = frame.entropy_sparkline
            self._last_defect_map = frame.defect_map
            self._last_trace_tokens = frame.trace_n_tokens

        self._render_spark()

    def _render_spark(self) -> None:
        if not self._values:
            self.update("  Phi: waiting for data...")
            return

        lo = min(self._values)
        hi = max(self._values)
        span = hi - lo if hi > lo else 1.0
        chars = self.SPARK_CHARS

        spark = ""
        for v in self._values:
            idx = int((v - lo) / span * (len(chars) - 1))
            spark += chars[idx]

        current = self._values[-1]
        delta = self._values[-1] - self._values[-2] if len(self._values) > 1 else 0.0
        sign = "+" if delta >= 0 else ""

        header = f"  Phi: {current:.4f} ({sign}{delta:.4f})  steps: {len(self._values)}"
        scale = f"  [{lo:.3f} {'─' * 20} {hi:.3f}]"

        lines = [header, f"  {spark}", scale]

        # Per-token field visualization (from trace)
        if self._last_phi_field:
            lines.append("")
            lines.append(f"  Phi field ({self._last_trace_tokens} tokens):")
            lines.append(f"  {self._last_phi_field}")
        if self._last_entropy_field:
            lines.append(f"  Entropy:  {self._last_entropy_field}")
        if self._last_defect_map:
            lines.append(f"  Defects:  {self._last_defect_map}")

        self.update("\n".join(lines))


class MetricsTable(Static):
    """Live-updating metrics display."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._last_frame: ObserveFrame | None = None

    def push(self, frame: ObserveFrame) -> None:
        self._last_frame = frame
        self._render()

    def _render(self) -> None:
        f = self._last_frame
        if f is None:
            self.update("  Metrics: waiting...")
            return

        lines = [
            f"  Step: {f.step}",
            f"  Loss: {f.loss:.4f}",
            f"  Reward: {f.reward_mean:.4f}",
            f"  Phi: {f.phi:.4f}",
        ]

        if f.rewards:
            lines.append("  ────────────────────")
            for name, val in sorted(f.rewards.items()):
                short = name.replace("reward_", "")
                lines.append(f"  {short:.<20s} {val:.4f}")

        # Trace-level details
        if f.trace_n_tokens > 0:
            lines.append("  ────────────────────")
            lines.append(f"  Tokens: {f.trace_n_tokens}")
            lines.append(f"  CARL:   {f.trace_carl_reward:.4f}")
            lines.append(f"  Cryst:  {f.trace_crystallizations}  Melt: {f.trace_meltings}")

        if f.phase_transition:
            lines.append("")
            lines.append("  ◈ PHASE TRANSITION")

        self.update("\n".join(lines))


class ObserveApp(App):
    """Camp CARL live training dashboard."""

    CSS = """
    Screen {
        layout: horizontal;
    }
    #left {
        width: 60%;
        border: round $accent;
        padding: 1;
    }
    #right {
        width: 40%;
        layout: vertical;
    }
    #metrics {
        height: 50%;
        border: round $accent;
        padding: 1;
    }
    #log {
        height: 50%;
        border: round $accent;
        padding: 1;
    }
    PhiChart {
        height: 100%;
    }
    MetricsTable {
        height: 100%;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("c", "clear_log", "Clear log"),
    ]

    def __init__(
        self,
        source_type: str = "file",
        source_path: str = "",
        source_space: str = "",
        source_project: str = "",
        source_run: str = "",
        poll_interval: float = 2.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._source_type = source_type
        self._source_path = source_path
        self._source_space = source_space
        self._source_project = source_project
        self._source_run = source_run
        self._poll_interval = poll_interval

        # Build data source
        if source_type == "trackio":
            self._source = TrackioSource(space=source_space, project=source_project, run=source_run)
        else:
            self._source = FileSource(source_path)

    def compose(self) -> ComposeResult:
        theme = load_theme()
        yield Header(show_clock=True)
        with Horizontal():
            with Vertical(id="left"):
                yield PhiChart(id="phi-chart")
            with Vertical(id="right"):
                yield MetricsTable(id="metrics")
                yield RichLog(id="log", highlight=True, markup=True)
        yield Footer()

    def on_mount(self) -> None:
        self.title = "Camp CARL Observer"
        self.sub_title = f"source: {self._source_type}"
        self.set_interval(self._poll_interval, self._poll_source)

    def _poll_source(self) -> None:
        phi_chart = self.query_one("#phi-chart", PhiChart)
        metrics = self.query_one("#metrics", MetricsTable)
        log = self.query_one("#log", RichLog)

        try:
            frames = self._source.poll()
        except Exception as exc:
            log.write(f"[bold red]Trackio error:[/] {exc}")
            return

        for frame in frames:
            phi_chart.push(frame.phi, frame.phase_transition, frame=frame)
            metrics.push(frame)

            if frame.completion_sample:
                log.write(f"[dim]step {frame.step}:[/] {frame.completion_sample[:200]}")

            if frame.phase_transition:
                log.write("[bold red]◈ PHASE TRANSITION DETECTED[/]")

    def action_clear_log(self) -> None:
        self.query_one("#log", RichLog).clear()


def run_app(
    source: str = "file",
    path: str = "",
    space: str = "",
    project: str = "",
    run: str = "",
    poll: float = 2.0,
) -> None:
    """Entry point for ``carl observe --live``."""
    app = ObserveApp(
        source_type=source,
        source_path=path,
        source_space=space,
        source_project=project,
        source_run=run,
        poll_interval=poll,
    )
    app.run()
