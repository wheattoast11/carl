"""carl observe — Live training monitor TUI.

3 views, TI-89 density, information IS structure.

Views:
  1. PULSE  — Single-screen health. Everything you need in one glance.
              Phase gauge (ASCII), task sparkline, tool sparkline,
              health signal, Carl's Take (1 sentence).
  2. REWARD — Reward composition over time. Stacked area chart.
              Phase-colored background. Annotation at transitions.
  3. TRACE  — Crystal analytics. Φ trajectory, τ, R, defect events.

Navigation:
  1/2/3     — Switch view
  r         — Refresh data
  q / Esc   — Quit
  Ctrl+K    — Command palette
  ?         — Help overlay

Usage:
  carl observe --job JOB_ID
  carl observe --job JOB_ID --refresh 30
  carl observe --compare JOB_ID_1 JOB_ID_2
"""
from __future__ import annotations

from dataclasses import dataclass

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container
from textual.reactive import reactive
from textual.widgets import Label, Static
from textual_plotext import PlotextPlot


# ═══════════════════════════════════════════════════════════════
# DATA
# ═══════════════════════════════════════════════════════════════

@dataclass
class StepData:
    step: int
    task: float = 0.0
    tools: float = 0.0
    length: float = 0.0
    tau: float = 0.5
    R: float = 0.0
    persist: float = 0.0
    err_util: float = 0.0
    explore: float = 0.0
    diversity: float = 0.0
    carl: float = 0.0
    zero_std: float = 0.0
    loss: float = 0.0
    lr: float = 0.0
    phi: float = 0.0
    reward: float = 0.0
    step_time: float = 0.0


def parse_steps_from_logs(logs: list[str]) -> list[StepData]:
    """Parse StepData from HF job log lines."""
    import ast
    steps = []
    n = 0
    for line in logs:
        text = line.strip() if hasattr(line, 'strip') else str(line)
        if "'loss'" not in text:
            continue
        start = text.find("{")
        if start < 0:
            continue
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{": depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        d = ast.literal_eval(text[start:i+1])
                        n += 1
                        steps.append(StepData(
                            step=n,
                            task=float(d.get("rewards/task_completion_reward/mean", 0)),
                            tools=float(d.get("tools/call_frequency", 0)),
                            length=float(d.get("completions/mean_length", 0)),
                            tau=float(d.get("witness/tau", 0.5)),
                            R=float(d.get("witness/R", 0)),
                            persist=float(d.get("rewards/persistence_reward/mean", 0)),
                            err_util=float(d.get("rewards/error_utilization_reward/mean", 0)),
                            explore=float(d.get("rewards/exploration_reward/mean", 0)),
                            diversity=float(d.get("rewards/diversity_reward/mean", 0)),
                            carl=float(d.get("rewards/gated_carl_reward/mean", 0)),
                            zero_std=float(d.get("frac_reward_zero_std", 0)),
                            loss=float(d.get("loss", 0)),
                            lr=float(d.get("learning_rate", 0)),
                            phi=float(d.get("crystal/phi_mean", 0)),
                            reward=float(d.get("reward", 0)),
                            step_time=float(d.get("step_time", 0)),
                        ))
                    except Exception:
                        pass
                    break
    return steps


def fetch_steps(job_id: str) -> list[StepData]:
    """Fetch and parse steps from HF Jobs."""
    from huggingface_hub import HfApi, get_token
    api = HfApi(token=get_token())
    logs = list(api.fetch_job_logs(job_id=job_id))
    return parse_steps_from_logs(logs)


# ═══════════════════════════════════════════════════════════════
# ASCII PRIMITIVES — TI-89 density
# ═══════════════════════════════════════════════════════════════

SPARK_CHARS = "▁▂▃▄▅▆▇█"

def sparkline(values: list[float], width: int = 40) -> str:
    """Render a sparkline from values. Fixed width, sampled if needed."""
    if not values:
        return "─" * width
    # Sample to fit width
    if len(values) > width:
        step = len(values) / width
        sampled = [values[int(i * step)] for i in range(width)]
    else:
        sampled = values
    lo, hi = min(sampled), max(sampled)
    rng = hi - lo if hi > lo else 1.0
    return "".join(SPARK_CHARS[min(7, int((v - lo) / rng * 7.99))] for v in sampled)


def phase_bar(tau: float, width: int = 30) -> str:
    """ASCII phase gauge. Purple=gaseous, blue=fluid, green=crystalline."""
    # Zones: [0, 0.3) crystalline, [0.3, 0.7) fluid, [0.7, 1.0] gaseous
    needle_pos = int(tau * (width - 1))
    bar = []
    for i in range(width):
        frac = i / (width - 1)
        if i == needle_pos:
            bar.append("▼")
        elif frac < 0.3:
            bar.append("█")  # crystalline zone
        elif frac < 0.7:
            bar.append("▓")  # fluid zone
        else:
            bar.append("░")  # gaseous zone
    phase = "CRYSTAL" if tau < 0.3 else "FLUID" if tau < 0.7 else "GAS"
    return f"[{''.join(bar)}] {phase} τ={tau:.3f}"


def health_icon(zero_std: float, task: float) -> str:
    """Single character health signal."""
    if zero_std >= 1.0:
        return "✕ DEAD"
    if task >= 0.8:
        return "◆ HEALTHY"
    if task >= 0.5:
        return "▲ LEARNING"
    return "○ EXPLORING"


def pct_bar(value: float, width: int = 15, label: str = "") -> str:
    """Compact percentage bar."""
    filled = int(value * width)
    bar = "█" * filled + "░" * (width - filled)
    return f"{label}{bar} {value:.0%}"


# ═══════════════════════════════════════════════════════════════
# WIDGETS
# ═══════════════════════════════════════════════════════════════

class PulseView(Static):
    """View 1: Single-screen health. Everything in one glance."""

    def render_pulse(self, steps: list[StepData], name: str = "") -> str:
        if not steps:
            return "No data yet. Waiting for first step..."

        s = steps[-1]  # latest
        n = len(steps)

        # Trend data
        tasks = [x.task for x in steps]
        tools = [x.tools for x in steps]
        lengths = [x.length for x in steps]
        taus = [x.tau for x in steps]

        # Carl's Take (1 sentence)
        take = ""
        if n >= 5:
            t_early = sum(tasks[:5]) / 5
            t_late = sum(tasks[-5:]) / 5
            tool_early = sum(tools[:5]) / 5
            tool_late = sum(tools[-5:]) / 5
            if t_late > t_early * 1.2:
                take = f"Task improving ({t_early:.0%}→{t_late:.0%})"
            elif tool_late < tool_early * 0.7:
                take = f"Compressing: {tool_early:.1f}→{tool_late:.1f} tools/step"
            elif s.tau < 0.3:
                take = f"Crystallized at step ~{n-5}. R={s.R:.3f}"
            else:
                take = f"Learning. τ={s.tau:.3f}, {n} steps"

        title = f"═══ {name or 'Training'} ═══ Step {n} ═══ {health_icon(s.zero_std, s.task)} ═══"

        lines = [
            title,
            "",
            phase_bar(s.tau),
            "",
            f"TASK  {pct_bar(s.task, 20)}  {sparkline(tasks, 30)}",
            f"TOOLS {s.tools:5.1f}/step          {sparkline(tools, 30)}",
            f"LEN   {s.length:5.0f} tok           {sparkline(lengths, 30)}",
            f"TAU   {s.tau:5.3f}               {sparkline(taus, 30)}",
            "",
            f"PERSIST {s.persist:.3f}  ERR_UTIL {s.err_util:.3f}  EXPLORE {s.explore:.3f}  DIV {s.diversity:.3f}",
            f"CARL    {s.carl:.3f}  REWARD   {s.reward:.3f}  ZERO_STD {s.zero_std:.2f}  LOSS {s.loss:.4f}",
            "",
            f"▸ {take}" if take else "",
        ]
        return "\n".join(lines)


class RewardView(Static):
    """View 2: Reward composition chart."""
    pass


class TraceView(Static):
    """View 3: Crystal analytics — Φ, τ, R."""
    pass


# ═══════════════════════════════════════════════════════════════
# APP
# ═══════════════════════════════════════════════════════════════

class CarlMonitor(App):
    """carl observe — Live training monitor."""

    CSS = """
    Screen {
        background: #0a0a0a;
    }
    #header-bar {
        dock: top;
        height: 1;
        background: #1a1a1a;
        color: #00ffff;
    }
    #pulse {
        padding: 1 2;
        color: #e0e0e0;
    }
    #reward-chart {
        height: 100%;
        min-height: 20;
    }
    #trace-chart {
        height: 100%;
        min-height: 20;
    }
    .view-container {
        height: 1fr;
    }
    #footer-info {
        dock: bottom;
        height: 1;
        background: #1a1a1a;
        color: rgba(255,255,255,0.5);
    }
    """

    BINDINGS = [
        Binding("1", "view_pulse", "Pulse", priority=True),
        Binding("2", "view_reward", "Rewards", priority=True),
        Binding("3", "view_trace", "Trace", priority=True),
        Binding("r", "refresh", "Refresh", priority=True),
        Binding("q", "quit", "Quit", priority=True),
        Binding("escape", "quit", "Quit", show=False),
    ]

    current_view = reactive(1)
    steps: list[StepData] = []
    job_id: str = ""
    run_name: str = ""

    def __init__(self, job_id: str = "", run_name: str = "", **kwargs):
        super().__init__(**kwargs)
        self.job_id = job_id
        self.run_name = run_name or job_id[:12]

    def compose(self) -> ComposeResult:
        yield Label(f" carl observe — {self.run_name} ", id="header-bar")
        yield Container(
            PulseView(id="pulse"),
            PlotextPlot(id="reward-chart"),
            PlotextPlot(id="trace-chart"),
            id="main",
            classes="view-container",
        )
        yield Label(" [1] Pulse  [2] Rewards  [3] Trace  [r] Refresh  [q] Quit ", id="footer-info")

    def on_mount(self) -> None:
        self._show_view(1)
        if self.job_id:
            self.load_data()
            self.set_interval(30.0, self.load_data)

    @work(thread=True)
    def load_data(self) -> None:
        """Fetch latest data from HF Jobs."""
        if not self.job_id:
            return
        try:
            new_steps = fetch_steps(self.job_id)
            self.call_from_thread(self._update_data, new_steps)
        except Exception as e:
            self.call_from_thread(self.notify, f"Fetch error: {e}", severity="error")

    def _update_data(self, new_steps: list[StepData]) -> None:
        self.steps = new_steps
        self._render_current_view()

    def _show_view(self, n: int) -> None:
        self.current_view = n
        pulse = self.query_one("#pulse")
        reward = self.query_one("#reward-chart")
        trace = self.query_one("#trace-chart")

        pulse.display = n == 1
        reward.display = n == 2
        trace.display = n == 3

        self._render_current_view()

    def _render_current_view(self) -> None:
        if self.current_view == 1:
            self._render_pulse()
        elif self.current_view == 2:
            self._render_rewards()
        elif self.current_view == 3:
            self._render_trace()

    def _render_pulse(self) -> None:
        pulse = self.query_one("#pulse", PulseView)
        pulse.update(pulse.render_pulse(self.steps, self.run_name))

    def _render_rewards(self) -> None:
        if not self.steps:
            return
        chart = self.query_one("#reward-chart", PlotextPlot)
        plt = chart.plt
        plt.clear_data()
        plt.clear_figure()
        plt.theme("dark")
        plt.plotsize(None, None)

        x = [s.step for s in self.steps]
        plt.stacked_bar(x,
            [[s.task * 3.0 for s in self.steps],
             [s.tools * 0.1 for s in self.steps],  # scaled for visibility
             [s.carl for s in self.steps],
             [s.persist * 1.5 for s in self.steps],
             [s.explore * 0.75 for s in self.steps]],
            label=["task×3", "engage", "carl", "persist", "explore"],
            color=["cyan", "green", "magenta", "yellow", "blue"],
        )
        plt.title("Reward Composition")
        plt.xlabel("Step")
        chart.refresh()

    def _render_trace(self) -> None:
        if not self.steps:
            return
        chart = self.query_one("#trace-chart", PlotextPlot)
        plt = chart.plt
        plt.clear_data()
        plt.clear_figure()
        plt.theme("dark")
        plt.plotsize(None, None)

        x = [s.step for s in self.steps]
        plt.plot(x, [s.tau for s in self.steps], label="τ", color="cyan")
        plt.plot(x, [s.R for s in self.steps], label="R", color="green")
        plt.plot(x, [s.phi for s in self.steps], label="Φ", color="magenta")

        # Phase background markers
        plt.hline(0.3, color="white")
        plt.hline(0.7, color="white")
        plt.title("Phase Dynamics — τ, R, Φ")
        plt.xlabel("Step")
        chart.refresh()

    def action_view_pulse(self) -> None:
        self._show_view(1)

    def action_view_reward(self) -> None:
        self._show_view(2)

    def action_view_trace(self) -> None:
        self._show_view(3)

    def action_refresh(self) -> None:
        self.notify("Refreshing...", timeout=2)
        self.load_data()


# ═══════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description="carl observe — Live training monitor")
    parser.add_argument("--job", required=True, help="HuggingFace Job ID to monitor")
    parser.add_argument("--name", default="", help="Display name for the run")
    parser.add_argument("--refresh", type=int, default=30, help="Refresh interval in seconds")
    args = parser.parse_args()

    app = CarlMonitor(job_id=args.job, run_name=args.name or args.job[:12])
    app.run()


if __name__ == "__main__":
    main()
