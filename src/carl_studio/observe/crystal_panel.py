from __future__ import annotations
from collections import deque
from carl_studio.observe.sparkline import sparkline, phase_char
from carl_studio.observe.ansi import color_rgb, bold, bar


class CrystalPanel:
    """Full-screen overlay showing crystallization dynamics.
    Renders as a single string with ANSI codes. No curses needed."""

    def __init__(self) -> None:
        self.phi_history: list[float] = []
        self.r_history: list[float] = []
        self.tau_history: list[float] = []
        self.reward_values: dict[str, float] = {}
        self.sematons: deque[dict] = deque(maxlen=10)
        self._gate_fired: bool = False
        self._step: int = 0

    def update(self, step_data: dict) -> None:
        """Ingest one training step's data.

        Args:
            step_data: Dict with optional keys: step, phi_mean, kuramoto_r,
                task_completion, tool_engagement, gated_carl, tool_format,
                gr3_length, diversity, sematon, gate_fired.
        """
        self._step = step_data.get("step", self._step + 1)
        if "phi_mean" in step_data:
            self.phi_history.append(step_data["phi_mean"])
        if "kuramoto_r" in step_data:
            self.r_history.append(step_data["kuramoto_r"])
            self.tau_history.append(1.0 - step_data["kuramoto_r"])
        for key in (
            "task_completion", "tool_engagement", "gated_carl",
            "tool_format", "gr3_length", "diversity",
        ):
            if key in step_data:
                self.reward_values[key] = step_data[key]
        if "sematon" in step_data:
            self.sematons.append(step_data["sematon"])
        if "gate_fired" in step_data:
            self._gate_fired = step_data["gate_fired"]

    def render(self, cols: int = 80) -> str:
        """Render the panel as a single string with ANSI codes.

        Args:
            cols: Terminal width in columns.
        """
        lines: list[str] = []
        width = min(cols - 10, 60)

        # Header
        lines.append(bold(color_rgb("CARL Crystal Observatory", 0, 255, 200)))
        lines.append("")

        # Sparklines
        lines.append(f"  {color_rgb('Phi', 100, 200, 255)}: {sparkline(self.phi_history, width=width)}")
        lines.append(f"    {color_rgb('R', 200, 100, 255)}: {sparkline(self.r_history, width=width)}")
        lines.append(f"  {color_rgb('tau', 255, 200, 100)}: {sparkline(self.tau_history, width=width)}")
        lines.append("")

        # Reward bars
        lines.append(bold("  Reward Decomposition"))
        for name, value in sorted(self.reward_values.items()):
            lines.append(f"    {name:>20s} {bar(value)} {value:.3f}")
        lines.append("")

        # Sematon table
        lines.append(bold(color_rgb("  Sematon Witness Table", 0, 200, 255)))
        for s in list(self.sematons)[-5:]:
            c_flag = color_rgb("+", 0, 255, 100) if s.get("c") else color_rgb("-", 255, 100, 100)
            r_val = s.get("W", {}).get("R", 0)
            h_val = s.get("H", 0)
            lines.append(f"    [{c_flag}] step={s.get('A', '-'):>4} R={r_val:.3f} H={h_val:.3f}")
        lines.append("")

        # Status line
        tau = self.tau_history[-1] if self.tau_history else 0.5
        phase = phase_char(tau)
        gate_str = color_rgb("FIRED", 0, 255, 100) if self._gate_fired else color_rgb("PENDING", 255, 200, 0)
        lines.append(f"  Phase: {bold(phase)}  |  Gate: {gate_str}  |  Step: {self._step}")

        return "\n".join(lines)
