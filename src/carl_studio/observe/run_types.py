"""Type-safe data model for training run comparison and monitoring.

The mise en place — every UI component, CLI tool, and dashboard consumes these types.
Works for both HTML comparison dashboards and CLI TUI rendering.

Types:
    StepMetrics     — All observable signals from one training step
    PhaseAnnotation — Phase classification at a step (gaseous/fluid/crystalline)
    TrainingRun     — Complete run: metadata + ordered step metrics
    RunComparison   — Two or more runs aligned by step for comparison
    MetricGroup     — Named group of related metrics for chart layout
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Phase(str, Enum):
    """Training phase derived from τ (entropy disorder)."""
    GASEOUS = "gaseous"        # τ > 0.7 — exploring
    FLUID = "fluid"            # 0.3 ≤ τ ≤ 0.7 — learning
    CRYSTALLINE = "crystalline" # τ < 0.3 — converged
    UNKNOWN = "unknown"

    @classmethod
    def from_tau(cls, tau: float) -> Phase:
        if tau > 0.7:
            return cls.GASEOUS
        elif tau < 0.3:
            return cls.CRYSTALLINE
        else:
            return cls.FLUID


@dataclass
class StepMetrics:
    """All observable signals from one training step."""
    step: int
    # Task performance
    task_completion: float = 0.0
    tool_engagement: float = 0.0
    tool_format: float = 0.0
    tools_call_frequency: float = 0.0
    tools_failure_frequency: float = 0.0
    # Persistence (v14+)
    persistence: float = 0.0
    error_utilization: float = 0.0
    exploration: float = 0.0
    # Coherence
    gated_carl: float = 0.0
    diversity: float = 0.0
    adaptive_gr3: float = 0.0
    # Crystal analytics
    phi_mean: float = 0.0
    phi_std: float = 0.0
    entropy_mean: float = 0.0
    cloud_quality: float = 0.0
    defect_density: float = 0.0
    lyapunov_proxy: float = 0.0
    # Witness
    R: float = 0.0
    tau: float = 0.5
    converged: bool = False
    constructive: bool = False
    # Training dynamics
    loss: float = 0.0
    learning_rate: float = 0.0
    grad_norm: float = 0.0
    mean_length: float = 0.0
    reward: float = 0.0
    reward_std: float = 0.0
    frac_zero_std: float = 0.0
    step_time: float = 0.0
    # Dynamic weights (v14 reviewed+)
    dyn_phase: str = ""
    dyn_tau: float = 0.0
    dyn_tau_dot: float = 0.0
    # Cascade
    cascade_stage: int = 0
    # Raw dict for extensibility
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def phase(self) -> Phase:
        return Phase.from_tau(self.tau)

    @classmethod
    def from_log_dict(cls, step: int, d: dict[str, Any]) -> StepMetrics:
        """Parse from a TRL training log dict."""
        return cls(
            step=step,
            task_completion=float(d.get("rewards/task_completion_reward/mean", 0)),
            tool_engagement=float(d.get("rewards/tool_engagement_reward/mean", 0)),
            tool_format=float(d.get("rewards/tool_format_reward/mean", 0)),
            tools_call_frequency=float(d.get("tools/call_frequency", 0)),
            tools_failure_frequency=float(d.get("tools/failure_frequency", 0)),
            persistence=float(d.get("rewards/persistence_reward/mean", 0)),
            error_utilization=float(d.get("rewards/error_utilization_reward/mean", 0)),
            exploration=float(d.get("rewards/exploration_reward/mean", 0)),
            gated_carl=float(d.get("rewards/gated_carl_reward/mean", 0)),
            diversity=float(d.get("rewards/diversity_reward/mean", 0)),
            adaptive_gr3=float(d.get("rewards/adaptive_gr3_length_penalty/mean", 0)),
            phi_mean=float(d.get("crystal/phi_mean", 0)),
            phi_std=float(d.get("crystal/phi_std", 0)),
            entropy_mean=float(d.get("crystal/entropy_mean", 0)),
            cloud_quality=float(d.get("crystal/cloud_quality", 0)),
            defect_density=float(d.get("crystal/defect_density", 0)),
            lyapunov_proxy=float(d.get("crystal/lyapunov_proxy", 0)),
            R=float(d.get("witness/R", 0)),
            tau=float(d.get("witness/tau", 0.5)),
            converged=bool(d.get("witness/converged", False)),
            constructive=bool(d.get("witness/constructive", False)),
            loss=float(d.get("loss", 0)),
            learning_rate=float(d.get("learning_rate", 0)),
            grad_norm=float(d.get("grad_norm", 0)),
            mean_length=float(d.get("completions/mean_length", 0)),
            reward=float(d.get("reward", 0)),
            reward_std=float(d.get("reward_std", 0)),
            frac_zero_std=float(d.get("frac_reward_zero_std", 0)),
            step_time=float(d.get("step_time", 0)),
            dyn_phase=str(d.get("dyn_weights/phase", "")),
            dyn_tau=float(d.get("dyn_weights/tau", 0)),
            dyn_tau_dot=float(d.get("dyn_weights/tau_dot", 0)),
            cascade_stage=int(d.get("cascade/stage", 0)),
            raw=d,
        )


@dataclass
class PhaseAnnotation:
    """Phase transition event for a training run."""
    step: int
    from_phase: Phase
    to_phase: Phase
    tau_before: float
    tau_after: float


@dataclass
class TrainingRun:
    """Complete training run with metadata and metrics."""
    name: str
    job_id: str = ""
    model_id: str = ""
    version: str = ""
    num_rewards: int = 0
    dataset: str = ""
    steps: list[StepMetrics] = field(default_factory=list)
    phase_transitions: list[PhaseAnnotation] = field(default_factory=list)

    @property
    def n_steps(self) -> int:
        return len(self.steps)

    @property
    def latest(self) -> StepMetrics | None:
        return self.steps[-1] if self.steps else None

    def detect_phase_transitions(self) -> list[PhaseAnnotation]:
        """Find all steps where the phase changed."""
        transitions = []
        for i in range(1, len(self.steps)):
            prev = self.steps[i - 1]
            curr = self.steps[i]
            if prev.phase != curr.phase:
                transitions.append(PhaseAnnotation(
                    step=curr.step,
                    from_phase=prev.phase,
                    to_phase=curr.phase,
                    tau_before=prev.tau,
                    tau_after=curr.tau,
                ))
        self.phase_transitions = transitions
        return transitions

    def metric_series(self, field_name: str) -> tuple[list[int], list[float]]:
        """Extract a (steps, values) pair for a named metric field."""
        steps = []
        values = []
        for s in self.steps:
            val = getattr(s, field_name, None)
            if val is None:
                val = s.raw.get(field_name, 0.0)
            steps.append(s.step)
            values.append(float(val))
        return steps, values

    @classmethod
    def from_hf_logs(cls, name: str, job_id: str, logs: list[str]) -> TrainingRun:
        """Parse a TrainingRun from HuggingFace Job logs."""
        run = cls(name=name, job_id=job_id)
        step_num = 0
        for line in logs:
            text = line.strip() if hasattr(line, "strip") else str(line)
            if "'loss':" in text and "'step_time':" in text:
                try:
                    d = eval(text)
                    step_num += 1
                    run.steps.append(StepMetrics.from_log_dict(step_num, d))
                except Exception:
                    pass
        run.detect_phase_transitions()
        return run


class MetricGroup(str, Enum):
    """Chart groups for dashboard layout."""
    TASK = "Task Performance"
    PERSISTENCE = "Persistence & Exploration"
    COHERENCE = "Coherence & Crystal"
    DYNAMICS = "Training Dynamics"
    WEIGHTS = "Dynamic Weights"
    HEALTH = "Health"


# Metric → group mapping
METRIC_GROUPS: dict[MetricGroup, list[tuple[str, str]]] = {
    MetricGroup.TASK: [
        ("task_completion", "Task Completion"),
        ("tool_engagement", "Tool Engagement"),
        ("tool_format", "Tool Format"),
        ("tools_call_frequency", "Tool Calls / Step"),
    ],
    MetricGroup.PERSISTENCE: [
        ("persistence", "Persistence"),
        ("error_utilization", "Error Utilization"),
        ("exploration", "Exploration"),
    ],
    MetricGroup.COHERENCE: [
        ("R", "Kuramoto R"),
        ("tau", "τ (Entropy Disorder)"),
        ("phi_mean", "Φ Mean"),
        ("entropy_mean", "Entropy Mean"),
        ("gated_carl", "CARL Reward"),
    ],
    MetricGroup.DYNAMICS: [
        ("loss", "Loss"),
        ("learning_rate", "Learning Rate"),
        ("mean_length", "Completion Length"),
        ("reward", "Total Reward"),
        ("step_time", "Step Time (s)"),
    ],
    MetricGroup.WEIGHTS: [
        ("dyn_tau", "Effective τ"),
        ("dyn_tau_dot", "τ Derivative"),
    ],
    MetricGroup.HEALTH: [
        ("frac_zero_std", "Zero-Std Fraction"),
        ("diversity", "Diversity"),
        ("adaptive_gr3", "Adaptive GR3"),
    ],
}


@dataclass
class RunComparison:
    """Two or more runs aligned by step for comparison."""
    runs: list[TrainingRun]
    step_range: tuple[int, int] = (0, 0)

    def __post_init__(self):
        if self.runs and self.step_range == (0, 0):
            max_steps = max(r.n_steps for r in self.runs)
            self.step_range = (1, max_steps)

    @property
    def run_names(self) -> list[str]:
        return [r.name for r in self.runs]

    def aligned_series(
        self, field_name: str, step_start: int = 0, step_end: int = 0
    ) -> list[tuple[list[int], list[float]]]:
        """Get aligned metric series for all runs within step range."""
        start = step_start or self.step_range[0]
        end = step_end or self.step_range[1]
        result = []
        for run in self.runs:
            steps, values = run.metric_series(field_name)
            filtered_steps = []
            filtered_values = []
            for s, v in zip(steps, values):
                if start <= s <= end:
                    filtered_steps.append(s)
                    filtered_values.append(v)
            result.append((filtered_steps, filtered_values))
        return result
