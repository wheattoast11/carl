"""CARL autonomous agent loop — 7-state FSM."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

from carl_studio.agent.scheduler import Scheduler
from carl_studio.agent.states import AgentState, valid_transition

logger = logging.getLogger("carl.agent")


@dataclass
class Observation:
    """Collected metrics from one observe cycle."""

    step: int
    task_completion: float
    phi_mean: float
    kuramoto_r: float
    tau: float
    reward_values: dict[str, float] = field(default_factory=dict)
    anomalies: list[str] = field(default_factory=list)


@dataclass
class Hypothesis:
    """A generated hypothesis about training dynamics."""

    claim: str
    predictions: list[str]
    experiment_config: dict
    priority: float  # 0-1, signal strength


@dataclass
class AgentLog:
    """One tick of the agent loop."""

    timestamp: str
    from_state: str
    to_state: str
    action: str
    details: dict = field(default_factory=dict)


class AutonomyAgent:
    """Autonomous agent loop. PAID tier only.

    7-state FSM: IDLE -> OBSERVE -> HYPOTHESIZE -> EXECUTE -> GATE -> PROMOTE -> SHADOW

    Each state handler returns the next state. The tick() method
    validates transitions and logs state changes.

    Not to be confused with :class:`carl_studio.chat_agent.CARLAgent`, which
    is the Anthropic-backed interactive chat loop used by all CLI paths.
    ``AutonomyAgent`` is the background autonomy FSM gated behind the
    experiment feature flag.
    """

    def __init__(self, scheduler: Scheduler | None = None):
        self.state = AgentState.IDLE
        self.scheduler = scheduler
        self._observations: list[Observation] = []
        self._current_hypothesis: Hypothesis | None = None
        self._current_job_id: str | None = None
        self._history: list[AgentLog] = []
        self._tick_count = 0
        self._handlers = {
            AgentState.IDLE: self._on_idle,
            AgentState.OBSERVE: self._on_observe,
            AgentState.HYPOTHESIZE: self._on_hypothesize,
            AgentState.EXECUTE: self._on_execute,
            AgentState.GATE: self._on_gate,
            AgentState.PROMOTE: self._on_promote,
            AgentState.SHADOW: self._on_shadow,
        }

    def tick(self) -> AgentState:
        """Execute one FSM step. Returns the new state."""
        self._tick_count += 1
        old_state = self.state

        handler = self._handlers.get(old_state, lambda: AgentState.IDLE)

        new_state = handler()

        # Validate transition — invalid transitions fall back to IDLE
        if not valid_transition(old_state, new_state) and not (
            old_state == AgentState.IDLE and new_state == AgentState.IDLE
        ):
            logger.warning(
                "Invalid transition %s -> %s, falling back to IDLE",
                old_state.value,
                new_state.value,
            )
            new_state = AgentState.IDLE

        self._history.append(
            AgentLog(
                timestamp=datetime.now(timezone.utc).isoformat(),
                from_state=old_state.value,
                to_state=new_state.value,
                action=f"tick_{self._tick_count}",
            )
        )

        self.state = new_state
        return new_state

    # -- State handlers --------------------------------------------------

    def _on_idle(self) -> AgentState:
        """Check for triggers: scheduler, manual trigger, or stay idle."""
        if self.scheduler:
            due = self.scheduler.check_due()
            if due is not None:
                logger.info("Schedule fired: %s (%s)", due.action, due.cron_expr)
                return AgentState.OBSERVE
        return AgentState.IDLE

    def _on_observe(self) -> AgentState:
        """Collect metrics and check for anomalies."""
        if self._observations:
            return AgentState.HYPOTHESIZE
        return AgentState.IDLE

    def _on_hypothesize(self) -> AgentState:
        """Form hypothesis from observations."""
        if self._current_hypothesis is not None:
            return AgentState.EXECUTE
        return AgentState.IDLE

    def _on_execute(self) -> AgentState:
        """Run experiment. Always transitions to GATE."""
        return AgentState.GATE

    def _on_gate(self) -> AgentState:
        """Evaluate results. PASS -> PROMOTE, FAIL -> HYPOTHESIZE."""
        if self._current_job_id:
            return AgentState.PROMOTE
        return AgentState.HYPOTHESIZE

    def _on_promote(self) -> AgentState:
        """Push results. Can go to SHADOW or IDLE."""
        return AgentState.IDLE

    def _on_shadow(self) -> AgentState:
        """Run shadow cycle. Returns to OBSERVE."""
        return AgentState.OBSERVE

    # -- Injection API (testing / manual triggers) -----------------------

    def inject_observation(self, obs: Observation) -> None:
        """Manually inject an observation."""
        self._observations.append(obs)

    def inject_hypothesis(self, hyp: Hypothesis) -> None:
        """Manually inject a hypothesis."""
        self._current_hypothesis = hyp

    def inject_job_id(self, job_id: str) -> None:
        """Set the current job ID for gate evaluation."""
        self._current_job_id = job_id

    # -- Properties ------------------------------------------------------

    @property
    def history(self) -> list[AgentLog]:
        """Return a copy of the transition history."""
        return list(self._history)

    @property
    def tick_count(self) -> int:
        """Number of ticks executed so far."""
        return self._tick_count

    # -- Main loop -------------------------------------------------------

    def run(self, max_ticks: int = 0, interval_seconds: float = 300.0) -> None:
        """Main loop. Tick every interval_seconds.

        Args:
            max_ticks: Stop after N ticks. 0 = run forever.
            interval_seconds: Sleep between ticks.
        """
        ticks = 0
        while max_ticks == 0 or ticks < max_ticks:
            self.tick()
            ticks += 1
            if max_ticks == 0 or ticks < max_ticks:
                time.sleep(interval_seconds)
