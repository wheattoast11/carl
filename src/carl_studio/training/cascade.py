"""Cascade reward manager — adaptive stage gating.

Two modes:
  Fixed: Stage A → B at a predetermined step (carl_start).
  Adaptive: Stage A → B when the gate metric shows sustained capability.

The adaptive gate fires when:
  1. The metric has nonzero variance (the model is differentiating)
  2. The metric mean exceeds a percentile of its own running distribution
  3. Both conditions hold for min_above of the last window steps

This replaces hardcoded thresholds with a self-calibrating gate that
adapts to any model, dataset, and reward scale.

Production hardening:
  - WS-T3: every reward value flowing through wrap_reward is clamped via
    ``_clamp_reward`` so NaN / inf / runaway magnitudes cannot poison the
    optimizer.
  - WS-T4: ``_compute_adaptive_threshold`` returns +inf BEFORE indexing into
    an empty history (previously the indexing still ran when ``_gate_history``
    was empty via other entry points).
"""

from __future__ import annotations

import logging
import math
from typing import Any, Callable

try:
    from transformers import TrainerCallback
except Exception:

    class TrainerCallback:  # type: ignore[no-redef]
        """Fallback base when transformers is unavailable."""

        pass


from carl_studio.training.rewards.multiscale import _clamp_reward


logger = logging.getLogger(__name__)


def _clamp_reward_scalar(v: float) -> float:
    """Module-local alias exposing the shared clamp helper.

    Kept as a named helper at this level so callers of the cascade module
    (tests, downstream trainers) don't have to reach across package lines.
    """
    return _clamp_reward(v)


class CascadeRewardManager:
    """Manages cascade stages for reward activation.

    Fixed mode (carl_start):
      Stage A: task rewards from step 0
      Stage B: task + CARL from carl_start

    Adaptive mode (gate_metric):
      Stage A: task rewards until gate fires
      Stage B: task + CARL after gate fires

    The gate fires when the metric shows sustained nonzero performance.
    No hardcoded threshold — the gate self-calibrates from the metric's
    running distribution.
    """

    def __init__(
        self,
        carl_start: int | None = None,
        warmup_steps: int = 10,
        # Adaptive gate parameters
        gate_metric: str | None = None,
        gate_percentile: float = 0.6,
        gate_window: int = 5,
        gate_min_above: int = 3,
        gate_min_steps: int = 5,
    ) -> None:
        if gate_metric is not None:
            self._mode = "adaptive"
            self.carl_start = float("inf")
            self._gate_metric = gate_metric
            self._gate_percentile = gate_percentile
            self._gate_window = gate_window
            self._gate_min_above = gate_min_above
            self._gate_min_steps = gate_min_steps
            self._gate_history: list[float] = []
            self._gate_fired = False
            self._gate_fired_at: int | None = None
        elif carl_start is not None:
            self._mode = "fixed"
            self.carl_start = carl_start
            self._gate_metric = None
            self._gate_history = []
            self._gate_fired = False
            self._gate_fired_at = None
        else:
            self._mode = "fixed"
            self.carl_start = 50
            self._gate_metric = None
            self._gate_history = []
            self._gate_fired = False
            self._gate_fired_at = None

        self.warmup_steps = warmup_steps
        self._step: int = 0

    def _compute_adaptive_threshold(self) -> float:
        """Compute threshold from running distribution at the configured percentile.

        WS-T4: the empty-history guard runs BEFORE any indexing, and returns
        ``+inf`` so callers short-circuit the gate-fire condition (``>= +inf``
        is never satisfied for finite metrics).
        """
        if not self._gate_history:
            return float("inf")
        sorted_vals = sorted(self._gate_history)
        idx = min(int(len(sorted_vals) * self._gate_percentile), len(sorted_vals) - 1)
        # idx is already bounded to [0, len-1] by the min() above,
        # but re-check in case the caller mutated history mid-call.
        if idx < 0 or idx >= len(sorted_vals):
            return float("inf")
        return sorted_vals[idx]

    def record_metric(self, value: float) -> None:
        """Record a gate metric value. Called by CascadeCallback on each log."""
        if self._mode != "adaptive" or self._gate_fired:
            return

        # WS-T3: clamp incoming metric so a NaN from a blown-up reward can't
        # contaminate the gate's own distribution.
        clamped = _clamp_reward_scalar(value)
        self._gate_history.append(clamped)

        # Need minimum history before gating
        if len(self._gate_history) < self._gate_min_steps:
            return

        threshold = self._compute_adaptive_threshold()
        if not math.isfinite(threshold):
            # Empty/degenerate history — gate cannot fire yet.
            return

        # Gate condition: metric >= adaptive threshold for min_above of last window steps
        # AND the threshold itself is nonzero (model is actually succeeding sometimes)
        tail = self._gate_history[-self._gate_window :]
        above = sum(1 for v in tail if v >= threshold and v > 0)

        if above >= self._gate_min_above and threshold > 0:
            self._gate_fired = True
            self._gate_fired_at = self._step
            self.carl_start = self._step

    def get_stage(self) -> str:
        """Return current cascade stage."""
        return "B" if self._step >= self.carl_start else "A"

    def get_stage_weight(self, active_in_stages: set[str]) -> float:
        """Return weight multiplier [0, 1] with linear warmup at transitions."""
        stage = self.get_stage()
        if stage not in active_in_stages:
            return 0.0
        if stage == "B" and self._step < self.carl_start + self.warmup_steps:
            return (self._step - self.carl_start + 1) / self.warmup_steps
        return 1.0

    def wrap_reward(
        self,
        reward_fn: Callable[..., list[float]],
        active_in_stages: set[str],
    ) -> Callable[..., list[float]]:
        """Wrap a reward function with cascade staging and warmup scaling.

        Every reward value returned is passed through ``_clamp_reward`` so
        that NaN/inf/runaway rewards produced by an upstream function cannot
        propagate to the optimizer.
        """
        manager = self

        def wrapped(completions: list, **kwargs: Any) -> list[float]:
            weight = manager.get_stage_weight(active_in_stages)
            if weight <= 0.0:
                return [0.0] * len(completions)
            try:
                raw = reward_fn(completions, **kwargs)
            except Exception as exc:
                # Isolate reward-function failures — training must not die
                # because one reward raised. Log and return zeros.
                logger.warning(
                    "cascade: reward fn %s raised %s; substituting zeros",
                    getattr(reward_fn, "__name__", repr(reward_fn)),
                    exc,
                )
                return [0.0] * len(completions)

            if not isinstance(raw, list):
                logger.warning(
                    "cascade: reward fn returned non-list (%s); coercing",
                    type(raw).__name__,
                )
                try:
                    raw = list(raw)
                except TypeError:
                    return [0.0] * len(completions)

            # WS-T3: clamp every reward; apply warmup scaling on the clean value.
            clamped = [_clamp_reward_scalar(r) for r in raw]
            if weight >= 1.0:
                return clamped
            return [_clamp_reward_scalar(r * weight) for r in clamped]

        wrapped.__name__ = getattr(reward_fn, "__name__", "wrapped_reward")

        # Propagate CARL metadata through the cascade wrapper so callbacks
        # can find _last_traces, _last_metrics, etc. on the outermost function.
        for attr in ("_last_metrics", "_last_traces", "_last_components", "_metrics_lock", "_step"):
            if hasattr(reward_fn, attr):
                setattr(wrapped, attr, getattr(reward_fn, attr))

        return wrapped


class CascadeCallback(TrainerCallback):
    """TRL TrainerCallback that advances cascade stage counter and logs state."""

    def __init__(self, cascade_manager: CascadeRewardManager) -> None:
        self.cascade = cascade_manager

    def on_step_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        try:
            self.cascade._step = state.global_step
        except Exception as exc:
            logger.warning("CascadeCallback.on_step_begin failed: %s", exc)

    def on_log(
        self,
        args: Any,
        state: Any,
        control: Any,
        logs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        try:
            if logs is None:
                return
            # Feed gate metric if adaptive gating is active
            if self.cascade._mode == "adaptive" and not self.cascade._gate_fired:
                metric_key = self.cascade._gate_metric
                if metric_key is not None and metric_key in logs:
                    self.cascade.record_metric(logs[metric_key])

            stage = self.cascade.get_stage()
            logs["cascade/stage"] = 0 if stage == "A" else 1
            logs["cascade/step"] = state.global_step
            if self.cascade._gate_fired_at is not None:
                logs["cascade/gate_fired_at"] = self.cascade._gate_fired_at
            if getattr(self.cascade, "_gate_history", None):
                logs["cascade/adaptive_threshold"] = self.cascade._compute_adaptive_threshold()
        except Exception as exc:
            logger.warning("CascadeCallback.on_log failed: %s", exc)
