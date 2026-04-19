"""Cascade reward manager — adaptive stage gating.

Three modes:
  Fixed: Stage A → B at a predetermined step (carl_start).
  Adaptive ("metric"): Stage A → B when a scalar gate metric shows sustained
      capability against a self-calibrating running-percentile threshold.
  Adaptive ("crystallization"): Stage A → B when the CoherenceTrace field
      shows enough crystallization events (delta_phi > DEFECT_THRESHOLD) in
      a rolling window. This is the phase-transition signature — the model
      is committing semantic structure, not just racking up reward volume.

Recommended for GRPO + CARLReward runs: ``gate_mode="crystallization"``.
Reward volume can be gamed (trivial patterns, reward hacking); phase
transitions in the Phi field cannot — they are a structural property of
the model's output distribution.

Production hardening:
  - WS-T3: every reward value flowing through wrap_reward is clamped via
    ``_clamp_reward`` so NaN / inf / runaway magnitudes cannot poison the
    optimizer.
  - WS-T4: ``_compute_adaptive_threshold`` returns +inf BEFORE indexing into
    an empty history (previously the indexing still ran when ``_gate_history``
    was empty via other entry points).
  - SEM-006: ``gate_mode="crystallization"`` reads ``_last_traces`` from the
    wrapped CARLReward on every step. Traces are never required — if they
    are missing (e.g. first step before training produced any output), the
    gate simply holds stage A.
"""

from __future__ import annotations

import collections
import logging
import math
from typing import Any, Callable, Literal

try:
    from transformers import TrainerCallback
except Exception:

    class TrainerCallback:  # type: ignore[no-redef]
        """Fallback base when transformers is unavailable."""

        pass


from carl_studio.training.rewards.multiscale import _clamp_reward


logger = logging.getLogger(__name__)

GateMode = Literal["metric", "crystallization"]


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

    Adaptive modes fire the gate without a hardcoded threshold:

      gate_mode="metric" (default; requires ``gate_metric``):
        Gate fires when the named scalar metric (e.g. task_completion_rate)
        exceeds its own running percentile for ``gate_min_above`` of the
        last ``gate_window`` observations. Self-calibrates to any reward
        scale — but can be gamed by reward volume alone.

      gate_mode="crystallization" (recommended for GRPO + CARLReward):
        Gate fires when the CARLReward's CoherenceTrace shows at least
        ``n_crystallizations_required`` crystallization events summed over
        the last ``crystallization_window`` steps. A crystallization is
        ``delta_phi > DEFECT_THRESHOLD`` at low prior phi — the structural
        signature of the model committing semantic order. Cannot be gamed
        by reward volume; it is a phase-transition property of the Phi
        field itself.

    The crystallization mode consumes traces by inspecting the wrapped
    reward function's ``_last_traces`` attribute on each step. The wrapped
    function must therefore be a ``CARLReward`` factory output (i.e. from
    ``make_carl_reward``) — anything else has no traces, so the gate will
    hold stage A. This is by design: crystallization gating is specifically
    for runs with a coherence-aware reward.
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
        gate_mode: GateMode = "metric",
        n_crystallizations_required: int = 3,
        crystallization_window: int = 10,
    ) -> None:
        if gate_mode not in ("metric", "crystallization"):
            raise ValueError(
                f"gate_mode must be 'metric' or 'crystallization', got {gate_mode!r}"
            )
        if n_crystallizations_required < 1:
            raise ValueError(
                "n_crystallizations_required must be >= 1, got "
                f"{n_crystallizations_required}"
            )
        if crystallization_window < 1:
            raise ValueError(
                f"crystallization_window must be >= 1, got {crystallization_window}"
            )

        # Crystallization gating is adaptive even without a gate_metric —
        # the "metric" it watches is the CARLReward trace field.
        is_adaptive = gate_metric is not None or gate_mode == "crystallization"

        if is_adaptive:
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

        self.warmup_steps = max(1, warmup_steps)
        self._step: int = 0

        # Crystallization-mode state. The rolling window tracks the per-step
        # total of ``trace.n_crystallizations`` across the current batch.
        self._gate_mode: GateMode = gate_mode
        self._n_crystallizations_required = n_crystallizations_required
        self._crystallization_window = crystallization_window
        self._recent_crystallizations: collections.deque[int] = collections.deque(
            maxlen=crystallization_window
        )
        # Reward function bound later by ``attach_reward_fn`` so the callback
        # can find the traces without us keeping a second copy.
        self._reward_fn: Callable[..., list[float]] | None = None

    # ------------------------------------------------------------------
    # Crystallization-mode reward binding
    # ------------------------------------------------------------------

    def attach_reward_fn(self, reward_fn: Callable[..., list[float]]) -> None:
        """Bind the CARLReward-producing reward function for trace access.

        Only meaningful when ``gate_mode == "crystallization"``. The callback
        reads ``reward_fn._last_traces`` on each step. If the reward function
        does not expose ``_last_traces`` (e.g. it is not a CARLReward factory
        output), the gate will simply hold stage A — no crash.
        """
        self._reward_fn = reward_fn

    def record_crystallizations(self, n: int) -> None:
        """Push a per-step crystallization count into the rolling window.

        Called by ``CascadeCallback.on_step_end`` after reading
        ``trace.n_crystallizations`` from the bound reward function. Also
        callable directly from tests without a Trainer to exercise the
        gate logic.
        """
        if self._mode != "adaptive" or self._gate_fired:
            return
        if self._gate_mode != "crystallization":
            return

        if n < 0 or not math.isfinite(n):
            # Defensive: n_crystallizations is an int count; anything
            # pathological gets normalized to 0.
            n = 0
        self._recent_crystallizations.append(int(n))

        total = sum(self._recent_crystallizations)
        if total >= self._n_crystallizations_required:
            self._gate_fired = True
            self._gate_fired_at = self._step
            self.carl_start = self._step

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
        # Crystallization gating ignores scalar metric reads entirely — the
        # trace window is the signal, not reward volume.
        if self._gate_mode != "metric":
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

        When ``gate_mode == "crystallization"`` the wrapped function is also
        auto-bound to this manager so ``CascadeCallback`` can fetch
        ``_last_traces`` without extra wiring at the training-pipeline level.
        """
        manager = self
        if manager._gate_mode == "crystallization" and manager._reward_fn is None:
            manager.attach_reward_fn(reward_fn)

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

    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        """Pump the crystallization window from the bound reward's traces.

        Only active when the cascade is in ``gate_mode="crystallization"``
        and has a reward function bound (via ``wrap_reward`` or explicit
        ``attach_reward_fn``). Silently no-ops otherwise.
        """
        try:
            cascade = self.cascade
            if cascade._gate_fired or cascade._mode != "adaptive":
                return
            if cascade._gate_mode != "crystallization":
                return
            reward_fn = cascade._reward_fn
            if reward_fn is None:
                return
            traces_ref = getattr(reward_fn, "_last_traces", None)
            # Factory stores list[CoherenceTrace] | None in a single-slot list.
            if traces_ref is None:
                return
            traces = traces_ref[0] if isinstance(traces_ref, list) else None
            if not traces:
                return
            total = 0
            for trace in traces:
                n = getattr(trace, "n_crystallizations", None)
                if n is None:
                    continue
                total += int(n)
            cascade.record_crystallizations(total)
        except Exception as exc:
            logger.warning("CascadeCallback.on_step_end failed: %s", exc)

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
            # Feed gate metric if adaptive metric-mode gating is active
            if (
                self.cascade._mode == "adaptive"
                and not self.cascade._gate_fired
                and self.cascade._gate_mode == "metric"
            ):
                metric_key = self.cascade._gate_metric
                if metric_key is not None and metric_key in logs:
                    self.cascade.record_metric(logs[metric_key])

            stage = self.cascade.get_stage()
            logs["cascade/stage"] = 0 if stage == "A" else 1
            logs["cascade/step"] = state.global_step
            logs["cascade/gate_mode"] = self.cascade._gate_mode
            if self.cascade._gate_fired_at is not None:
                logs["cascade/gate_fired_at"] = self.cascade._gate_fired_at
            if self.cascade._gate_mode == "metric" and getattr(
                self.cascade, "_gate_history", None
            ):
                logs["cascade/adaptive_threshold"] = self.cascade._compute_adaptive_threshold()
            if self.cascade._gate_mode == "crystallization":
                logs["cascade/recent_crystallizations"] = sum(
                    self.cascade._recent_crystallizations
                )
                logs["cascade/crystallizations_required"] = (
                    self.cascade._n_crystallizations_required
                )
        except Exception as exc:
            logger.warning("CascadeCallback.on_log failed: %s", exc)
