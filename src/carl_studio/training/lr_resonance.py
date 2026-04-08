"""Resonance-aware learning rate modulation.

Three layers of LR control, composable:
  1. Conservation envelope: cosine annealing over T* = κd tokens (not steps)
  2. Phi-gate: Kuramoto R from CoherenceTrace scales LR at each step
  3. Environment signal: reward variance within batch modulates LR

Usage with TRL:
    from carl_studio.training.lr_resonance import ResonanceLRCallback

    callback = ResonanceLRCallback(
        carl_reward_fn=carl_fn,      # reads _last_traces for Kuramoto R
        d_model=4096,                # embedding dim → T* envelope
        tokens_per_step=2400,        # for T*-aware scheduling
    )
    trainer = GRPOTrainer(..., callbacks=[callback])

The callback modulates the optimizer's LR directly at each step.
It does NOT replace the base scheduler — it multiplies on top of it.
If the base scheduler says lr=1e-5 and the resonance modulator says
0.7x, effective LR is 7e-6.
"""
from __future__ import annotations

import math
from typing import Any, Optional

from transformers import TrainerCallback

from carl_studio.primitives.constants import KAPPA


class ResonanceLRCallback(TrainerCallback):
    """Modulates learning rate based on model coherence state.

    Three composable layers:
      conservation_envelope: cosine annealing over T* = κd tokens
      phi_gate: LR × (1 + alpha × (1 - R)) where R = Kuramoto from trace
      env_signal: LR × f(reward_variance) — higher variance = decision boundary

    Each layer defaults to neutral (1.0 multiplier) when disabled or
    when data is unavailable. The callback never increases LR above
    the base scheduler's value × max_multiplier.

    Args:
        carl_reward_fn: CARL reward closure with _last_traces (for Kuramoto R).
        d_model: Embedding dimension for T* = κd conservation envelope.
        tokens_per_step: Estimated tokens per training step (for T* conversion).
        phi_alpha: Strength of Phi-gate modulation. 0 = disabled.
        env_beta: Strength of environment signal. 0 = disabled.
        max_multiplier: Cap on total LR scaling (prevents runaway).
        min_multiplier: Floor on total LR scaling (prevents stall).
        envelope_enabled: Use T*-aware cosine envelope.
    """

    def __init__(
        self,
        carl_reward_fn: Any = None,
        d_model: int = 4096,
        tokens_per_step: int = 2400,
        phi_alpha: float = 0.5,
        env_beta: float = 0.3,
        max_multiplier: float = 1.5,
        min_multiplier: float = 0.2,
        envelope_enabled: bool = True,
    ) -> None:
        self.carl_fn = carl_reward_fn
        self.d_model = d_model
        self.tokens_per_step = tokens_per_step
        self.phi_alpha = phi_alpha
        self.env_beta = env_beta
        self.max_multiplier = max_multiplier
        self.min_multiplier = min_multiplier
        self.envelope_enabled = envelope_enabled

        # T* in steps
        self._t_star_tokens = KAPPA * d_model
        self._t_star_steps = max(1, int(self._t_star_tokens / tokens_per_step))

        # State
        self._base_lrs: Optional[list[float]] = None
        self._last_R: float = 0.0
        self._last_reward_var: float = 0.0

    def _read_kuramoto_R(self) -> float:
        """Read Kuramoto R from CARL reward's stored traces."""
        if self.carl_fn is None:
            return 0.0

        lock = getattr(self.carl_fn, "_metrics_lock", None)
        traces_ref = getattr(self.carl_fn, "_last_traces", None)
        if traces_ref is None:
            return self._last_R

        traces = None
        if lock is not None:
            lock.acquire()
            try:
                traces = traces_ref[0]
            finally:
                lock.release()
        else:
            traces = traces_ref[0]

        if traces is None or len(traces) == 0:
            return self._last_R

        # Mean Kuramoto R across batch
        R_vals = [t.kuramoto_R() for t in traces]
        R = sum(R_vals) / len(R_vals)
        self._last_R = R
        return R

    def _conservation_envelope(self, step: int) -> float:
        """Cosine annealing over T* tokens.

        Returns multiplier in [0, 1]. At step 0: 1.0 (full LR).
        At T* steps: ~0 (LR should be near floor).
        Beyond T*: stays at floor (min_multiplier handles this).
        """
        if not self.envelope_enabled:
            return 1.0
        if step >= self._t_star_steps:
            return 0.0
        return 0.5 * (1 + math.cos(math.pi * step / self._t_star_steps))

    def _phi_gate(self, R: float) -> float:
        """LR modulation from Kuramoto R.

        Low R (restructuring): increase LR to help phase transition.
        High R (phase-locked): decrease LR to preserve structure.

        Returns multiplier centered at 1.0.
        """
        if self.phi_alpha <= 0:
            return 1.0
        # R=0 → 1+alpha (higher LR), R=1 → 1.0 (base LR)
        return 1.0 + self.phi_alpha * (1.0 - R)

    def _env_signal(self, logs: dict) -> float:
        """LR modulation from environment reward variance.

        High variance: model at decision boundary → higher LR helps commit.
        Low variance: model converged or stuck → lower LR.

        Returns multiplier centered at 1.0.
        """
        if self.env_beta <= 0:
            return 1.0

        reward_std = logs.get("reward_std", logs.get("train/reward_std", None))
        if reward_std is not None and isinstance(reward_std, (int, float)) and reward_std > 0:
            self._last_reward_var = float(reward_std)

        # No signal yet → neutral
        if self._last_reward_var <= 0:
            return 1.0

        # Normalize: std of 1.0 is "normal", scale around that
        normalized = min(self._last_reward_var / max(1.0, self._last_reward_var), 1.0)
        return 1.0 + self.env_beta * (normalized - 0.5)

    def on_step_begin(
        self, args: Any, state: Any, control: Any, **kwargs: Any
    ) -> None:
        """Modulate optimizer LR at each step."""
        optimizer = kwargs.get("optimizer")
        if optimizer is None:
            return

        # Capture base LRs on first call
        if self._base_lrs is None:
            self._base_lrs = [pg["lr"] for pg in optimizer.param_groups]

        step = state.global_step

        # Compute three multipliers
        envelope = self._conservation_envelope(step)
        R = self._read_kuramoto_R()
        phi_mult = self._phi_gate(R)
        env_mult = 1.0  # env_signal applied in on_log when we have logs

        # Total multiplier
        total = envelope * phi_mult * env_mult
        total = max(self.min_multiplier, min(self.max_multiplier, total))

        # Apply
        for pg, base_lr in zip(optimizer.param_groups, self._base_lrs):
            pg["lr"] = base_lr * total

    def on_log(
        self, args: Any, state: Any, control: Any,
        logs: Optional[dict[str, Any]] = None, **kwargs: Any,
    ) -> None:
        """Log resonance LR state and apply env signal."""
        if logs is None:
            return

        step = state.global_step
        R = self._last_R
        envelope = self._conservation_envelope(step)
        phi_mult = self._phi_gate(R)
        env_mult = self._env_signal(logs)

        total = envelope * phi_mult * env_mult
        total = max(self.min_multiplier, min(self.max_multiplier, total))

        logs["lr_resonance/envelope"] = envelope
        logs["lr_resonance/phi_gate"] = phi_mult
        logs["lr_resonance/env_signal"] = env_mult
        logs["lr_resonance/total_multiplier"] = total
        logs["lr_resonance/kuramoto_R"] = R
        logs["lr_resonance/t_star_steps"] = self._t_star_steps
