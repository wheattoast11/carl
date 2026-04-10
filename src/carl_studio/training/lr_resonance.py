"""Resonance-aware learning rate modulation.

Composable LR control layers that modulate learning rate based on
model coherence state during training.

Requires ``terminals-runtime`` for the implementation.
Install via ``pip install terminals-runtime`` or contact
terminals.tech for access.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np

try:
    from transformers import TrainerCallback
except Exception:

    class TrainerCallback:  # type: ignore[no-redef]
        """Fallback base when transformers is unavailable."""

        pass


from carl_studio.primitives.constants import KAPPA


class ResonanceLRCallback(TrainerCallback):
    """Modulates learning rate based on model coherence dynamics.

    Three composable control layers for resonance-aware LR scheduling.
    Multiplies on top of the base scheduler — never replaces it.

    Requires terminals-runtime for the implementation.

    Args:
        carl_reward_fn: CARL reward closure (for coherence state).
        d_model: Embedding dimension.
        tokens_per_step: Estimated tokens per training step.
        phi_alpha: Strength of coherence-gate modulation.
        env_beta: Strength of environment signal.
        max_multiplier: Cap on total LR scaling.
        min_multiplier: Floor on total LR scaling.
        envelope_enabled: Use conservation-aware cosine envelope.
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
        self.carl_reward_fn = carl_reward_fn
        self.d_model = d_model
        self.tokens_per_step = tokens_per_step
        self.phi_alpha = phi_alpha
        self.env_beta = env_beta
        self.max_multiplier = max_multiplier
        self.min_multiplier = min_multiplier
        self.envelope_enabled = envelope_enabled
        self._t_star_steps = int(KAPPA * self.d_model / max(self.tokens_per_step, 1))
        self._kuramoto_R = 0.5
        self._impl = None

        try:
            from terminals_runtime.training import ResonanceLRCallbackImpl

            self._impl = ResonanceLRCallbackImpl(
                carl_reward_fn=carl_reward_fn,
                d_model=d_model,
                tokens_per_step=tokens_per_step,
                phi_alpha=phi_alpha,
                env_beta=env_beta,
                max_multiplier=max_multiplier,
                min_multiplier=min_multiplier,
                envelope_enabled=envelope_enabled,
            )
        except Exception:
            self._impl = None

    def _conservation_envelope(self, step: int) -> float:
        if not self.envelope_enabled or self._t_star_steps <= 0:
            return 1.0
        if step <= 0:
            return 1.0
        if step >= self._t_star_steps:
            return 0.0

        progress = step / self._t_star_steps
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    def _phi_gate(self, kuramoto_R: float | None = None) -> float:
        if self.phi_alpha <= 0:
            return 1.0

        if kuramoto_R is None:
            kuramoto_R = self._kuramoto_R

        r = max(0.0, min(1.0, float(kuramoto_R)))
        multiplier = 1.0 + self.phi_alpha * (1.0 - r)
        return max(self.min_multiplier, min(self.max_multiplier, multiplier))

    def _env_signal(self, logs: Optional[dict[str, Any]] = None) -> float:
        if self.env_beta <= 0 or not logs:
            return 1.0

        reward_std = float(logs.get("reward_std", 0.0) or 0.0)
        if reward_std <= 0:
            return 1.0

        signal = 1.0 + self.env_beta * min(reward_std, 1.0)
        return max(self.min_multiplier, min(self.max_multiplier, signal))

    def _read_kuramoto_R(self) -> float:
        if self.carl_reward_fn is None:
            self._kuramoto_R = 0.5
            return self._kuramoto_R

        traces_ref = getattr(self.carl_reward_fn, "_last_traces", None)
        if not traces_ref:
            self._kuramoto_R = 0.5
            return self._kuramoto_R

        lock = getattr(self.carl_reward_fn, "_metrics_lock", None)
        if lock is not None:
            lock.acquire()
            try:
                traces = traces_ref[0]
            finally:
                lock.release()
        else:
            traces = traces_ref[0]

        if not traces:
            self._kuramoto_R = 0.5
            return self._kuramoto_R

        values: list[float] = []
        for trace in traces:
            phi = getattr(trace, "phi", None)
            if phi is None:
                continue
            arr = np.asarray(phi, dtype=float)
            if arr.size == 0:
                continue
            values.append(float(abs(np.mean(np.exp(1j * math.pi * arr)))))

        self._kuramoto_R = float(sum(values) / len(values)) if values else 0.5
        return self._kuramoto_R

    def on_step_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        if self._impl is not None:
            self._impl.on_step_begin(args, state, control, **kwargs)
            return

        optimizer = kwargs.get("optimizer")
        if optimizer is None:
            return

        step = int(getattr(state, "global_step", 0) or 0)
        envelope = self._conservation_envelope(step)
        kuramoto_R = self._read_kuramoto_R()
        phi_gate = self._phi_gate(kuramoto_R)
        env_signal = self._env_signal(kwargs.get("logs"))

        total_multiplier = envelope * phi_gate * env_signal
        total_multiplier = max(self.min_multiplier, min(self.max_multiplier, total_multiplier))

        for group in optimizer.param_groups:
            base_lr = float(group.get("lr", 0.0) or 0.0)
            group["lr"] = base_lr * total_multiplier

    def on_log(
        self,
        args: Any,
        state: Any,
        control: Any,
        logs: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        if self._impl is not None:
            self._impl.on_log(args, state, control, logs=logs, **kwargs)
            return

        if logs is None:
            logs = {}

        step = int(getattr(state, "global_step", 0) or 0)
        kuramoto_R = self._read_kuramoto_R()
        envelope = self._conservation_envelope(step)
        phi_gate = self._phi_gate(kuramoto_R)
        env_signal = self._env_signal(logs)
        total_multiplier = envelope * phi_gate * env_signal
        total_multiplier = max(self.min_multiplier, min(self.max_multiplier, total_multiplier))

        logs["lr_resonance/envelope"] = envelope
        logs["lr_resonance/phi_gate"] = phi_gate
        logs["lr_resonance/total_multiplier"] = total_multiplier
        logs["lr_resonance/kuramoto_R"] = kuramoto_R
        logs["lr_resonance/t_star_steps"] = self._t_star_steps
