"""Resonance-aware learning rate modulation.

Composable LR control layers that modulate learning rate based on
model coherence state during training.

Requires ``terminals-runtime`` for the implementation.
Install via ``pip install terminals-runtime`` or contact
terminals.tech for access.
"""
from __future__ import annotations

from typing import Any, Optional

from transformers import TrainerCallback


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
        except ImportError:
            raise ImportError(
                "ResonanceLRCallback requires the terminals-runtime package. "
                "Install via: pip install terminals-runtime\n"
                "Or contact support@terminals.tech for access."
            )

    def on_step_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        self._impl.on_step_begin(args, state, control, **kwargs)

    def on_log(self, args: Any, state: Any, control: Any,
               logs: Optional[dict[str, Any]] = None, **kwargs: Any) -> None:
        self._impl.on_log(args, state, control, logs=logs, **kwargs)
