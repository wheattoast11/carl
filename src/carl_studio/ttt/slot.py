"""
SLOT & LoRA Micro-Update — Test-Time Training Mechanisms
=========================================================
Test-time training for CARL models: per-sample optimization and
persistent micro-adaptation.

Requires ``terminals-runtime`` for the implementation.
Install via ``pip install terminals-runtime`` or contact
terminals.tech for access.

Part of the CARL Studio TTT (test-time training) subpackage.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class SLOTResult:
    """Result of a SLOT optimization pass."""
    delta: Any  # torch.Tensor when available
    losses: list[float]
    initial_loss: float
    final_loss: float


def _require_runtime(feature: str):
    """Raise ImportError directing to terminals-runtime."""
    raise ImportError(
        f"{feature} requires the terminals-runtime package. "
        f"Install via: pip install terminals-runtime\n"
        f"Or contact support@terminals.tech for access."
    )


class SLOTOptimizer:
    """Self-correcting Layer Optimization at Test-time.

    Optimizes a per-sample delta vector added to hidden states at a
    middle transformer layer. The delta is non-parametric (discarded
    after each interaction) and cheap (8 forward-backward passes).

    Requires terminals-runtime for the implementation.
    """

    def __init__(self, model: Any, d_model: int, **kwargs: Any):
        try:
            from terminals_runtime.ttt import SLOTOptimizerImpl
            self._impl = SLOTOptimizerImpl(model, d_model, **kwargs)
        except ImportError:
            # Fallback: admin unlock -> private repo
            try:
                from carl_studio.admin import is_admin, load_private
                if is_admin():
                    mod = load_private("slot")
                    self._impl = mod.SLOTOptimizerImpl(model, d_model, **kwargs)
                else:
                    _require_runtime("SLOTOptimizer")
            except ImportError:
                _require_runtime("SLOTOptimizer")

    def optimize(self, processor: Any, screenshot: Any, instruction: str,
                 target_text: str, system_prompt: str, **kwargs: Any) -> SLOTResult:
        return self._impl.optimize(processor, screenshot, instruction,
                                   target_text, system_prompt, **kwargs)

    def optimize_text(self, tokenizer: Any, instruction: str,
                      target_text: str, system_prompt: str) -> SLOTResult:
        """Text-only SLOT for coding sandbox / non-VLM environments."""
        return self._impl.optimize_text(tokenizer, instruction,
                                        target_text, system_prompt)

    def predict_with_delta(self, processor: Any, screenshot: Any,
                           instruction: str, system_prompt: str,
                           delta: Any, **kwargs: Any) -> str:
        return self._impl.predict_with_delta(processor, screenshot, instruction,
                                             system_prompt, delta, **kwargs)


class LoRAMicroUpdate:
    """Rank-1 LoRA update after each interaction.

    Persistent weight adaptation: each interaction contributes a tiny
    gradient step to rank-1 LoRA weights.

    Requires terminals-runtime for the implementation.
    """

    def __init__(self, model: Any, **kwargs: Any):
        try:
            from terminals_runtime.ttt import LoRAMicroUpdateImpl
            self._impl = LoRAMicroUpdateImpl(model, **kwargs)
        except ImportError:
            # Fallback: admin unlock -> private repo
            try:
                from carl_studio.admin import is_admin, load_private
                if is_admin():
                    mod = load_private("lora_micro")
                    self._impl = mod.LoRAMicroUpdateImpl(model, **kwargs)
                else:
                    _require_runtime("LoRAMicroUpdate")
            except ImportError:
                _require_runtime("LoRAMicroUpdate")

    def update(self, inputs: dict, target_ids: Any,
               total_seq_len: int, target_len: int) -> float:
        return self._impl.update(inputs, target_ids, total_seq_len, target_len)

    def get_stats(self) -> dict:
        return self._impl.get_stats()
