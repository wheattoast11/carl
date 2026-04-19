"""
GRPO ↔ TTT Integration — Between-step adaptation for GRPO training.
====================================================================

Two-timescale learning:
  - Slow (GRPO): Group advantage → gradient on all completions
  - Fast (LoRA micro): High-reward completions → rank-1 weight consolidation

The callback fires between GRPO steps. It reads traces from the CARL
reward function, selects high-reward completions, and applies LoRA
micro-updates that consolidate successful patterns.

Gated by phase: only fires when τ < 0.3 (crystalline). During
exploration (gaseous/fluid), the GRPO gradient alone drives learning.
During crystallization, the micro-update accelerates convergence toward
the most efficient strategies.

Requires terminals-runtime for LoRAMicroUpdate implementation.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any

try:
    import torch
except ImportError:
    torch = None  # type: ignore

try:
    from transformers import TrainerCallback
except ImportError:
    TrainerCallback = object  # type: ignore


@dataclass
class TTTStepResult:
    """Result of one between-step TTT pass."""
    step: int
    lora_loss: float = 0.0
    completions_updated: int = 0
    tau: float = 0.5
    phase: str = "fluid"
    skipped_reason: str = ""


class GRPOReflectionCallback(TrainerCallback):
    """Between-step LoRA micro-update for GRPO training.

    After each GRPO step:
    1. Read τ from logs (phase detection)
    2. If crystalline (τ < 0.3): read crystal traces from CARL reward
    3. For completions with high coherence, apply rank-1 LoRA update
    4. Model consolidates efficient patterns between gradient steps

    The micro-update operates on a SEPARATE rank-1 LoRA adapter,
    stacked on top of the GRPO training LoRA. This prevents interference
    with the GRPO optimizer's momentum.

    Args:
        carl_reward_fn: The CARL reward closure from make_carl_reward().
            Must have _last_traces attribute.
        tokenizer: The tokenizer for encoding completions.
        reward_threshold: Minimum reward to trigger micro-update (default 0.5).
        tau_gate: Maximum τ to allow updates (default 0.3 = crystalline only).
        enable: Master switch. Set False to disable all TTT.
    """

    def __init__(
        self,
        carl_reward_fn: Any,
        tokenizer: Any,
        reward_threshold: float = 0.5,
        tau_gate: float = 0.3,
        enable: bool = True,
    ):
        self._carl_fn = carl_reward_fn
        self._tokenizer = tokenizer
        self._reward_threshold = reward_threshold
        self._tau_gate = tau_gate
        self._enable = enable
        self._lora: Any = None
        self._last_tau: float = 0.5
        self._results: list[TTTStepResult] = []

    def _ensure_lora(self, model: Any) -> bool:
        """Lazy-init LoRA micro-update on first crystalline step."""
        if self._lora is not None:
            return True
        try:
            from carl_studio.ttt import LoRAMicroUpdate
            self._lora = LoRAMicroUpdate(model, rank=1)
            print("[TTT] LoRA micro-update initialized (rank-1)", file=sys.stderr)
            return True
        except ImportError:
            print("[TTT] LoRA unavailable — needs terminals-runtime", file=sys.stderr)
            self._enable = False
            return False

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Track τ from training logs."""
        if logs is None:
            return
        self._last_tau = float(logs.get("witness/tau", self._last_tau))

    def on_step_end(self, args, state, control, **kwargs):
        """Apply between-step LoRA micro-update if crystalline."""
        if not self._enable:
            return

        step = state.global_step
        tau = self._last_tau

        # Phase gate: only update during crystalline regime
        if tau > self._tau_gate:
            phase = "gaseous" if tau > 0.7 else "fluid"
            self._results.append(TTTStepResult(
                step=step, tau=tau, phase=phase,
                skipped_reason=f"tau={tau:.3f} > gate={self._tau_gate}",
            ))
            return

        # Get traces from CARL reward function
        traces_ref = getattr(self._carl_fn, "_last_traces", None)
        traces = traces_ref[0] if traces_ref and traces_ref[0] else None
        if not traces:
            self._results.append(TTTStepResult(
                step=step, tau=tau, phase="crystalline",
                skipped_reason="no traces available",
            ))
            return

        # Get model from kwargs
        model = kwargs.get("model")
        if model is None:
            return

        # Lazy-init LoRA
        if not self._ensure_lora(model):
            return

        # Select high-coherence traces for micro-update
        updated = 0
        total_loss = 0.0

        for trace in traces:
            phi_mean = float(trace.phi.mean()) if hasattr(trace, "phi") else 0.0
            if phi_mean < self._reward_threshold:
                continue

            # Tokenize the completion that produced this trace
            # The trace doesn't carry the text, but we can reconstruct
            # from token_ids if available, or skip if not
            if not hasattr(trace, "_token_ids_for_ttt"):
                continue

            token_ids = trace._token_ids_for_ttt
            if token_ids is None or len(token_ids) < 2:
                continue

            try:
                inputs = {"input_ids": torch.tensor([token_ids], device=model.device)}
                target_ids = torch.tensor([token_ids[1:]], device=model.device)
                loss = self._lora.update(
                    inputs=inputs,
                    target_ids=target_ids,
                    total_seq_len=len(token_ids),
                    target_len=len(token_ids) - 1,
                )
                total_loss += loss
                updated += 1
            except Exception as e:
                print(f"[TTT] Micro-update failed: {e}", file=sys.stderr)

        result = TTTStepResult(
            step=step,
            lora_loss=total_loss / max(updated, 1),
            completions_updated=updated,
            tau=tau,
            phase="crystalline",
        )
        self._results.append(result)

        if updated > 0:
            # Log to trainer
            logs = kwargs.get("logs", {})
            if isinstance(logs, dict):
                logs["ttt/lora_loss"] = result.lora_loss
                logs["ttt/completions_updated"] = updated

    def get_results(self) -> list[TTTStepResult]:
        """Return all TTT step results for analysis."""
        return list(self._results)

    def summary(self) -> dict:
        """Summary statistics for post-training analysis."""
        if not self._results:
            return {"steps": 0}

        updated = [r for r in self._results if r.completions_updated > 0]
        skipped = [r for r in self._results if r.skipped_reason]
        return {
            "total_steps": len(self._results),
            "steps_updated": len(updated),
            "steps_skipped": len(skipped),
            "mean_lora_loss": (
                sum(r.lora_loss for r in updated) / len(updated)
                if updated else 0.0
            ),
            "total_completions_updated": sum(r.completions_updated for r in updated),
            "skip_reasons": {
                r.skipped_reason: sum(1 for r2 in skipped if r2.skipped_reason == r.skipped_reason)
                for r in skipped
            },
        }
