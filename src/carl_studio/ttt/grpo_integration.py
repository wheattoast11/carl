"""
GRPO ↔ TTT Integration — Between-step adaptation for GRPO training.
====================================================================

Two-timescale learning:
  - Slow (GRPO): Group advantage → gradient on all completions
  - Fast (LoRA micro): High-reward completions → rank-1 weight consolidation

The callback fires between GRPO steps. It reads traces from the CARL
reward function, selects high-reward completions, and applies LoRA
micro-updates that consolidate successful patterns.

Gated by phase: only fires when τ < TTT_CRYSTALLINE_THRESHOLD (0.3).
During exploration (gaseous/fluid), the GRPO gradient alone drives
learning. During crystallization, the micro-update accelerates
convergence toward the most efficient strategies.

TTT micro-update precondition: tau < TTT_CRYSTALLINE_THRESHOLD.
Tau is now computed from the CARL reward's ``_last_traces`` via
Kuramoto-R in the public path (SEM-009). Previously the gate was
dormant without ``terminals-runtime`` writing ``witness/tau`` to the
logs stream — the callback fell back to the initial ``_last_tau=0.5``
for every step, which means ``0.5 > 0.3`` and the micro-update never
fired. The public callback now derives ``tau = 1 - mean_kuramoto_R``
from the reward function's cached trace batch, mirroring
``training/lr_resonance.py::ResonanceLRCallback._read_kuramoto_R``.

Precedence: callback-computed tau (fresh traces this step) takes
priority over any ``witness/tau`` pre-written to the log stream by
``terminals-runtime`` or another upstream source. If no traces are
available this step, the callback falls back to ``logs["witness/tau"]``
(runtime-supplied), then to the last observed tau
(``self._last_tau``).

Requires terminals-runtime for LoRAMicroUpdate implementation.
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import torch
except ImportError:
    torch = None  # type: ignore

try:
    from transformers import TrainerCallback
except ImportError:
    TrainerCallback = object  # type: ignore


# ---------------------------------------------------------------------------
# Phase thresholds (shared with the lr_resonance classifier + the CARL
# phase-adaptive composite). Kept module-level so tests and downstream
# callers can reference the exact gate the micro-update enforces.
# ---------------------------------------------------------------------------
TTT_CRYSTALLINE_THRESHOLD: float = 0.3
"""Upper bound on tau for the TTT micro-update to fire.

tau = 1 - mean(kuramoto_R). Low tau = high phase alignment = crystalline.
"""


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
        carl_reward_fn: Any = None,
        tokenizer: Any = None,
        reward_threshold: float = 0.5,
        tau_gate: float = TTT_CRYSTALLINE_THRESHOLD,
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

    def _read_kuramoto_R(self) -> float | None:
        """Read current mean Kuramoto-R from the CARL reward's _last_traces.

        Mirrors ``training/lr_resonance.py::ResonanceLRCallback._read_kuramoto_R``
        so the public training path produces ``witness/tau`` without depending
        on ``terminals-runtime``. If no reward function or no traces are
        available, returns ``None`` and the caller falls back to whatever
        upstream populated ``witness/tau`` (if anything), then to the last
        observed tau.

        The ``make_carl_reward`` closure exposes ``_last_traces`` as a
        one-slot list whose single element is a ``list[CoherenceTrace]``
        (the current training batch). This helper unwraps both that
        one-slot shape and the direct-list shape for forward-compat. When
        a metrics lock is exposed, it is acquired to avoid racing the
        reward function's batch write (same pattern as the resonance LR
        callback).
        """
        reward_fn = getattr(self, "_carl_fn", None)
        if reward_fn is None:
            return None

        traces_holder = getattr(reward_fn, "_last_traces", None)
        if traces_holder is None:
            return None

        # Guard against races with the reward function's batch write.
        lock = getattr(reward_fn, "_metrics_lock", None)
        if lock is not None:
            lock.acquire()
            try:
                traces = self._unwrap_traces(traces_holder)
            finally:
                lock.release()
        else:
            traces = self._unwrap_traces(traces_holder)

        if not traces:
            return None

        try:
            rs: list[float] = []
            for trace_any in traces:
                t: Any = trace_any
                if t is None:
                    continue
                # Prefer the trace's own kuramoto_R (used by the phase-adaptive
                # composite). Fall back to a phi-field integration if the trace
                # doesn't expose it (mirrors the lr_resonance computation, which
                # hits the raw phi array).
                r_val: float | None
                if hasattr(t, "kuramoto_R"):
                    try:
                        r_val = float(t.kuramoto_R())
                    except Exception:
                        r_val = self._kuramoto_from_phi(t)
                else:
                    r_val = self._kuramoto_from_phi(t)
                if r_val is None or not math.isfinite(r_val):
                    continue
                rs.append(r_val)
            if not rs:
                return None
            return sum(rs) / len(rs)
        except Exception:
            # Classifier must never crash training. Fall back to upstream tau.
            return None

    @staticmethod
    def _unwrap_traces(holder: Any) -> list[Any]:
        """Normalize the ``_last_traces`` holder to a flat ``list[trace]``.

        Accepts both the one-slot list ``[list_of_traces]`` emitted by
        ``make_carl_reward`` and a direct ``list[trace]`` emitted by the
        phase-adaptive composite. Returns ``[]`` for any other shape.
        """
        if not isinstance(holder, list):
            return []
        holder_list: list[Any] = list(holder)  # type: ignore[arg-type]
        if not holder_list:
            return []
        first: Any = holder_list[0]
        if isinstance(first, list):
            # One-slot wrapper: [list_of_traces]
            first_list: list[Any] = list(first)  # type: ignore[arg-type]
            return [t for t in first_list if t is not None]
        # Direct list of trace objects (or Nones); keep non-null entries only.
        return [t for t in holder_list if t is not None]

    @staticmethod
    def _kuramoto_from_phi(trace: Any) -> float | None:
        """Compute Kuramoto-R from a trace's raw phi field.

        Mirrors ``ResonanceLRCallback._read_kuramoto_R`` (lr_resonance.py:146-156):
        half-circle mapping ``theta = pi * phi`` followed by the magnitude of
        the mean complex exponential. Used as a fallback for trace objects
        that don't expose ``kuramoto_R()`` directly (e.g. test stubs that
        only provide ``phi``).
        """
        phi = getattr(trace, "phi", None)
        if phi is None:
            return None
        arr = np.asarray(phi, dtype=float)
        if arr.size == 0:
            return None
        try:
            return float(abs(np.mean(np.exp(1j * math.pi * arr))))
        except Exception:
            return None

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Track τ from training logs.

        Publishes ``witness/tau`` and ``witness/kuramoto_R`` to the logs
        dict so downstream callbacks (and the trainer's log stream) can
        consume the same signal without depending on ``terminals-runtime``.
        See module docstring (SEM-009) for the precedence rules:

          1. Callback-computed tau from fresh traces this step wins.
          2. Otherwise, any ``witness/tau`` already present in logs
             (e.g. written by terminals-runtime).
          3. Otherwise, the last observed tau (initial ``_last_tau=0.5``
             on first call).
        """
        if logs is None:
            return

        r = self._read_kuramoto_R()
        if r is not None:
            tau = 1.0 - r
            # Publish the witness signals the TTT + lr_resonance stack expects.
            # Only overwrite if we actually have a fresh computation — never
            # clobber an upstream value with stale data.
            logs["witness/tau"] = tau
            logs["witness/kuramoto_R"] = r
        else:
            # Fall back to whatever upstream populated, then to the last tau.
            tau = float(logs.get("witness/tau", self._last_tau))

        self._last_tau = tau

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
