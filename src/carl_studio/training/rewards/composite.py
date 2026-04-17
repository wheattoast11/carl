"""CARL composite reward: 50% multiscale + 30% cloud + 20% discontinuity.

Now built on CoherenceTrace — the per-token field is computed once and
all metrics derive from it. No triple compute_phi() calls.

Implements the RewardFunction protocol and provides make_carl_reward() factory
that returns a TRL-compatible closure.

Fixes applied:
  L1  -- eval/train mode toggle during forward pass
  L2  -- thread-safe _last_metrics with threading.Lock
  L8  -- VRAM cleanup after forward pass
  L9  -- log OOM to stderr
"""

from __future__ import annotations

import sys
import threading
from typing import Any, TypedDict

import numpy as np

from carl_core.coherence_trace import CoherenceTrace


class RewardComponents(TypedDict):
    coherence: float
    cloud_quality: float
    discontinuity: float


from carl_studio.training.rewards.base import extract_text


# ---------------------------------------------------------------------------
# CARLReward composite
# ---------------------------------------------------------------------------

class CARLReward:
    """Composite CARL reward: 50% multiscale + 30% cloud + 20% discontinuity.

    Now backed by CoherenceTrace — computes the Phi field once, derives
    all three components from the cached field arrays.
    """

    def __init__(
        self,
        weight_multiscale: float = 0.5,
        weight_cloud: float = 0.3,
        weight_discontinuity: float = 0.2,
    ) -> None:
        self.weight_multiscale = weight_multiscale
        self.weight_cloud = weight_cloud
        self.weight_discontinuity = weight_discontinuity

    def score(
        self, logits: np.ndarray, token_ids: np.ndarray
    ) -> tuple[float, dict[str, float]]:
        """Compute composite CARL score from logits.

        Args:
            logits: [T, V] raw logits.
            token_ids: [T] selected token indices.

        Returns:
            (composite_score, component_dict) where component_dict has keys:
            multiscale, cloud_quality, discontinuity.
        """
        trace = CoherenceTrace.from_logits(logits, token_ids)
        return self.score_from_trace(trace)

    def score_from_trace(
        self, trace: CoherenceTrace
    ) -> tuple[float, dict[str, float]]:
        """Compute composite CARL score from a pre-computed trace.

        This is the efficient path — no logits needed, everything
        is already in the trace's per-token arrays.
        """
        composite = trace.carl_reward(
            w_coherence=self.weight_multiscale,
            w_cloud=self.weight_cloud,
            w_discontinuity=self.weight_discontinuity,
        )
        components = {
            "multiscale": trace.multiscale_coherence,
            "cloud_quality": trace.cloud_quality,
            "discontinuity": trace.discontinuity_score,
        }
        return float(composite), components


# ---------------------------------------------------------------------------
# TRL-compatible factory
# ---------------------------------------------------------------------------

def make_carl_reward(
    model: Any,
    tokenizer: Any,
    vocab_size: int = 128000,
    active_after_step: int = 0,
    max_length: int = 512,
) -> Any:
    """Factory returning a TRL-compatible CARL reward function.

    The returned closure:
    1. Tokenizes each completion text
    2. Runs a torch.no_grad() forward pass to get logits
    3. Constructs a CoherenceTrace from the logits (computed ONCE)
    4. Derives CARL composite from the trace (no triple compute_phi)
    5. Stores traces for CoherenceTraceCallback pickup

    Fixes applied:
      L1 -- eval/train mode toggle
      L2 -- thread-safe _last_metrics with threading.Lock
      L8 -- VRAM cleanup (del + empty_cache)
      L9 -- log OOM to stderr

    Args:
        model: The model (captured in closure for forward pass).
        tokenizer: The tokenizer (captured in closure).
        vocab_size: Vocabulary size for entropy normalization.
        active_after_step: Return 0.0 before this step (cascade integration).
        max_length: Max token length for CARL forward pass.
    """
    import torch

    carl = CARLReward()
    _step_counter = [0]
    _last_metrics: list[Any] = [None]
    _last_components: list[Any] = [None]
    _last_traces: list[Any] = [None]  # NEW: stores list[CoherenceTrace] for last batch
    _metrics_lock = threading.Lock()

    @torch.no_grad()
    def carl_composite_reward(completions: list, **kwargs: Any) -> list[float]:
        _step_counter[0] += 1

        # Cascade: return zeros before activation step
        if _step_counter[0] < active_after_step:
            return [0.0] * len(completions)

        rewards: list[float] = []
        batch_metrics: list[dict[str, float]] = []
        batch_traces: list[CoherenceTrace] = []

        for idx, completion in enumerate(completions):
            text = extract_text(completion)
            if not text.strip() or len(text) < 10:
                rewards.append(0.0)
                continue

            try:
                inputs = tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_length,
                    padding=False,
                )
                inputs = {k: v.to(model.device) for k, v in inputs.items()}

                # Fix L1: eval/train mode toggle
                was_training = model.training
                model.eval()
                try:
                    outputs = model(**inputs)
                finally:
                    if was_training:
                        model.train()

                logits_t = outputs.logits[0]  # [T, V]
                token_ids_t = inputs["input_ids"][0]  # [T]

                logits_np = logits_t.cpu().float().numpy()
                token_ids_np = token_ids_t.cpu().numpy()

                # Fix L8: VRAM cleanup
                del outputs, logits_t, inputs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Build trace ONCE — all metrics derive from it
                trace = CoherenceTrace.from_logits(
                    logits_np,
                    token_ids_np,
                    step=_step_counter[0],
                    sample_idx=idx,
                )
                # Store token IDs for TTT micro-updates (dynamic attr, not in dataclass)
                trace._token_ids_for_ttt = token_ids_np.tolist()  # type: ignore[attr-defined]
                batch_traces.append(trace)

                score, components = carl.score_from_trace(trace)
                batch_metrics.append(components)
                rewards.append(round(float(score), 4))

            except Exception as e:
                # Fix L9: log OOM to stderr
                print(f"[CARL] Forward pass failed: {e}", file=sys.stderr)
                rewards.append(0.0)

        # Fix L2: thread-safe metrics + trace storage
        if batch_metrics:
            with _metrics_lock:
                _last_metrics[0] = (_step_counter[0], batch_metrics)
                _last_components[0] = [
                    RewardComponents(
                        coherence=m.get("multiscale", 0.0),
                        cloud_quality=m.get("cloud_quality", 0.0),
                        discontinuity=m.get("discontinuity", 0.0),
                    )
                    for m in batch_metrics
                ]
                _last_traces[0] = batch_traces

        return rewards

    # Expose internal state for monitoring callbacks
    carl_composite_reward._last_metrics = _last_metrics  # type: ignore[attr-defined]
    carl_composite_reward._last_components = _last_components  # type: ignore[attr-defined]
    carl_composite_reward._last_traces = _last_traces  # type: ignore[attr-defined]
    carl_composite_reward._metrics_lock = _metrics_lock  # type: ignore[attr-defined]
    carl_composite_reward._step = _step_counter  # type: ignore[attr-defined]
    return carl_composite_reward
