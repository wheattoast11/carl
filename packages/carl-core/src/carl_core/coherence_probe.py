"""
coherence_probe.py -- Observability probe for coherence-aware training.

Ported from zero-rl-pipeline/phase1/crystal_observer.py (CrystalProbe).

Fixes applied:
  L3: np.diff(phi) instead of np.diff(phi, prepend=phi[0]) --
      prepend=phi[0] produces a leading zero that masks the first real delta.
      Cross-batch continuity uses _prev_phi only for the boundary element.
  L4: scale_coherence clamped to [0, 1].
  B2: discontinuity_score added to CoherenceSnapshot.
"""

from __future__ import annotations

import math
import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from .coherence_trace import CoherenceTrace, LayeredTrace

# Opt-in gate for multi-layer probing. Callers in training/reward paths can
# branch on this to decide whether to collect hidden_states (they do NOT
# need to be collected by default — the fast logits-only path remains the
# default). Flipping this env var to a truthy value signals that the caller
# should materialize hidden_states and invoke ``measure_multi_layer``.
CARL_LAYER_PROBE_ENABLED: bool = os.environ.get("CARL_LAYER_PROBE") in {
    "1",
    "true",
    "True",
}


class CoherenceSnapshot(BaseModel):
    """One measurement of the coherence state at a training step."""

    step: int
    n_tokens: int

    # Order parameter
    phi_mean: float
    phi_std: float
    phi_trajectory: List[float] = Field(default_factory=list)

    # Defects
    n_defects: int
    n_crystallizations: int
    n_meltings: int
    defect_density: float

    # B2: context-dependent discontinuity scoring
    # Normalized score in [0, 1] reflecting severity-weighted defect load.
    # 0 = no discontinuities; 1 = every token boundary is a max-severity defect.
    discontinuity_score: float = 0.0

    # Cloud quality
    cloud_quality_mean: float

    # Multi-scale coherence (clamped to [0, 1] per L4)
    scale_coherence: Dict[int, float] = Field(default_factory=dict)

    # Entropy
    entropy_mean: float
    entropy_std: float

    # Surprisal
    surprisal_mean: float
    surprisal_std: float

    # Top-K concentration
    top_k_mass: float

    # Advantage signal (optional, from GRPO trainer)
    advantage_mean: Optional[float] = None
    advantage_above_sigma: Optional[float] = None


class CoherenceProbe:
    """
    Extracts coherence dynamics from logits. Pure math -- no API, no side effects.

    Stateful: retains _prev_phi across calls for cross-batch continuity
    at the sequence boundary (fix L3).

    Call probe.measure(logits, token_ids) on every training batch.
    Returns a CoherenceSnapshot with all metrics.
    """

    def __init__(self, vocab_size: int, top_k: int = 10) -> None:
        if vocab_size < 1:
            raise ValueError(f"vocab_size must be >= 1, got {vocab_size}")
        self.vocab_size = vocab_size
        self.top_k = max(1, top_k)
        self.log_vocab = math.log(vocab_size)
        self._prev_phi: Optional[float] = None

    def measure(
        self,
        logits,  # numpy array or torch tensor, shape [T, V]
        token_ids,  # numpy array or torch tensor, shape [T]
        step: int = 0,
        advantages=None,  # optional [T] advantage values from GRPO
    ) -> CoherenceSnapshot:
        """
        Measure the coherence state from a single generation sequence's logits.

        Accepts numpy arrays or torch tensors -- converts internally.
        Returns CoherenceSnapshot (backward compat). Use measure_trace() for
        the full CoherenceTrace primitive.
        """
        trace = self.measure_trace(logits, token_ids, step=step)
        return trace.to_snapshot(advantages=advantages)

    def measure_trace(
        self,
        logits,  # numpy array or torch tensor, shape [T, V]
        token_ids,  # numpy array or torch tensor, shape [T]
        step: int = 0,
        sample_idx: int = 0,
    ) -> "CoherenceTrace":
        """
        Measure the coherence field from a single generation sequence's logits.

        Returns the full CoherenceTrace — the foundational CARL primitive.
        All metrics (phi, cloud quality, coherence, discontinuity) derive from
        the per-token arrays in the trace.

        Accepts numpy arrays or torch tensors -- converts internally.
        Stateful: updates _prev_phi for cross-batch continuity.
        """
        from .coherence_trace import CoherenceTrace

        # Convert torch tensors to numpy
        if hasattr(logits, "detach"):
            logits = logits.detach().cpu().float().numpy()
            token_ids = token_ids.detach().cpu().numpy()

        trace = CoherenceTrace.from_logits(
            logits=logits,
            token_ids=token_ids,
            step=step,
            sample_idx=sample_idx,
            prev_phi=self._prev_phi,
        )

        # Update cross-batch state
        if trace.n_tokens > 0:
            self._prev_phi = float(trace.phi[-1])

        return trace

    # ------------------------------------------------------------------
    # Multi-layer residual-stream probe (opt-in, SEM-004)
    # ------------------------------------------------------------------

    def measure_multi_layer(
        self,
        hidden_states: List[Any],
        logits: Any,
        token_ids: Any,
        *,
        step: int = 0,
        sample_idx: int = 0,
        include_attention_entropy: bool = False,
        attentions: Optional[List[Any]] = None,
    ) -> "LayeredTrace":
        """Measure coherence across the residual stream, not just at the output.

        - Per-layer residual coherence: ``cos(h_l, h_{l-1})`` summarized over
          tokens. High cosine == tight residual flow (momentum preserved).
          Low cosine == layer-wise correction, which often precedes
          crystallization.
        - Per-layer attention entropy (optional, requires ``attentions``):
          each layer's softmax over attention weights, averaged over heads
          and tokens.
        - Final-logit trace is still computed via the existing fast path
          for consumer parity.

        Args:
            hidden_states: list of ``[T, D]`` arrays/tensors, one per residual
                layer. Must contain at least 2 layers for a pair to exist.
                Torch tensors are auto-detached/converted.
            logits: ``[T, V]`` final-layer logits (numpy or torch).
            token_ids: ``[T]`` selected token indices (numpy or torch).
            step: Training step number (forwarded to the final trace).
            sample_idx: Index within the batch (forwarded to the final trace).
            include_attention_entropy: If True, compute per-layer attention
                entropy. Requires ``attentions`` to be not None, else ignored.
            attentions: Optional list of attention-weight arrays/tensors,
                one per layer. Accepted shapes (flexible; averaged down to
                a single scalar per layer):
                  ``[H, T, T]``, ``[T, H, T]``, ``[T, T]``,
                  or any shape whose last axis is the attention distribution.

        Returns:
            ``LayeredTrace`` bundling the per-layer stats with the ordinary
            ``CoherenceTrace`` output.
        """
        import numpy as np

        from .coherence_trace import LayeredTrace

        def _to_numpy(x: Any) -> Any:
            if hasattr(x, "detach"):
                return x.detach().cpu().float().numpy()
            return np.asarray(x)

        if not isinstance(hidden_states, list):
            raise TypeError(
                "hidden_states must be a list of per-layer [T, D] arrays/tensors"
            )
        if len(hidden_states) < 2:
            raise ValueError(
                "measure_multi_layer requires at least 2 residual layers "
                f"to form an adjacent pair; got {len(hidden_states)}"
            )

        # Compute the fast-path final trace. measure_trace handles
        # numpy/torch conversion internally and updates _prev_phi.
        final = self.measure_trace(
            logits=logits,
            token_ids=token_ids,
            step=step,
            sample_idx=sample_idx,
        )

        # Per-layer residual cosine similarity.
        layer_residual_cos: List[float] = []
        prev_np = _to_numpy(hidden_states[0])
        if prev_np.ndim != 2:
            raise ValueError(
                f"hidden_states[0] must be 2-D [T, D]; got shape {prev_np.shape}"
            )
        for layer_idx in range(1, len(hidden_states)):
            cur_np = _to_numpy(hidden_states[layer_idx])
            if cur_np.ndim != 2:
                raise ValueError(
                    f"hidden_states[{layer_idx}] must be 2-D [T, D]; "
                    f"got shape {cur_np.shape}"
                )
            if cur_np.shape != prev_np.shape:
                raise ValueError(
                    f"hidden_states[{layer_idx}] shape {cur_np.shape} does not "
                    f"match previous layer shape {prev_np.shape}"
                )
            # Per-token cosine, then mean over T.
            dot = np.sum(prev_np * cur_np, axis=-1)  # [T]
            norm_prev = np.linalg.norm(prev_np, axis=-1)  # [T]
            norm_cur = np.linalg.norm(cur_np, axis=-1)  # [T]
            cos_per_token = dot / (norm_prev * norm_cur + 1e-12)  # [T]
            layer_residual_cos.append(float(np.mean(cos_per_token)))
            prev_np = cur_np

        # Per-layer attention entropy (optional).
        layer_attention_entropy: Optional[List[float]]
        if include_attention_entropy and attentions is not None:
            layer_attention_entropy = []
            for att in attentions:
                att_np = _to_numpy(att)
                # Entropy of a probability distribution along the last axis.
                # Clip/guard against zeros. Then reduce every leading axis
                # (heads / tokens / batch) down to a single scalar mean.
                log_att = np.log(att_np + 1e-12)
                per_dist_entropy = -np.sum(att_np * log_att, axis=-1)
                layer_attention_entropy.append(float(np.mean(per_dist_entropy)))
        else:
            layer_attention_entropy = None

        return LayeredTrace(
            final=final,
            layer_residual_cos=layer_residual_cos,
            layer_attention_entropy=layer_attention_entropy,
            n_layers=len(layer_residual_cos),
        )

    def measure_from_entropy(
        self,
        entropy,  # numpy or torch, shape [T]
        selected_logprobs,  # numpy or torch, shape [T]
        step: int = 0,
        sample_idx: int = 0,
    ) -> "CoherenceTrace":
        """
        Construct CoherenceTrace from TRL's pre-computed entropy.

        This is the efficient path — no second forward pass needed.
        TRL already computed per-token entropy and selected logprobs
        during its training forward pass. We reuse them directly.
        """
        from .coherence_trace import CoherenceTrace

        # Convert torch tensors to numpy
        if hasattr(entropy, "detach"):
            entropy = entropy.detach().cpu().float().numpy()
            selected_logprobs = selected_logprobs.detach().cpu().float().numpy()

        trace = CoherenceTrace.from_entropy(
            entropy=entropy,
            selected_logprobs=selected_logprobs,
            vocab_size=self.vocab_size,
            step=step,
            sample_idx=sample_idx,
            prev_phi=self._prev_phi,
        )

        if trace.n_tokens > 0:
            self._prev_phi = float(trace.phi[-1])

        return trace
