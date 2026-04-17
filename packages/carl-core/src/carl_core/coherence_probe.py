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
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


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
