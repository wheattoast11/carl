"""CoherenceTrace — the per-token coherence field.

This is the foundational primitive of CARL. Every CARL metric
(phi, cloud quality, multiscale coherence, discontinuity score)
is a reduction of this field. We stop reducing to scalars and
keep the field.

Two construction paths:
  from_logits() — compute from raw model logits (current CARL path)
  from_entropy() — compute from TRL's pre-computed entropy tensor
                   (eliminates double forward pass)

Both produce the same internal representation: per-token arrays
of phi, entropy, selected_prob, delta_phi. All CARL metrics are
derived on demand from these arrays.

Constants (from the conservation law):
    KAPPA = 64/3   (conservation constant)
    SIGMA = 3/16   (semantic quantum)
    DEFECT_THRESHOLD = 0.03
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from .constants import DEFECT_THRESHOLD, SIGMA


@dataclass(frozen=False)
class CoherenceTrace:
    """Per-token coherence field for a single completion.

    The field arrays are the source of truth. All metrics derive from them.
    """

    # --- The field (per-token arrays) ---
    phi: np.ndarray              # [T] order parameter, 0=uniform, 1=delta
    entropy: np.ndarray          # [T] Shannon entropy in nats
    selected_prob: np.ndarray    # [T] P(chosen token) at each position
    delta_phi: np.ndarray        # [T-1] Phi discontinuities between adjacent tokens

    # --- Metadata ---
    vocab_size: int
    n_tokens: int
    step: int = 0
    sample_idx: int = 0

    # --- Cached derived metrics (computed lazily) ---
    _cache: dict = field(default_factory=dict, repr=False, compare=False)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_logits(
        cls,
        logits: np.ndarray,
        token_ids: np.ndarray,
        step: int = 0,
        sample_idx: int = 0,
        prev_phi: Optional[float] = None,
    ) -> "CoherenceTrace":
        """Construct from raw model logits [T, V] and selected token_ids [T].

        This is the full-information path. Computes softmax, entropy, phi,
        and selected_prob from the complete logit distribution.

        Args:
            logits: [T, V] raw logits (numpy).
            token_ids: [T] selected token indices (numpy).
            step: Training step number.
            sample_idx: Index within the batch.
            prev_phi: Last phi from prior trace (for cross-batch continuity).
        """
        T, V = logits.shape

        # Softmax (numerically stable)
        logits_shifted = logits - logits.max(axis=-1, keepdims=True)
        exp_logits = np.exp(logits_shifted)
        probs = exp_logits / exp_logits.sum(axis=-1, keepdims=True)  # [T, V]

        # Per-token entropy (nats)
        log_probs = np.log(probs + 1e-10)
        entropy = -np.sum(probs * log_probs, axis=-1)  # [T]

        # Order parameter
        log_vocab = math.log(V)
        phi = 1.0 - (entropy / log_vocab)  # [T]

        # Selected token probability
        selected_prob = probs[np.arange(T), token_ids]  # [T]

        # Discontinuities
        delta_phi_internal = np.diff(phi)  # [T-1]
        if prev_phi is not None and T > 0:
            boundary = np.array([phi[0] - prev_phi])
            delta_phi = np.concatenate([boundary, delta_phi_internal])
        else:
            delta_phi = delta_phi_internal

        return cls(
            phi=phi,
            entropy=entropy,
            selected_prob=selected_prob,
            delta_phi=delta_phi,
            vocab_size=V,
            n_tokens=T,
            step=step,
            sample_idx=sample_idx,
        )

    @classmethod
    def from_entropy(
        cls,
        entropy: np.ndarray,
        selected_logprobs: np.ndarray,
        vocab_size: int,
        step: int = 0,
        sample_idx: int = 0,
        prev_phi: Optional[float] = None,
    ) -> "CoherenceTrace":
        """Construct from TRL's pre-computed entropy and selected logprobs.

        This is the efficient path — reuses TRL's forward pass output
        instead of running a second forward pass.

        Args:
            entropy: [T] per-token Shannon entropy in nats (from TRL).
            selected_logprobs: [T] log P(selected token) per position (from TRL).
            vocab_size: Model vocabulary size (for phi normalization).
            step: Training step number.
            sample_idx: Index within the batch.
            prev_phi: Last phi from prior trace (for cross-batch continuity).
        """
        T = len(entropy)
        log_vocab = math.log(vocab_size)

        phi = 1.0 - (entropy / log_vocab)  # [T]
        selected_prob = np.exp(selected_logprobs)  # [T]

        delta_phi_internal = np.diff(phi)
        if prev_phi is not None and T > 0:
            boundary = np.array([phi[0] - prev_phi])
            delta_phi = np.concatenate([boundary, delta_phi_internal])
        else:
            delta_phi = delta_phi_internal

        return cls(
            phi=phi,
            entropy=entropy,
            selected_prob=selected_prob,
            delta_phi=delta_phi,
            vocab_size=vocab_size,
            n_tokens=T,
            step=step,
            sample_idx=sample_idx,
        )

    # ------------------------------------------------------------------
    # Derived CARL metrics (lazy-computed, cached)
    # ------------------------------------------------------------------

    @property
    def phi_mean(self) -> float:
        if "phi_mean" not in self._cache:
            self._cache["phi_mean"] = float(np.mean(self.phi))
        return self._cache["phi_mean"]

    @property
    def phi_std(self) -> float:
        if "phi_std" not in self._cache:
            self._cache["phi_std"] = float(np.std(self.phi))
        return self._cache["phi_std"]

    @property
    def entropy_mean(self) -> float:
        if "entropy_mean" not in self._cache:
            self._cache["entropy_mean"] = float(np.mean(self.entropy))
        return self._cache["entropy_mean"]

    @property
    def cloud_quality(self) -> float:
        """Mean cloud quality = mean(P(selected) * Phi). Range [0, 1]."""
        if "cloud_quality" not in self._cache:
            cq = self.selected_prob * self.phi  # [T]
            self._cache["cloud_quality"] = float(np.mean(cq))
        return self._cache["cloud_quality"]

    @property
    def multiscale_coherence(self) -> float:
        """Weighted multi-scale coherence across dyadic scales.

        Scale coherence_j = clamp(1 - mean(block_stds), 0, 1).
        Compromise weighting: w_j = 2^(j/2).
        """
        if "multiscale_coherence" not in self._cache:
            T = self.n_tokens
            if T < 1:
                self._cache["multiscale_coherence"] = 0.5
            else:
                N_max = int(math.log2(max(T, 1)))
                total_weight = 0.0
                weighted_sum = 0.0
                for j in range(min(N_max + 1, 16)):
                    block_size = 2 ** j
                    if block_size > T:
                        break
                    n_blocks = T // block_size
                    trimmed = self.phi[:n_blocks * block_size].reshape(n_blocks, block_size)
                    block_stds = np.std(trimmed, axis=1)
                    coherence_j = max(0.0, min(1.0, 1.0 - float(np.mean(block_stds))))
                    w_j = 2 ** (j / 2)
                    weighted_sum += w_j * coherence_j
                    total_weight += w_j
                self._cache["multiscale_coherence"] = (
                    float(weighted_sum / total_weight) if total_weight > 0 else 0.5
                )
        return self._cache["multiscale_coherence"]

    @property
    def scale_coherence(self) -> Dict[int, float]:
        """Per-scale coherence values (for detailed logging)."""
        if "scale_coherence" not in self._cache:
            T = self.n_tokens
            result: Dict[int, float] = {}
            if T >= 1:
                N_max = int(math.log2(max(T, 1)))
                for j in range(min(N_max + 1, 16)):
                    block_size = 2 ** j
                    if block_size > T:
                        break
                    n_blocks = T // block_size
                    trimmed = self.phi[:n_blocks * block_size].reshape(n_blocks, block_size)
                    block_stds = np.std(trimmed, axis=1)
                    result[j] = max(0.0, min(1.0, 1.0 - float(np.mean(block_stds))))
            self._cache["scale_coherence"] = result
        return self._cache["scale_coherence"]

    @property
    def discontinuity_score(self) -> float:
        """Context-dependent defect score in [0, 1].

        Crystallization after low-order (phi < 0.5) = 0.8 (commitment).
        Melting at high-order (phi > 0.7) = 0.2 (dissolution).
        Returns 0.5 if no defects.
        """
        if "discontinuity_score" not in self._cache:
            if len(self.delta_phi) < 1:
                self._cache["discontinuity_score"] = 0.5
            else:
                defect_scores: list[float] = []
                for k in range(len(self.delta_phi)):
                    dp = self.delta_phi[k]
                    if abs(dp) <= DEFECT_THRESHOLD:
                        continue
                    # phi at position before the transition
                    prev_order = self.phi[k] if k < len(self.phi) else 0.5
                    if dp > DEFECT_THRESHOLD:
                        defect_scores.append(0.8 if prev_order < 0.5 else 0.3)
                    else:
                        defect_scores.append(0.2 if prev_order > 0.7 else 0.5)
                self._cache["discontinuity_score"] = (
                    float(sum(defect_scores) / len(defect_scores))
                    if defect_scores else 0.5
                )
        return self._cache["discontinuity_score"]

    @property
    def n_crystallizations(self) -> int:
        if "n_crystallizations" not in self._cache:
            self._cache["n_crystallizations"] = int(np.sum(self.delta_phi > DEFECT_THRESHOLD))
        return self._cache["n_crystallizations"]

    @property
    def n_meltings(self) -> int:
        if "n_meltings" not in self._cache:
            self._cache["n_meltings"] = int(np.sum(self.delta_phi < -DEFECT_THRESHOLD))
        return self._cache["n_meltings"]

    @property
    def n_defects(self) -> int:
        return self.n_crystallizations + self.n_meltings

    @property
    def defect_density(self) -> float:
        return self.n_defects / max(self.n_tokens, 1)

    @property
    def surprisal_mean(self) -> float:
        """Mean surprisal = mean(-log P(selected)). Annihilation energy."""
        if "surprisal_mean" not in self._cache:
            self._cache["surprisal_mean"] = float(np.mean(-np.log(self.selected_prob + 1e-10)))
        return self._cache["surprisal_mean"]

    # ------------------------------------------------------------------
    # CARL composite reward
    # ------------------------------------------------------------------

    def carl_reward(
        self,
        w_coherence: float = 0.5,
        w_cloud: float = 0.3,
        w_discontinuity: float = 0.2,
    ) -> float:
        """Compute the CARL composite reward from this trace.

        Default weights: 50% multiscale + 30% cloud + 20% discontinuity.
        All components derived from the cached field — no recomputation.
        """
        return (
            w_coherence * self.multiscale_coherence
            + w_cloud * self.cloud_quality
            + w_discontinuity * self.discontinuity_score
        )

    # ------------------------------------------------------------------
    # Resonance (Kuramoto coupling)
    # ------------------------------------------------------------------

    def kuramoto_R(self) -> float:
        """Kuramoto order parameter from the per-token Phi field.

        Half-circle mapping: theta = pi * Phi. This preserves the linear
        ordering — disorder (Phi=0) and order (Phi=1) are maximally separated
        at opposite poles (0 and pi). Full-circle mapping incorrectly treats
        them as adjacent due to 0/2pi wraparound.

        R=1: all tokens at the same coherence level (phase-locked).
        R~0: tokens at diverse coherence levels (desynchronized).
        """
        if "kuramoto_R" not in self._cache:
            if self.n_tokens < 2:
                self._cache["kuramoto_R"] = 0.0
            else:
                phases = self.phi * np.pi  # half-circle: [0,1] -> [0, pi]
                R = float(abs(np.mean(np.exp(1j * phases))))
                self._cache["kuramoto_R"] = R
        return self._cache["kuramoto_R"]

    def kuramoto_R_at(self, positions: np.ndarray) -> float:
        """Kuramoto R computed only at specific token positions.

        Measures phase coherence at targeted locations (e.g., tool-call
        boundaries). High R_at = consistent commitment level at all
        interaction points.

        Args:
            positions: integer indices into the phi array.
        """
        if len(positions) < 2:
            return 0.0
        valid = positions[positions < self.n_tokens]
        if len(valid) < 2:
            return 0.0
        phi_at = self.phi[valid]
        phases = phi_at * np.pi
        return float(abs(np.mean(np.exp(1j * phases))))

    def transition_sharpness(self, positions: np.ndarray) -> float:
        """Mean |delta_phi| at specified positions.

        Measures how sharply the model commits/dissolves at specific
        boundaries. High sharpness = clean phase transitions at
        interaction points.

        Args:
            positions: integer indices into the delta_phi array.
        """
        if len(positions) == 0 or len(self.delta_phi) == 0:
            return 0.0
        valid = positions[positions < len(self.delta_phi)]
        if len(valid) == 0:
            return 0.0
        return float(np.mean(np.abs(self.delta_phi[valid])))

    def crystallization_positions(self) -> np.ndarray:
        """Token positions where crystallization events occur (delta_phi > threshold)."""
        return np.where(self.delta_phi > DEFECT_THRESHOLD)[0]

    def melting_positions(self) -> np.ndarray:
        """Token positions where melting events occur (delta_phi < -threshold)."""
        return np.where(self.delta_phi < -DEFECT_THRESHOLD)[0]

    def segment_phi_means(self, boundaries: list[int]) -> list[float]:
        """Mean Phi per segment defined by boundary positions.

        Useful for measuring Phi progression across turns in multi-turn
        completions.

        Args:
            boundaries: sorted list of token positions marking segment starts.
                        Implicitly, the last segment runs to end of trace.

        Returns:
            List of mean Phi values, one per segment.
        """
        if not boundaries:
            return [self.phi_mean]
        means = []
        starts = list(boundaries) + [self.n_tokens]
        for i in range(len(starts) - 1):
            seg = self.phi[starts[i]:starts[i + 1]]
            means.append(float(np.mean(seg)) if len(seg) > 0 else 0.0)
        return means

    # ------------------------------------------------------------------
    # Backward compatibility
    # ------------------------------------------------------------------

    def to_snapshot(self, advantages: Optional[np.ndarray] = None) -> Any:
        """Convert to CoherenceSnapshot for backward compatibility.

        Lazy import to avoid circular dependency.
        """
        from .coherence_probe import CoherenceSnapshot

        adv_mean: Optional[float] = None
        adv_above_sigma: Optional[float] = None
        if advantages is not None:
            adv_mean = float(np.mean(np.abs(advantages)))
            adv_above_sigma = float(np.mean(np.abs(advantages) > SIGMA))

        return CoherenceSnapshot(
            step=self.step,
            n_tokens=self.n_tokens,
            phi_mean=self.phi_mean,
            phi_std=self.phi_std,
            phi_trajectory=[float(x) for x in self.phi[::max(1, self.n_tokens // 64)]],
            n_defects=self.n_defects,
            n_crystallizations=self.n_crystallizations,
            n_meltings=self.n_meltings,
            defect_density=self.defect_density,
            discontinuity_score=self.discontinuity_score,
            cloud_quality_mean=self.cloud_quality,
            scale_coherence=self.scale_coherence,
            entropy_mean=self.entropy_mean,
            entropy_std=float(np.std(self.entropy)),
            surprisal_mean=self.surprisal_mean,
            surprisal_std=float(np.std(-np.log(self.selected_prob + 1e-10))),
            top_k_mass=0.0,  # Not available without full probs
            advantage_mean=adv_mean,
            advantage_above_sigma=adv_above_sigma,
        )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self, include_arrays: bool = True) -> dict:
        """Serialize to dict. Arrays as lists for JSON compatibility.

        Args:
            include_arrays: If True, include per-token arrays (phi, entropy,
                selected_prob, delta_phi). If False, only scalar summaries.
        """
        d: dict[str, Any] = {
            "step": self.step,
            "sample_idx": self.sample_idx,
            "vocab_size": self.vocab_size,
            "n_tokens": self.n_tokens,
            # Scalar summaries (always included)
            "phi_mean": self.phi_mean,
            "phi_std": self.phi_std,
            "entropy_mean": self.entropy_mean,
            "cloud_quality": self.cloud_quality,
            "multiscale_coherence": self.multiscale_coherence,
            "discontinuity_score": self.discontinuity_score,
            "carl_reward": self.carl_reward(),
            "n_crystallizations": self.n_crystallizations,
            "n_meltings": self.n_meltings,
            "defect_density": self.defect_density,
            "surprisal_mean": self.surprisal_mean,
        }
        if include_arrays:
            d["phi"] = self.phi.tolist()
            d["entropy"] = self.entropy.tolist()
            d["selected_prob"] = self.selected_prob.tolist()
            d["delta_phi"] = self.delta_phi.tolist()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "CoherenceTrace":
        """Reconstruct from serialized dict."""
        return cls(
            phi=np.array(d["phi"], dtype=np.float64),
            entropy=np.array(d["entropy"], dtype=np.float64),
            selected_prob=np.array(d["selected_prob"], dtype=np.float64),
            delta_phi=np.array(d["delta_phi"], dtype=np.float64),
            vocab_size=d["vocab_size"],
            n_tokens=d["n_tokens"],
            step=d.get("step", 0),
            sample_idx=d.get("sample_idx", 0),
        )

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    SPARK_CHARS = " \u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588"

    def sparkline(self, width: int = 60) -> str:
        """ASCII sparkline of the Phi field, downsampled to width."""
        if self.n_tokens == 0:
            return ""
        # Downsample phi to width
        indices = np.linspace(0, self.n_tokens - 1, min(width, self.n_tokens), dtype=int)
        values = self.phi[indices]

        # Map [0, 1] to spark chars
        chars = self.SPARK_CHARS
        n_chars = len(chars) - 1
        result = []
        for v in values:
            idx = int(max(0, min(n_chars, v * n_chars)))
            result.append(chars[idx])
        return "".join(result)

    def entropy_sparkline(self, width: int = 60) -> str:
        """ASCII sparkline of the entropy field, auto-scaled."""
        if self.n_tokens == 0:
            return ""
        indices = np.linspace(0, self.n_tokens - 1, min(width, self.n_tokens), dtype=int)
        values = self.entropy[indices]

        lo, hi = float(np.min(values)), float(np.max(values))
        span = hi - lo if hi > lo else 1.0

        chars = self.SPARK_CHARS
        n_chars = len(chars) - 1
        result = []
        for v in values:
            normalized = (v - lo) / span
            idx = int(max(0, min(n_chars, normalized * n_chars)))
            result.append(chars[idx])
        return "".join(result)

    def defect_map(self, width: int = 60) -> str:
        """ASCII map of crystallization/melting events.

        '.' = no defect, '+' = crystallization, '-' = melting.
        """
        if len(self.delta_phi) == 0:
            return ""
        indices = np.linspace(0, len(self.delta_phi) - 1, min(width, len(self.delta_phi)), dtype=int)
        result = []
        for i in indices:
            dp = self.delta_phi[i]
            if dp > DEFECT_THRESHOLD:
                result.append("+")
            elif dp < -DEFECT_THRESHOLD:
                result.append("-")
            else:
                result.append("\u00b7")  # middle dot
        return "".join(result)

    def __repr__(self) -> str:
        return (
            f"CoherenceTrace(step={self.step}, T={self.n_tokens}, "
            f"\u03a6={self.phi_mean:.3f}\u00b1{self.phi_std:.3f}, "
            f"CQ={self.cloud_quality:.3f}, "
            f"C={self.multiscale_coherence:.3f}, "
            f"D={self.discontinuity_score:.3f}, "
            f"R={self.carl_reward():.4f})"
        )


# ---------------------------------------------------------------------------
# Batch-level trace utilities
# ---------------------------------------------------------------------------

def select_traces(
    traces: list[CoherenceTrace],
    k: int = 4,
) -> list[CoherenceTrace]:
    """Select k representative traces from a batch.

    Returns: best (highest carl_reward), worst (lowest), median, and random.
    If batch is smaller than k, returns all.
    """
    if len(traces) <= k:
        return traces

    rewards = [(t.carl_reward(), i, t) for i, t in enumerate(traces)]
    rewards.sort(key=lambda x: x[0])

    selected_indices = set()
    result = []

    # Worst
    result.append(rewards[0][2])
    selected_indices.add(rewards[0][1])

    # Best
    result.append(rewards[-1][2])
    selected_indices.add(rewards[-1][1])

    # Median
    mid = len(rewards) // 2
    result.append(rewards[mid][2])
    selected_indices.add(rewards[mid][1])

    # Random (not already selected)
    remaining = [r for r in rewards if r[1] not in selected_indices]
    if remaining:
        rng = np.random.default_rng()
        chosen = remaining[rng.integers(len(remaining))]
        result.append(chosen[2])

    return result[:k]
