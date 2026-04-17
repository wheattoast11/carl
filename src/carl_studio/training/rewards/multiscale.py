"""Multi-scale coherence reward via 2^j block decomposition.

Compromise weighting w_j = 2^(j/2).
Scale coherence = 1 - mean(block_stds), CLAMPED to [0, 1] (fix L4).
Returns weighted average across scales.
"""

from __future__ import annotations

import math

import numpy as np

from carl_core.math import compute_phi


def multiscale_coherence_reward(logits: np.ndarray, token_ids: np.ndarray) -> float:
    """Compute multi-scale coherence from logits.

    For each dyadic scale j, partitions the order-parameter sequence Phi
    into blocks of size 2^j, computes per-block std, then
    scale_coherence_j = clamp(1 - mean(block_stds), 0, 1).

    Scales are weighted by w_j = 2^(j/2) (compromise weighting).

    Args:
        logits: [T, V] raw logits array.
        token_ids: [T] selected token indices (unused, kept for API symmetry).

    Returns:
        Weighted average coherence in [0, 1]. Falls back to 0.5 if T < 1.
    """
    T, V = logits.shape
    if T < 1:
        return 0.5

    phi, probs, entropy = compute_phi(logits)

    # Multi-scale decomposition
    N_max = int(math.log2(max(T, 1)))
    total_weight = 0.0
    weighted_sum = 0.0

    for j in range(min(N_max + 1, 16)):
        block_size = 2 ** j
        if block_size > T:
            break
        n_blocks = T // block_size
        trimmed = phi[:n_blocks * block_size].reshape(n_blocks, block_size)
        block_stds = np.std(trimmed, axis=1)
        # Fix L4: clamp to [0, 1]
        coherence_j = float(max(0.0, min(1.0, 1.0 - float(np.mean(block_stds)))))
        w_j = 2 ** (j / 2)
        weighted_sum += w_j * coherence_j
        total_weight += w_j

    if total_weight <= 0:
        return 0.5

    return float(weighted_sum / total_weight)
