"""Cloud quality reward: P(selected) * (1 - H/H_max).

Takes logits [T, V] + token_ids [T], returns mean cloud quality (scalar).
"""

from __future__ import annotations

import numpy as np

from carl_studio.primitives.math import compute_phi


def cloud_quality_reward(logits: np.ndarray, token_ids: np.ndarray) -> float:
    """Compute mean cloud quality from logits and selected tokens.

    Cloud quality per token = P(selected_token) * Phi
    where Phi = 1 - H/H_max (order parameter).

    Args:
        logits: [T, V] raw logits array.
        token_ids: [T] selected token indices.

    Returns:
        Mean cloud quality score in [0, 1].
    """
    T, V = logits.shape
    if T == 0:
        return 0.0

    phi, probs, entropy = compute_phi(logits)

    # P(selected) per token
    selected_probs = probs[np.arange(T), token_ids]  # [T]

    # Cloud quality = P(selected) * (1 - H/H_max) = P(selected) * phi
    cloud_quality = selected_probs * phi  # [T]

    return float(np.mean(cloud_quality))
