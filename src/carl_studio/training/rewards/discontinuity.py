"""Per-token discontinuity reward from delta-Phi.

Context-dependent scoring:
  - Commitment after low-order (phi < 0.5) = 0.8
  - Dissolution at high-order (phi > 0.7) = 0.2
  - Uses np.diff(phi) NOT prepend (fix L3)

Takes logits [T, V] and token_ids [T], returns scalar score.
"""

from __future__ import annotations

import numpy as np

from carl_core.constants import DEFECT_THRESHOLD
from carl_core.math import compute_phi


def discontinuity_reward(logits: np.ndarray, token_ids: np.ndarray) -> float:
    """Compute context-dependent discontinuity score from logits.

    Args:
        logits: [T, V] raw logits array.
        token_ids: [T] selected token indices (unused for phi computation,
                   kept for API consistency with cloud/multiscale).

    Returns:
        Scalar score in [0, 1]. 0.5 if no defects detected.
    """
    T, V = logits.shape
    if T < 2:
        return 0.5

    phi, probs, entropy = compute_phi(logits)

    # Fix L3: use np.diff, NOT prepend -- produces [T-1] transitions
    delta_phi = np.diff(phi)  # [T-1]

    # Context-dependent defect scoring
    defect_scores: list[float] = []
    for k in range(len(delta_phi)):
        dp = delta_phi[k]
        if abs(dp) <= DEFECT_THRESHOLD:
            continue
        prev_order = phi[k]  # phi at position before the transition
        if dp > DEFECT_THRESHOLD:  # Crystallization (commitment)
            defect_scores.append(0.8 if prev_order < 0.5 else 0.3)
        else:  # Melting (dissolution)
            defect_scores.append(0.2 if prev_order > 0.7 else 0.5)

    if not defect_scores:
        return 0.5

    return float(sum(defect_scores) / len(defect_scores))
