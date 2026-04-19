"""Atropos environment adapter — inject CARL rewards into Atropos rollouts."""
from __future__ import annotations
from typing import Any
import math
import numpy as np

from carl_core.constants import DEFECT_THRESHOLD


class AtroposAdapter:
    """
    Adapter between Atropos rollout API and CARL coherence rewards.

    Atropos handles: rollout generation via vLLM, environment scoring
    CARL handles: coherence-aware reward augmentation, observability

    Integration pattern:
    1. Atropos generates rollouts with logprobs via vLLM
    2. This adapter extracts logprobs -> numpy logits
    3. Computes CARL coherence metrics (Phi, cloud quality, discontinuities)
    4. Augments Atropos rewards with coherence bonus
    5. Returns augmented rollouts to Atropos trainer

    Usage:
        adapter = AtroposAdapter(carl_weight=0.3)
        augmented = adapter.augment_rewards(rollouts)
    """

    def __init__(
        self,
        carl_weight: float = 0.3,
        vocab_size: int = 128000,
    ):
        self.carl_weight = carl_weight
        self.vocab_size = vocab_size
        self.log_vocab = math.log(vocab_size)

    def compute_coherence_bonus(
        self,
        logprobs: list[float],
        token_ids: list[int],
    ) -> float:
        """Compute CARL coherence bonus from logprobs.

        Args:
            logprobs: Per-token log probabilities from vLLM
            token_ids: Selected token IDs

        Returns:
            Scalar coherence bonus in [0, 1]
        """
        if not logprobs or len(logprobs) < 2:
            return 0.0

        probs = np.exp(np.array(logprobs, dtype=np.float64))
        probs = np.clip(probs, 1e-10, 1.0)

        # Order parameter from token-level probabilities
        # When we only have P(selected), Phi ~= -log(P) normalized
        surprisal = -np.array(logprobs, dtype=np.float64)
        phi = 1.0 - np.clip(surprisal / self.log_vocab, 0, 1)

        # Cloud quality: P(selected) * Phi
        cloud_quality = float(np.mean(probs * phi))

        # Discontinuity detection
        if len(phi) > 1:
            delta_phi = np.diff(phi)
            commitments = int(np.sum(delta_phi > DEFECT_THRESHOLD))
            dissolutions = int(np.sum(delta_phi < -DEFECT_THRESHOLD))
            n_disc = commitments + dissolutions

            if n_disc > 0:
                disc_score = commitments / n_disc
            else:
                disc_score = 0.5
        else:
            disc_score = 0.5

        # Composite: 60% cloud quality + 40% discontinuity score
        # (multiscale requires full logits, which logprobs don't provide)
        bonus = 0.6 * cloud_quality + 0.4 * disc_score
        return float(np.clip(bonus, 0, 1))

    def augment_rewards(
        self,
        rollouts: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Augment Atropos rollout rewards with CARL coherence bonus.

        Each rollout dict should have:
            - "reward": float (original Atropos reward)
            - "logprobs": list[float] (per-token log probabilities)
            - "token_ids": list[int] (selected token IDs)
        """
        for rollout in rollouts:
            logprobs = rollout.get("logprobs", [])
            token_ids = rollout.get("token_ids", [])
            original_reward = rollout.get("reward", 0.0)

            carl_bonus = self.compute_coherence_bonus(logprobs, token_ids)

            # Weighted combination
            rollout["reward"] = (
                (1 - self.carl_weight) * original_reward
                + self.carl_weight * carl_bonus
            )
            rollout["carl_bonus"] = carl_bonus

        return rollouts
