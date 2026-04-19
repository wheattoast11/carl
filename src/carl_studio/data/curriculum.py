"""Curriculum engine — adaptive sampling and difficulty filtering.

Implements three strategies from the GRPO literature:

1. Nemotron dynamic filtering: discard all-pass/all-fail prompts (zero gradient)
2. AceGRPO learnability: prioritize problems at ~50% solve rate
3. Cascade stage matching: easy for Stage A, medium+ for Stage B

Usage:
    curriculum = CurriculumSampler(samples)
    batch = curriculum.sample(n=64, tier=DifficultyTier.MEDIUM)

    # After training step, update pass rates:
    curriculum.update_pass_rates(sample_ids, pass_rates)

    # Adaptive: shift distribution toward harder problems:
    curriculum.advance()
"""

from __future__ import annotations

import random
from collections import defaultdict

from carl_studio.data.types import DifficultyTier, Domain, Modality, UnifiedSample


class CurriculumSampler:
    """Adaptive difficulty-aware sampler for GRPO training.

    Tracks per-sample pass rates from training rollouts and adjusts
    the sampling distribution toward the learnable frontier.
    """

    def __init__(
        self,
        samples: list[UnifiedSample],
        target_pass_rate: float = 0.5,
        tolerance: float = 0.2,
    ) -> None:
        self._samples = {s.id: s for s in samples}
        self._pass_rates: dict[str, float] = {}
        self._target = target_pass_rate
        self._tolerance = tolerance
        self._by_tier: dict[DifficultyTier, set[str]] = defaultdict(set)
        self._sample_tier: dict[str, DifficultyTier] = {}  # Reverse index for O(1) removal

        for s in samples:
            tier = s.difficulty_tier or DifficultyTier.MEDIUM
            self._by_tier[tier].add(s.id)
            self._sample_tier[s.id] = tier
            if s.difficulty_score is not None:
                self._pass_rates[s.id] = s.difficulty_score

    def sample(
        self,
        n: int,
        tier: DifficultyTier | None = None,
        domain: Domain | None = None,
        modality: Modality | None = None,
    ) -> list[UnifiedSample]:
        """Sample n records, optionally filtered by tier/domain/modality."""
        pool = list(self._samples.values())

        if tier:
            tier_ids = set(self._by_tier.get(tier, []))
            pool = [s for s in pool if s.id in tier_ids]
        if domain:
            pool = [s for s in pool if s.domain == domain]
        if modality:
            pool = [s for s in pool if s.modality == modality]

        if not pool:
            return []

        return random.sample(pool, min(n, len(pool)))

    def sample_learnable(self, n: int) -> list[UnifiedSample]:
        """Sample from the learnable frontier (Nemotron dynamic filtering).

        Selects samples where pass_rate is in [target - tolerance, target + tolerance].
        These produce maximum GRPO advantage variance.
        """
        low = self._target - self._tolerance
        high = self._target + self._tolerance

        frontier = [
            self._samples[sid]
            for sid, rate in self._pass_rates.items()
            if low <= rate <= high and sid in self._samples
        ]

        if not frontier:
            # Fall back to all non-trivial, non-intractable samples
            frontier = [
                self._samples[sid]
                for sid, rate in self._pass_rates.items()
                if 0.05 < rate < 0.95 and sid in self._samples
            ]

        if not frontier:
            return self.sample(n)

        return random.sample(frontier, min(n, len(frontier)))

    def update_pass_rates(self, results: dict[str, float]) -> None:
        """Update per-sample pass rates from training rollouts.

        Args:
            results: {sample_id: pass_rate} from the latest GRPO batch.
                     pass_rate = fraction of generations that succeeded.
        """
        for sid, rate in results.items():
            if sid not in self._samples:
                continue
            # Clamp rate to [0, 1] to preserve Pydantic constraint
            rate = max(0.0, min(1.0, rate))
            old = self._pass_rates.get(sid, rate)
            self._pass_rates[sid] = max(0.0, min(1.0, 0.3 * rate + 0.7 * old))

            # Reclassify tier (O(1) via reverse index)
            new_tier = DifficultyTier.from_pass_rate(self._pass_rates[sid])
            old_tier = self._sample_tier.get(sid)
            if old_tier and old_tier != new_tier:
                self._by_tier[old_tier].discard(sid)
            self._by_tier[new_tier].add(sid)
            self._sample_tier[sid] = new_tier

            sample = self._samples[sid]
            sample.difficulty_tier = new_tier
            sample.difficulty_score = self._pass_rates[sid]

    def advance(self, shift: float = 0.05) -> None:
        """Shift target pass rate down (toward harder problems).

        Call this when the model's overall success rate exceeds the target.
        """
        self._target = max(0.15, self._target - shift)

    def stats(self) -> dict[str, int]:
        """Return sample count per difficulty tier."""
        return {tier.value: len(ids) for tier, ids in self._by_tier.items()}

    def filter_zero_gradient(self, batch_results: dict[str, list[float]]) -> list[str]:
        """Identify samples that produced zero GRPO gradient (Nemotron filtering).

        A sample produces zero gradient when all generations scored identically
        (all-pass or all-fail). These waste compute.

        Args:
            batch_results: {sample_id: [reward_per_generation]}

        Returns:
            List of sample IDs that should be deprioritized.
        """
        zero_gradient = []
        for sid, rewards in batch_results.items():
            if len(set(rewards)) <= 1:  # All same score
                zero_gradient.append(sid)
        return zero_gradient
