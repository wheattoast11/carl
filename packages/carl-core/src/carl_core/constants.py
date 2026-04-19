"""
CARL conservation law constants.

Derived in:
  - Desai, "Bounded Informational Time Crystals" (2026), DOI: 10.5281/zenodo.18906944
  - Desai, "Semantic Realizability" (2026), DOI: 10.5281/zenodo.18992031

These are NOT tuning parameters. They are mathematical constants from the
conservation law governing information organization in attention-coupled systems.
"""

from __future__ import annotations

KAPPA = 64 / 3          # = 21.333... -- conservation constant (2^6 / 3)
SIGMA = 3 / 16          # = 0.1875    -- semantic quantum
DEFECT_THRESHOLD = 0.03  # minimum |delta_phi| for discontinuity detection


def T_STAR(d: int) -> int:
    """Decompression boundary from conservation law T* = kappa * d."""
    return int(KAPPA * d)


# Kuramoto-R phase boundaries for PhaseAdaptiveCARLReward
# -------------------------------------------------------
# Empirical thresholds for classifying training-run synchronization phase from
# the instantaneous Kuramoto order parameter R in [0, 1]. Calibrated against
# coherence trajectories across GRPO ablations (see docs/phase_thresholds.md
# for derivation). These are NOT derived from KAPPA/SIGMA conservation theory
# -- they are empirical phase-boundary markers that should be reported in any
# methods section that uses PhaseAdaptiveCARLReward.

KURAMOTO_R_GASEOUS_MAX: float = 0.30
"""Upper bound of the gaseous phase. R < 0.30 -> gaseous (low synchronization)."""

KURAMOTO_R_LIQUID_MAX: float = 0.70
"""Upper bound of the liquid phase. 0.30 <= R < 0.70 -> liquid.
R >= 0.70 -> crystalline (high synchronization, phase-locked)."""

# Per-phase reward-weight profiles for PhaseAdaptiveCARLReward.
# Tuple order: (w_multiscale, w_cloud, w_discontinuity). Each sums to 1.0.
# Rationale:
#   - Gaseous phase: reward COMMITMENT (high w_disc) to nudge model toward
#     phase nucleation. Low w_mc because multiscale coherence is noise at low R.
#   - Liquid phase: BALANCED profile -- the model is mid-transition, no single
#     observable dominates.
#   - Crystalline phase: reward STABILITY (high w_mc) to preserve the
#     crystallized attractor. Low w_disc so the model doesn't thrash out
#     of the basin.
PHASE_WEIGHTS_GASEOUS:     tuple[float, float, float] = (0.20, 0.30, 0.50)
PHASE_WEIGHTS_LIQUID:      tuple[float, float, float] = (0.40, 0.30, 0.30)
PHASE_WEIGHTS_CRYSTALLINE: tuple[float, float, float] = (0.60, 0.30, 0.10)
