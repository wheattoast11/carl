"""
CARL conservation law constants.

Derived in:
  - Desai, "Bounded Informational Time Crystals" (2026), DOI: 10.5281/zenodo.18906944
  - Desai, "Semantic Realizability" (2026), DOI: 10.5281/zenodo.18992031

These are NOT tuning parameters. They are mathematical constants from the
conservation law governing information organization in attention-coupled systems.
"""

KAPPA = 64 / 3          # = 21.333... -- conservation constant (2^6 / 3)
SIGMA = 3 / 16          # = 0.1875    -- semantic quantum
DEFECT_THRESHOLD = 0.03  # minimum |delta_phi| for discontinuity detection


def T_STAR(d: int) -> int:
    """Decompression boundary from conservation law T* = kappa * d."""
    return int(KAPPA * d)
