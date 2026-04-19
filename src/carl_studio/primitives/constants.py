"""Compat shim: `carl_studio.primitives.constants` -> `carl_core.constants`.

Use `from carl_core.constants import KAPPA, SIGMA, ...` in new code.
"""
from carl_core.constants import *  # noqa: F401, F403
from carl_core.constants import DEFECT_THRESHOLD, KAPPA, SIGMA, T_STAR

__all__ = ["DEFECT_THRESHOLD", "KAPPA", "SIGMA", "T_STAR"]
