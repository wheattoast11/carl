"""Compat shim: `carl_studio.primitives.math` -> `carl_core.math`."""
from carl_core.math import *  # noqa: F401, F403
from carl_core.math import compute_phi

__all__ = ["compute_phi"]
