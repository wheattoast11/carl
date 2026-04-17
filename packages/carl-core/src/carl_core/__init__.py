"""carl-core: coherence math primitives for CARL.

This package is the foundation of the CARL stack. Every metric is a reduction
of the per-token CoherenceTrace field defined here. Zero training dependencies,
zero network calls — pure math + a typed interaction trace primitive.
"""
from __future__ import annotations

__version__ = "0.1.0"

from carl_core.constants import KAPPA, SIGMA, DEFECT_THRESHOLD, T_STAR
from carl_core.coherence_trace import CoherenceTrace, select_traces
from carl_core.coherence_probe import CoherenceProbe, CoherenceSnapshot
from carl_core.coherence_observer import CoherenceObserver
from carl_core.frame_buffer import FrameBuffer, FrameRecord
from carl_core.math import compute_phi
from carl_core.interaction import ActionType, InteractionChain, Step

__all__ = [
    "KAPPA",
    "SIGMA",
    "DEFECT_THRESHOLD",
    "T_STAR",
    "CoherenceTrace",
    "select_traces",
    "CoherenceProbe",
    "CoherenceSnapshot",
    "CoherenceObserver",
    "FrameBuffer",
    "FrameRecord",
    "compute_phi",
    "ActionType",
    "InteractionChain",
    "Step",
    "__version__",
]
