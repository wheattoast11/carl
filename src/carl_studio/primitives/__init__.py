"""Backwards-compatibility shim for `carl_studio.primitives.*`.

The primitive math moved to the standalone `carl-core` package in v0.4.0.
This shim re-exports the old surface for any downstream consumer that hasn't
migrated yet. All known consumers (the private `resonance` package) have
been migrated to `carl_core.*` directly as of 2026-04-18. This shim will be
removed in v0.5.0.

Preferred (v0.4.0+):
    from carl_core.coherence_trace import CoherenceTrace
    from carl_core.constants import KAPPA

Deprecated (emits DeprecationWarning on first import):
    from carl_studio.primitives.coherence_trace import CoherenceTrace
    from carl_studio.primitives.constants import KAPPA
"""
from __future__ import annotations

import warnings

warnings.warn(
    "carl_studio.primitives is a compatibility shim and will be removed in v0.5.0. "
    "Import from carl_core.* directly instead.",
    DeprecationWarning,
    stacklevel=2,
)

from carl_core.coherence_observer import CoherenceObserver
from carl_core.coherence_probe import CoherenceProbe, CoherenceSnapshot
from carl_core.coherence_trace import CoherenceTrace, select_traces
from carl_core.constants import DEFECT_THRESHOLD, KAPPA, SIGMA, T_STAR
from carl_core.frame_buffer import FrameBuffer, FrameRecord
from carl_core.math import compute_phi

__all__ = [
    "CoherenceObserver",
    "CoherenceProbe",
    "CoherenceSnapshot",
    "CoherenceTrace",
    "DEFECT_THRESHOLD",
    "FrameBuffer",
    "FrameRecord",
    "KAPPA",
    "SIGMA",
    "T_STAR",
    "compute_phi",
    "select_traces",
]
