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
from carl_core.errors import (
    CARLError,
    ConfigError,
    ValidationError,
    CredentialError,
    NetworkError,
    BudgetError,
    PermissionError,
    CARLTimeoutError,
)
from carl_core.hashing import canonical_json, content_hash, content_hash_bytes
from carl_core.retry import (
    CircuitBreaker,
    CircuitState,
    RetryPolicy,
    async_retry,
    poll,
    retry,
)
from carl_core.safepath import PathEscape, SandboxedPath, safe_resolve, within
from carl_core.tier import (
    FEATURE_TIERS,
    Tier,
    TierGateError,
    feature_tier,
    tier_allows,
)

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
    "CARLError",
    "ConfigError",
    "ValidationError",
    "CredentialError",
    "NetworkError",
    "BudgetError",
    "PermissionError",
    "CARLTimeoutError",
    "canonical_json",
    "content_hash",
    "content_hash_bytes",
    "CircuitBreaker",
    "CircuitState",
    "RetryPolicy",
    "async_retry",
    "poll",
    "retry",
    "PathEscape",
    "SandboxedPath",
    "safe_resolve",
    "within",
    "Tier",
    "FEATURE_TIERS",
    "TierGateError",
    "feature_tier",
    "tier_allows",
    "__version__",
]
