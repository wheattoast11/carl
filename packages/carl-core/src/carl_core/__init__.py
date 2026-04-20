"""carl-core: coherence math primitives for CARL.

This package is the foundation of the CARL stack. Every metric is a reduction
of the per-token CoherenceTrace field defined here. Zero training dependencies,
zero network calls — pure math + a typed interaction trace primitive.
"""
from __future__ import annotations

__version__ = "0.1.0"

from carl_core.constants import KAPPA, SIGMA, DEFECT_THRESHOLD, T_STAR
from carl_core.coherence_trace import CoherenceTrace, LayeredTrace, select_traces
from carl_core.coherence_probe import (
    CARL_LAYER_PROBE_ENABLED,
    CoherenceProbe,
    CoherenceSnapshot,
)
from carl_core.coherence_observer import CoherenceObserver
from carl_core.frame_buffer import FrameBuffer, FrameRecord
from carl_core.math import compute_phi
from carl_core.interaction import ActionType, InteractionChain, Step
from carl_core.presence import PresenceReport, compose_presence_report
from carl_core.interaction_store import InteractionStore
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
from carl_core.memory import MemoryItem, MemoryLayer, MemoryStore
from carl_core.retry import (
    CircuitBreaker,
    CircuitState,
    RetryPolicy,
    async_retry,
    poll,
    retry,
)
from carl_core.resilience import BreakAndRetryStrategy, CircuitOpenError
from carl_core.safepath import PathEscape, SandboxedPath, safe_resolve, within
from carl_core.tier import (
    FEATURE_TIERS,
    Tier,
    TierGateError,
    feature_tier,
    tier_allows,
)
from carl_core.timeutil import ISO8601_Z, now_iso, now_iso_ms
from carl_core.connection import (
    VALID_TRANSITIONS,
    AsyncBaseConnection,
    BaseConnection,
    CARLConnectionError,
    ChannelCoherence,
    ConnectionAuthError,
    ConnectionBase,
    ConnectionClosedError,
    ConnectionDirection,
    ConnectionKind,
    ConnectionPolicyError,
    ConnectionRegistry,
    ConnectionScope,
    ConnectionSpec,
    ConnectionState,
    ConnectionStats,
    ConnectionTransitionError,
    ConnectionTransport,
    ConnectionTrust,
    ConnectionUnavailableError,
    can_transition,
    channel_coherence_diff,
    channel_coherence_distance,
    get_registry,
    reset_registry,
)

__all__ = [
    "KAPPA",
    "SIGMA",
    "DEFECT_THRESHOLD",
    "T_STAR",
    "CoherenceTrace",
    "LayeredTrace",
    "select_traces",
    "CARL_LAYER_PROBE_ENABLED",
    "CoherenceProbe",
    "CoherenceSnapshot",
    "CoherenceObserver",
    "FrameBuffer",
    "FrameRecord",
    "compute_phi",
    "ActionType",
    "InteractionChain",
    "InteractionStore",
    "Step",
    "PresenceReport",
    "compose_presence_report",
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
    "MemoryItem",
    "MemoryLayer",
    "MemoryStore",
    "CircuitBreaker",
    "CircuitState",
    "RetryPolicy",
    "async_retry",
    "poll",
    "retry",
    "BreakAndRetryStrategy",
    "CircuitOpenError",
    "PathEscape",
    "SandboxedPath",
    "safe_resolve",
    "within",
    "Tier",
    "FEATURE_TIERS",
    "TierGateError",
    "feature_tier",
    "tier_allows",
    # time helpers
    "ISO8601_Z",
    "now_iso",
    "now_iso_ms",
    # connection primitive
    "ConnectionSpec",
    "ConnectionScope",
    "ConnectionKind",
    "ConnectionDirection",
    "ConnectionTransport",
    "ConnectionTrust",
    "ConnectionState",
    "VALID_TRANSITIONS",
    "can_transition",
    "CARLConnectionError",
    "ConnectionUnavailableError",
    "ConnectionAuthError",
    "ConnectionTransitionError",
    "ConnectionClosedError",
    "ConnectionPolicyError",
    "BaseConnection",
    "AsyncBaseConnection",
    "ConnectionBase",
    "ConnectionStats",
    "ConnectionRegistry",
    "get_registry",
    "reset_registry",
    # channel coherence (cross-channel observable)
    "ChannelCoherence",
    "channel_coherence_diff",
    "channel_coherence_distance",
    "__version__",
]
