"""carl-core: coherence math primitives for CARL.

This package is the foundation of the CARL stack. Every metric is a reduction
of the per-token CoherenceTrace field defined here. Zero training dependencies,
zero network calls — pure math + a typed interaction trace primitive.
"""
from __future__ import annotations

__version__ = "0.2.0"

from carl_core.constants import KAPPA, PHI, SIGMA, DEFECT_THRESHOLD, T_STAR
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
from carl_core.presence import PresenceReport, compose_presence_report, success_rate_probe
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
from carl_core.eml import (
    CLAMP_X,
    EPS,
    MAX_DEPTH,
    EMLNode,
    EMLOp,
    EMLTree,
    eml,
    eml_array,
    eml_scalar_reward,
)
from carl_core.resonant import Resonant, compose_resonants, make_resonant
from carl_core.signing import (
    MIN_SECRET_LEN,
    SIG_LEN,
    sign_platform_countersig,
    sign_tree_software,
    verify_platform_countersig,
    verify_software_signature,
)
from carl_core.heartbeat import (
    ChainAppender,
    GradientFn,
    HeartbeatConfig,
    HeartbeatState,
    adam_step,
    detect_resonant_modes,
    heartbeat,
    initial_state,
    is_resonant,
    run_heartbeat,
)
from carl_core.optimizer_state import (
    DEFAULT_OPT_STATE_DIR,
    AdamMoments,
    OptimizerStateStore,
)
from carl_core.forecast import CoherenceForecast, ForecastMethod
from carl_core.substrate import SubstrateChannel, SubstrateState
from carl_core.anticipatory import (
    AnticipatoryGate,
    AnticipatoryPredicate,
    GateResult,
)
from carl_core.forecast_fractal import FractalForecast, phi_scale_bands
from carl_core.realizability import realizability_chain, realizability_gap
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
    "success_rate_probe",
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
    # EML primitive (Odrzywolek magma)
    "EPS",
    "MAX_DEPTH",
    "CLAMP_X",
    "EMLOp",
    "EMLNode",
    "EMLTree",
    "eml",
    "eml_array",
    "eml_scalar_reward",
    # Resonant (perceive -> cognize -> act)
    "Resonant",
    "make_resonant",
    "compose_resonants",
    # Heartbeat — CARL standing-wave loop
    "HeartbeatState",
    "HeartbeatConfig",
    "GradientFn",
    "ChainAppender",
    "adam_step",
    "heartbeat",
    "is_resonant",
    "run_heartbeat",
    "detect_resonant_modes",
    "initial_state",
    "KAPPA",
    "PHI",
    # Optimizer state — durable Adam (m, v)
    "DEFAULT_OPT_STATE_DIR",
    "AdamMoments",
    "OptimizerStateStore",
    # v0.19 anticipatory coherence trinity
    "CoherenceForecast",
    "ForecastMethod",
    "SubstrateChannel",
    "SubstrateState",
    "AnticipatoryGate",
    "AnticipatoryPredicate",
    "GateResult",
    "FractalForecast",
    "phi_scale_bands",
    "realizability_chain",
    "realizability_gap",
    "__version__",
]
