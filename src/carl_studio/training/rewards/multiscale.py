"""Multi-scale coherence reward via 2^j block decomposition.

Compromise weighting w_j = 2^(j/2).
Scale coherence = 1 - mean(block_stds), CLAMPED to [0, 1] (fix L4).
Returns weighted average across scales.

Production hardening (WS-T3/T4):
  - NaN/inf clamping on every scale contribution (prevents optimizer runaway).
  - Logits shape guard: accepts 2D [T, V] or auto-selects last step of 3D [B, T, V].
  - Empty / degenerate batches return 0.0 with a dedupe'd warning.
  - Module-level clamp counter for per-epoch aggregation.
"""

from __future__ import annotations

import logging
import math
import threading
from typing import Any

import numpy as np

from carl_core.errors import ValidationError
from carl_core.math import compute_phi


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reward-clamp telemetry (shared across all reward functions in this package)
# ---------------------------------------------------------------------------

# Hard cap on any reward value before handing it to the optimizer.
REWARD_CLAMP_MIN: float = -100.0
REWARD_CLAMP_MAX: float = 100.0

_clamp_counter_lock = threading.Lock()
_clamp_counter: dict[str, int] = {
    "nonfinite": 0,  # NaN / +/-inf coerced to 0.0
    "overflow": 0,   # |x| > REWARD_CLAMP_MAX clipped
    "total": 0,
}


def _clamp_reward(value: float) -> float:
    """Coerce any reward scalar into the safe range.

    Steps:
      1. Convert to float. Non-convertible inputs become 0.0.
      2. Non-finite (NaN / +/-inf) -> 0.0.
      3. Clip to [REWARD_CLAMP_MIN, REWARD_CLAMP_MAX].

    Thread-safe counter increment for per-epoch telemetry.
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        with _clamp_counter_lock:
            _clamp_counter["nonfinite"] += 1
            _clamp_counter["total"] += 1
        return 0.0

    if not math.isfinite(v):
        with _clamp_counter_lock:
            _clamp_counter["nonfinite"] += 1
            _clamp_counter["total"] += 1
        return 0.0

    if v > REWARD_CLAMP_MAX:
        with _clamp_counter_lock:
            _clamp_counter["overflow"] += 1
            _clamp_counter["total"] += 1
        return REWARD_CLAMP_MAX
    if v < REWARD_CLAMP_MIN:
        with _clamp_counter_lock:
            _clamp_counter["overflow"] += 1
            _clamp_counter["total"] += 1
        return REWARD_CLAMP_MIN

    return v


def clamp_counts() -> dict[str, int]:
    """Return a snapshot of the clamp counters (for callback telemetry)."""
    with _clamp_counter_lock:
        return dict(_clamp_counter)


def reset_clamp_counts() -> None:
    """Reset the clamp counters. Called at epoch boundaries by callbacks."""
    with _clamp_counter_lock:
        _clamp_counter["nonfinite"] = 0
        _clamp_counter["overflow"] = 0
        _clamp_counter["total"] = 0


# ---------------------------------------------------------------------------
# Logits shape normalization (WS-T4)
# ---------------------------------------------------------------------------


def _normalize_logits_2d(logits: Any) -> np.ndarray:
    """Accept 2D [T, V] as-is, or collapse 3D [B, T, V] to its last batch row.

    Raises ValidationError for unsupported rank. Also validates that the
    trailing dim (vocab) is non-trivial. A 1D input is rejected — ambiguous.
    """
    arr = np.asarray(logits)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        logger.debug(
            "multiscale: collapsing 3D logits %s -> 2D last-batch slice",
            list(arr.shape),
        )
        # Take the last batch element; matches TRL's per-sample dispatch path.
        return arr[-1]
    raise ValidationError(
        "expected 2D logits [T, V] or 3D [B, T, V]",
        code="carl.logits_shape",
        context={"shape": list(arr.shape)},
    )


# Dedupe cache for the "all scales failed" warning — one line per session.
_degenerate_warned: set[str] = set()


def multiscale_coherence_reward(logits: np.ndarray, token_ids: np.ndarray) -> float:
    """Compute multi-scale coherence from logits.

    For each dyadic scale j, partitions the order-parameter sequence Phi
    into blocks of size 2^j, computes per-block std, then
    scale_coherence_j = clamp(1 - mean(block_stds), 0, 1).

    Scales are weighted by w_j = 2^(j/2) (compromise weighting).

    Args:
        logits: [T, V] raw logits array (or [B, T, V] — last batch row is used).
        token_ids: [T] selected token indices (unused, kept for API symmetry).

    Returns:
        Weighted average coherence in [0, 1]. Falls back to 0.0 when no scales
        can be computed (empty or degenerate input) and emits a single
        per-session warning per failure class.
    """
    arr = _normalize_logits_2d(logits)

    T, V = arr.shape
    if T < 1 or V < 1:
        _warn_once("empty_input", "multiscale: empty logits %r -> 0.0", arr.shape)
        return 0.0

    try:
        phi, _probs, _entropy = compute_phi(arr)
    except Exception as exc:
        _warn_once(
            "compute_phi_fail",
            "multiscale: compute_phi failed (%s) -> 0.0",
            exc,
        )
        return 0.0

    # Guard the upstream result — a model spitting extreme logits can yield
    # non-finite Phi values that would poison the optimizer.
    phi = np.asarray(phi, dtype=float)
    if not np.all(np.isfinite(phi)):
        _warn_once("phi_nonfinite", "multiscale: compute_phi returned non-finite values -> clamped")
        phi = np.nan_to_num(phi, nan=0.0, posinf=1.0, neginf=0.0)

    # Multi-scale decomposition
    N_max = int(math.log2(max(T, 1)))
    total_weight = 0.0
    weighted_sum = 0.0

    for j in range(min(N_max + 1, 16)):
        block_size = 2**j
        if block_size > T:
            break
        n_blocks = T // block_size
        if n_blocks < 1:
            continue
        trimmed = phi[: n_blocks * block_size].reshape(n_blocks, block_size)
        block_stds = np.std(trimmed, axis=1)
        mean_std = float(np.mean(block_stds))
        if not math.isfinite(mean_std):
            continue
        coherence_j = float(max(0.0, min(1.0, 1.0 - mean_std)))
        w_j = 2 ** (j / 2)
        weighted_sum += w_j * coherence_j
        total_weight += w_j

    if total_weight <= 0:
        _warn_once(
            "all_scales_fail",
            "multiscale: no valid scales (T=%d V=%d) -> 0.0",
            T,
            V,
        )
        return 0.0

    result = float(weighted_sum / total_weight)
    return _clamp_reward(result)


def _warn_once(key: str, template: str, *args: Any) -> None:
    """Emit a warning exactly once per process for the given dedupe key."""
    if key in _degenerate_warned:
        return
    _degenerate_warned.add(key)
    logger.warning(template, *args)


def _reset_warn_cache() -> None:
    """Test hook: clear the dedupe set so assertions can re-trigger warnings."""
    _degenerate_warned.clear()
