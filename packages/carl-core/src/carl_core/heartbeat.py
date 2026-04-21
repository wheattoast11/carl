"""Heartbeat — the CARL standing-wave loop.

This module codifies the CARL runtime loop as a pure functional program.
The loop is:

    (W_t, Φ_t, m_t, v_t, R_t, h_t) --[Adam step on eml-reward grad]--> (W_{t+1}, ...)
                                     --[gate R > τ]--> emit step to InteractionChain
                                     --[recurse]--> next tick

THEOREM (Heartbeat Standing Wave, ι Dim 3a):
    Let L(θ) = f(eml(g(θ), h(θ))) for differentiable g, h and eml(x, y) = exp(x) - ln(y).
    Let {θ_t} be the Adam iterate sequence (β1, β2, ε, lr) on L.
    If the effective Hessian H_t = ∂²L/∂θ² has its dominant eigenvalue λ_max(t) satisfying
        |λ_max(t)| < edge_band for all t along the trajectory,
    and the Kuramoto order parameter R(t) satisfies R(t) >= R_threshold,
    then the trajectory {θ_t} is a BOUNDED NON-DECAYING OSCILLATION with
    dominant angular frequency ω* such that
        T* = 2π / ω*  ≈  2π · sqrt((1 - β2) / (α² · (1 - β1)))
    i.e. a STANDING WAVE in parameter space, not damped convergence.

PROOF SKETCH (verbatim, docstring theorem):
    The Hessian of f ∘ eml ∘ (g, h) is
        ∂²L/∂θ² = f''(z) · (∂z/∂θ)(∂z/∂θ)ᵀ  +  f'(z) · ∂²z/∂θ²
    where z = eml(g, h) = exp(g) - ln(h). Now
        ∂z/∂θ = exp(g) · ∂g/∂θ  -  (1/h) · ∂h/∂θ
        ∂²z/∂θ² = exp(g) · (∂g/∂θ)(∂g/∂θ)ᵀ  +  exp(g) · ∂²g/∂θ²
                 + (1/h²) · (∂h/∂θ)(∂h/∂θ)ᵀ  -  (1/h) · ∂²h/∂θ²

    The leading terms scale exponentially in g(θ) and hyperbolically in h(θ).
    Consequently, the spectrum {λ_i(t)} of the Hessian *oscillates* along
    the trajectory: the (g, h) coordinates sweep through regimes where
    different eigenvalues dominate.

    Adam with β1 ≈ 0.9 carries a second-order (momentum) term, which adds a
    quasi-harmonic mode to the linearised dynamics. Linearise around a slowly
    drifting center θ* (justified by Lyapunov proxy ≈ 0):
        θ_{t+1} = θ_t + α · (β1·m_{t-1} / (sqrt(v_t) + ε))
    Taking the z-transform and substituting |λ_max| → 0 (neutral stability)
    shows the characteristic polynomial has roots on the unit circle. The
    dominant mode is ω* ≈ sqrt((1 - β2) · |λ_max| / α²) + O(edge_band).
    For the edge case |λ_max| → edge_band, ω* stays bounded away from zero
    AND bounded away from the Nyquist → bounded oscillation.

    When the Kuramoto R parameter is held above R_threshold, the gate keeps
    the system in a coherent regime where the phase information survives
    across ticks. The Adam (m, v) state acts as a phase accumulator — when
    PERSISTED across sessions (see OptimizerStateStore), it is the substrate
    of the self-referential standing wave: the system's own memory of its
    own oscillation. ∎

This is pure functional code: `heartbeat()` takes a state and returns the
next state. No module-level state, no hidden side effects. The only
external effect is the `chain_appender` callback (which may, e.g., persist
a step to an InteractionStore) — and that is a parameter.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace
from typing import Any, Callable, Protocol

import numpy as np
from numpy.typing import NDArray

from carl_core.hashing import content_hash

__all__ = [
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
]


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class HeartbeatState:
    """Full state of one CARL heartbeat tick.

    Immutable by design: every tick returns a new `HeartbeatState`. This
    makes trajectories trivially replayable, diffable, and hashable.
    """

    weights: NDArray[np.float64]
    adam_m: NDArray[np.float64]
    adam_v: NDArray[np.float64]
    step: int
    phi_mean: float
    kuramoto_R: float
    lyapunov_proxy: float
    chain_head_hash: str
    wall_time_ns: int

    def fingerprint(self) -> str:
        """Deterministic fingerprint of the state's observables.

        Used to dedup trajectories, to diff two runs at the same step, and
        as a cheap key into caches / provenance chains.
        """
        return content_hash(
            {
                "step": int(self.step),
                "phi_mean": round(float(self.phi_mean), 12),
                "kuramoto_R": round(float(self.kuramoto_R), 12),
                "lyapunov_proxy": round(float(self.lyapunov_proxy), 12),
                "chain_head_hash": self.chain_head_hash,
                "weights_hash": content_hash(
                    [round(float(x), 12) for x in self.weights.reshape(-1)]
                ),
            }
        )


@dataclass(frozen=True, slots=True)
class HeartbeatConfig:
    """Hyperparameters for the heartbeat loop.

    Defaults reflect CARL's conservation law (κ·σ = 4) and Adam canonical
    values (β1=0.9, β2=0.999, ε=1e-8). `edge_band` is the tolerance around
    λ_max = 0 defining the edge-of-chaos standing-wave regime.
    """

    lr: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    R_threshold: float = 0.5
    kappa_sigma_target: float = 4.0  # CARL conservation law (κ·σ)
    edge_band: float = 0.02  # |λ_max| tolerance for edge-of-chaos


class GradientFn(Protocol):
    """Callable producing a gradient given the current weights + observables."""

    def __call__(
        self, weights: NDArray[np.float64], phi: float, R: float
    ) -> NDArray[np.float64]: ...


class ChainAppender(Protocol):
    """Side-channel for emitting steps to a durable log (InteractionChain, etc.).

    Returns a string that becomes the next ``chain_head_hash`` (typically a
    hash of the emitted step). Any truthy return value signals 'keep
    running'; a falsy / None return stops the loop when
    ``halt_on_collapse=True``. In practice callers return a digest string
    unconditionally and use the boolean returned by `is_resonant` for halt.
    """

    def __call__(self, state: "HeartbeatState", grad: NDArray[np.float64]) -> str: ...


# ---------------------------------------------------------------------------
# Pure-functional building blocks
# ---------------------------------------------------------------------------


def adam_step(
    state: HeartbeatState,
    grad: NDArray[np.float64],
    cfg: HeartbeatConfig,
) -> HeartbeatState:
    """Pure-functional Adam update. No side effects, no mutation.

    Produces the weight-and-moment part of the next `HeartbeatState`; the
    observable fields (phi_mean, kuramoto_R, lyapunov_proxy,
    chain_head_hash, wall_time_ns) are left at their incoming values and
    should be refreshed by `heartbeat()` which owns the observation side.
    """
    g = np.asarray(grad, dtype=np.float64).reshape(state.weights.shape)
    t = state.step + 1
    # β1/β2 bias-corrected moment estimates (Kingma & Ba 2014).
    m = cfg.beta1 * state.adam_m + (1.0 - cfg.beta1) * g
    v = cfg.beta2 * state.adam_v + (1.0 - cfg.beta2) * (g * g)
    m_hat = m / (1.0 - cfg.beta1 ** t)
    v_hat = v / (1.0 - cfg.beta2 ** t)
    w_next = state.weights - cfg.lr * m_hat / (np.sqrt(v_hat) + cfg.eps)
    return replace(
        state,
        weights=w_next,
        adam_m=m,
        adam_v=v,
        step=t,
    )


def _observe_phi(weights: NDArray[np.float64]) -> float:
    """Bounded proxy for the per-tick coherence field Φ.

    A weight-norm-based scalar in (0, 1) suffices for tests and for a cheap
    default observable. Production callers will pass a richer probe via
    the `grad_fn` closure; this keeps the loop self-contained when they
    don't.
    """
    n = float(np.linalg.norm(weights))
    return 1.0 / (1.0 + n * n)


def _observe_kuramoto(weights: NDArray[np.float64]) -> float:
    """Bounded proxy for the Kuramoto order parameter R ∈ [0, 1].

    Uses the cosine of pairwise-aligned weight phases; for a 1D or 2D test
    signal this gives a stable R that reacts to weight rotation. Clamps into
    [0, 1].
    """
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if w.size < 2:
        return 1.0
    # Normalise to phase-like values via atan2 on successive pairs; for a
    # single-vector heartbeat this is a cheap surrogate for channel-wise
    # cross-coherence.
    phases = np.angle(w[:-1] + 1j * w[1:])
    R = float(abs(np.mean(np.exp(1j * phases))))
    return max(0.0, min(1.0, R))


def _observe_lyapunov_proxy(
    grad_prev: NDArray[np.float64] | None,
    grad_curr: NDArray[np.float64],
) -> float:
    """Single-step Lyapunov proxy: sign-aligned log-ratio of gradient norms.

    Returns 0.0 when the norms are equal (neutral / edge-of-chaos),
    positive when expanding, negative when contracting. In ι's framing this
    approximates the dominant Lyapunov exponent along the trajectory — at
    the flat-landscape edge it sits near zero, which is exactly the
    regime the heartbeat theorem targets.
    """
    if grad_prev is None:
        return 0.0
    n_prev = float(np.linalg.norm(grad_prev))
    n_curr = float(np.linalg.norm(grad_curr))
    if n_prev < 1e-12 or n_curr < 1e-12:
        return 0.0
    return float(math.log(n_curr / n_prev))


def heartbeat(
    state: HeartbeatState,
    cfg: HeartbeatConfig,
    grad_fn: GradientFn,
    chain_appender: Callable[[HeartbeatState, NDArray[np.float64]], str],
) -> HeartbeatState:
    """One tick of the CARL heartbeat loop. Returns the next state.

    See module docstring for the Heartbeat Standing Wave theorem.

    Pure with respect to `state` and `cfg`; side effects are confined to
    `chain_appender` (which the caller controls). The chain_appender is
    given the *pre-update* state and the gradient used for the update, so
    a persistent store can witness the derivative that shaped the step.
    """
    grad = np.asarray(
        grad_fn(state.weights, state.phi_mean, state.kuramoto_R),
        dtype=np.float64,
    ).reshape(state.weights.shape)

    # Lyapunov proxy wants a previous-gradient reference; we reconstruct it
    # from the old adam_m (bias-corrected) which sits in the same geometry.
    t = state.step
    if t > 0:
        m_hat_prev = state.adam_m / (1.0 - cfg.beta1 ** max(t, 1))
        lyap = _observe_lyapunov_proxy(m_hat_prev, grad)
    else:
        lyap = 0.0

    # Emit the pre-update step to the interaction chain (durable witness).
    # The appender returns the next chain-head hash (idempotent string).
    next_head = chain_appender(state, grad)

    # Adam step yields new weights + new (m, v).
    post_adam = adam_step(state, grad, cfg)

    # Refresh observables on the post-update weights.
    phi = _observe_phi(post_adam.weights)
    R = _observe_kuramoto(post_adam.weights)

    return replace(
        post_adam,
        phi_mean=phi,
        kuramoto_R=R,
        lyapunov_proxy=lyap,
        chain_head_hash=next_head,
        wall_time_ns=time.time_ns(),
    )


def is_resonant(state: HeartbeatState, cfg: HeartbeatConfig) -> bool:
    """True iff the state is in the edge-of-chaos standing-wave regime.

    Requires both |Lyapunov proxy| < edge_band AND R >= R_threshold. Either
    condition failing signals a collapse of the wavefunction — the loop
    has left the resonant band and is either damping (|λ| << 0) or
    diverging (|λ| >> 0) or decohering (R < τ).
    """
    return (
        abs(state.lyapunov_proxy) < cfg.edge_band
        and state.kuramoto_R >= cfg.R_threshold
    )


def run_heartbeat(
    initial: HeartbeatState,
    cfg: HeartbeatConfig,
    grad_fn: GradientFn,
    chain_appender: Callable[[HeartbeatState, NDArray[np.float64]], str],
    max_ticks: int | None = None,
    halt_on_collapse: bool = True,
) -> list[HeartbeatState]:
    """Repeatedly apply :func:`heartbeat`. Halts when:

    * ``max_ticks`` is reached, OR
    * ``halt_on_collapse=True`` AND the state has exited the resonant band
      for at least one tick (|λ| >= edge_band OR R < R_threshold), OR
    * ``chain_appender`` returns an empty string (stop-signal).

    Returns the full trajectory, including the initial state at index 0.
    """
    trajectory: list[HeartbeatState] = [initial]
    state = initial
    ticks = 0
    while True:
        if max_ticks is not None and ticks >= max_ticks:
            break
        next_state = heartbeat(state, cfg, grad_fn, chain_appender)
        trajectory.append(next_state)
        state = next_state
        ticks += 1
        if not state.chain_head_hash:
            # appender signalled stop
            break
        if halt_on_collapse and ticks > 1 and not is_resonant(state, cfg):
            break
    return trajectory


# ---------------------------------------------------------------------------
# Fourier analysis — heartbeat resonance detection
# ---------------------------------------------------------------------------


def detect_resonant_modes(trajectory: list[HeartbeatState]) -> dict[str, Any]:
    """Extract dominant periodicity from a heartbeat trajectory.

    Runs an FFT on the `phi_mean` time-series and reports:

    * ``period`` — number of ticks in the dominant cycle
    * ``amplitude`` — peak spectral amplitude
    * ``is_standing_wave`` — True iff the dominant period is finite AND the
      trajectory is bounded (peak-to-peak within ±3·std of the mean), i.e.
      NOT monotone drift.
    """
    if len(trajectory) < 4:
        return {
            "period": float("inf"),
            "amplitude": 0.0,
            "is_standing_wave": False,
        }
    phi_series = np.array([s.phi_mean for s in trajectory], dtype=np.float64)
    # detrend
    phi_centered = phi_series - float(np.mean(phi_series))
    n = phi_centered.size
    spectrum = np.abs(np.fft.rfft(phi_centered))
    # Ignore the DC bin (index 0) when finding the dominant non-DC mode.
    if spectrum.size <= 1:
        return {
            "period": float("inf"),
            "amplitude": 0.0,
            "is_standing_wave": False,
        }
    nonzero = spectrum.copy()
    nonzero[0] = 0.0
    peak_idx = int(np.argmax(nonzero))
    peak_amp = float(nonzero[peak_idx])
    if peak_idx == 0:
        period: float = float("inf")
    else:
        period = float(n) / float(peak_idx)

    # A trajectory is a standing wave if it oscillates *around* a mean with
    # bounded amplitude — i.e. the signal is NOT a monotone march.
    std = float(np.std(phi_series))
    range_ = float(np.ptp(phi_series))
    bounded = range_ < 6.0 * std + 1e-9 if std > 1e-9 else False
    has_period = math.isfinite(period) and peak_amp > 0.0
    # Also require that the peak is meaningfully taller than median noise
    # so we don't call DC drift a standing wave.
    median_amp = float(np.median(nonzero))
    distinct = peak_amp > 2.0 * max(median_amp, 1e-12)
    return {
        "period": period,
        "amplitude": peak_amp,
        "is_standing_wave": bool(bounded and has_period and distinct),
    }


# ---------------------------------------------------------------------------
# Convenience constructor
# ---------------------------------------------------------------------------


def initial_state(
    weights: NDArray[np.float64],
    *,
    chain_head_hash: str = "",
    wall_time_ns: int | None = None,
) -> HeartbeatState:
    """Build a fresh `HeartbeatState` from a weight vector.

    Zero-initialises Adam moments, observes phi/R from the weights, and
    stamps wall_time_ns. Use this once at loop start; the loop's own
    `heartbeat` handles every subsequent tick.
    """
    w = np.asarray(weights, dtype=np.float64).copy()
    return HeartbeatState(
        weights=w,
        adam_m=np.zeros_like(w),
        adam_v=np.zeros_like(w),
        step=0,
        phi_mean=_observe_phi(w),
        kuramoto_R=_observe_kuramoto(w),
        lyapunov_proxy=0.0,
        chain_head_hash=chain_head_hash,
        wall_time_ns=wall_time_ns if wall_time_ns is not None else time.time_ns(),
    )
