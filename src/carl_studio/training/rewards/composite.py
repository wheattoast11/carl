"""CARL composite reward: 50% multiscale + 30% cloud + 20% discontinuity.

Now built on CoherenceTrace — the per-token field is computed once and
all metrics derive from it. No triple compute_phi() calls.

Implements the RewardFunction protocol and provides make_carl_reward() factory
that returns a TRL-compatible closure.

Fixes applied:
  L1  -- eval/train mode toggle during forward pass
  L2  -- thread-safe _last_metrics with threading.Lock
  L8  -- VRAM cleanup after forward pass
  L9  -- log OOM to stderr
  WS-T3 -- NaN/inf clamping on every composite score
  WS-T4 -- logits shape guard (2D required; 3D collapsed to last-batch slice)
"""

from __future__ import annotations

import logging
import math
import sys
import threading
from typing import Any, TypedDict

import numpy as np

from carl_core.coherence_trace import CoherenceTrace
from carl_core.constants import (
    KURAMOTO_R_GASEOUS_MAX,
    KURAMOTO_R_LIQUID_MAX,
    PHASE_WEIGHTS_CRYSTALLINE,
    PHASE_WEIGHTS_GASEOUS,
    PHASE_WEIGHTS_LIQUID,
)
from carl_core.errors import ValidationError

from carl_studio.training.rewards.base import extract_text
from carl_studio.training.rewards.multiscale import _clamp_reward


logger = logging.getLogger(__name__)


class RewardComponents(TypedDict):
    coherence: float
    cloud_quality: float
    discontinuity: float


# ---------------------------------------------------------------------------
# Shape validation (WS-T4)
# ---------------------------------------------------------------------------


def _ensure_2d_logits(logits: Any) -> np.ndarray:
    """Ensure logits are [T, V].

    Accepts:
      - 2D np.ndarray [T, V] -> returned as-is.
      - 3D np.ndarray [B, T, V] -> collapsed to last batch row with a debug log.

    Raises:
      ValidationError for any other rank. The error carries ``code=carl.logits_shape``
      and the offending shape in ``context`` so the trainer can surface it.
    """
    arr = np.asarray(logits)
    if arr.ndim == 2:
        if arr.shape[0] < 1 or arr.shape[1] < 1:
            raise ValidationError(
                "empty logits tensor",
                code="carl.logits_shape",
                context={"shape": list(arr.shape)},
            )
        return arr
    if arr.ndim == 3:
        logger.debug("composite: 3D logits %s -> last-batch slice", list(arr.shape))
        return arr[-1]
    raise ValidationError(
        "expected 2D logits [T, V] (or 3D [B, T, V])",
        code="carl.logits_shape",
        context={"shape": list(arr.shape)},
    )


# ---------------------------------------------------------------------------
# CARLReward composite
# ---------------------------------------------------------------------------


class CARLReward:
    """Composite CARL reward: 50% multiscale + 30% cloud + 20% discontinuity.

    Now backed by CoherenceTrace — computes the Phi field once, derives
    all three components from the cached field arrays.
    """

    def __init__(
        self,
        weight_multiscale: float = 0.5,
        weight_cloud: float = 0.3,
        weight_discontinuity: float = 0.2,
    ) -> None:
        self.weight_multiscale = weight_multiscale
        self.weight_cloud = weight_cloud
        self.weight_discontinuity = weight_discontinuity

    def score(
        self, logits: np.ndarray, token_ids: np.ndarray
    ) -> tuple[float, dict[str, float]]:
        """Compute composite CARL score from logits.

        Args:
            logits: [T, V] raw logits (3D [B, T, V] auto-collapsed to last row).
            token_ids: [T] selected token indices.

        Returns:
            (composite_score, component_dict) where component_dict has keys:
            multiscale, cloud_quality, discontinuity.
        """
        logits_2d = _ensure_2d_logits(logits)
        token_ids_arr = np.asarray(token_ids)
        # Align token_ids rank to the 2D logits if we just collapsed a batch dim.
        if token_ids_arr.ndim == 2:
            token_ids_arr = token_ids_arr[-1]
        trace = CoherenceTrace.from_logits(logits_2d, token_ids_arr)
        return self.score_from_trace(trace)

    def score_from_trace(
        self, trace: CoherenceTrace
    ) -> tuple[float, dict[str, float]]:
        """Compute composite CARL score from a pre-computed trace.

        This is the efficient path — no logits needed, everything
        is already in the trace's per-token arrays.
        """
        composite = trace.carl_reward(
            w_coherence=self.weight_multiscale,
            w_cloud=self.weight_cloud,
            w_discontinuity=self.weight_discontinuity,
        )
        # WS-T3: clamp each derived component too so we never surface NaN/inf
        # to downstream callbacks or optimizer gradients.
        components = {
            "multiscale": _clamp_reward(trace.multiscale_coherence),
            "cloud_quality": _clamp_reward(trace.cloud_quality),
            "discontinuity": _clamp_reward(trace.discontinuity_score),
        }
        score = _clamp_reward(composite)
        return float(score), components


# ---------------------------------------------------------------------------
# Phase-adaptive composite (SEM-010)
# ---------------------------------------------------------------------------


class PhaseAdaptiveCARLReward(CARLReward):
    """Phase-adaptive weighting of the CARL composite reward.

    Reads current Kuramoto-R from the most recent CoherenceTrace batch
    and shifts weights to match the detected phase:

      - GASEOUS (R < 0.30):        reward commitment — discontinuity dominates.
      - LIQUID  (0.30 <= R < 0.70): balanced — all three components contribute.
      - CRYSTALLINE (R >= 0.70):    reward stability — multiscale dominates.

    Fall-back (no traces yet): behave like static CARLReward using the weights
    supplied at construction time.

    Trace capture: the parent ``CARLReward`` does not maintain ``_last_traces``
    itself (only the ``make_carl_reward`` closure does). This subclass caches
    the most recently scored trace(s) on every ``score``/``score_from_trace``
    call so that the phase classifier can consult them on the next invocation
    and so that external callbacks (e.g. ``ResonanceLRCallback``) can poll
    ``self._last_traces`` for Kuramoto-R.
    """

    # Per-phase weight profiles and Kuramoto-R boundaries are now module-level
    # constants in ``carl_core.constants`` (KURAMOTO_R_GASEOUS_MAX,
    # KURAMOTO_R_LIQUID_MAX, PHASE_WEIGHTS_GASEOUS, PHASE_WEIGHTS_LIQUID,
    # PHASE_WEIGHTS_CRYSTALLINE) so researchers can cite them in methods
    # sections (see docs/phase_thresholds.md).

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._last_weights: tuple[float, float, float] = (
            self.weight_multiscale,
            self.weight_cloud,
            self.weight_discontinuity,
        )
        # Trace cache used by the phase classifier and external callbacks.
        # Empty until the first score_*() call lands. Entries may be ``None``
        # when tests / callers stub out a batch slot, so the classifier is
        # None-tolerant (see ``_current_R``).
        self._last_traces: list[CoherenceTrace | None] = []

    # ------------------------------------------------------------------
    # Phase classification
    # ------------------------------------------------------------------

    def _phase_weights_from_R(
        self, R: float
    ) -> tuple[float, float, float]:
        """Map Kuramoto-R to a (w_mc, w_cq, w_disc) profile."""
        if R < KURAMOTO_R_GASEOUS_MAX:
            return PHASE_WEIGHTS_GASEOUS
        if R < KURAMOTO_R_LIQUID_MAX:
            return PHASE_WEIGHTS_LIQUID
        return PHASE_WEIGHTS_CRYSTALLINE

    def _current_R(self) -> float | None:
        """Mean Kuramoto-R across cached traces, or None if unavailable.

        Robust to:
          - No traces cached yet (returns None, static weights retained).
          - Trace objects whose ``kuramoto_R`` raises (returns None).
          - ``None`` entries in the trace list.
        """
        traces = self._last_traces
        if not traces:
            return None
        try:
            Rs: list[float] = []
            for t in traces:
                if t is None:
                    continue
                R_val = t.kuramoto_R()
                if not math.isfinite(R_val):
                    continue
                Rs.append(float(R_val))
            if not Rs:
                return None
            return sum(Rs) / len(Rs)
        except Exception as exc:  # noqa: BLE001 -- classifier must never crash training
            logger.debug(
                "PhaseAdaptiveCARLReward: kuramoto_R read failed, retaining weights: %s",
                exc,
            )
            return None

    def _apply_phase_weights(self) -> None:
        """Consult cached traces; shift weights if phase is determinable.

        Silently retains current weights on fallback (no traces / error)."""
        R = self._current_R()
        if R is None:
            return
        w_mc, w_cq, w_disc = self._phase_weights_from_R(R)
        self.weight_multiscale = w_mc
        self.weight_cloud = w_cq
        self.weight_discontinuity = w_disc
        self._last_weights = (w_mc, w_cq, w_disc)

    # ------------------------------------------------------------------
    # Scoring overrides -- shift weights BEFORE delegating, cache AFTER.
    # ------------------------------------------------------------------

    def score_from_trace(
        self, trace: CoherenceTrace
    ) -> tuple[float, dict[str, float]]:
        # Classify phase from previously-cached traces, then delegate.
        self._apply_phase_weights()
        result = super().score_from_trace(trace)
        # Cache this trace so the NEXT call reflects the most recent phase.
        self._last_traces = [trace]
        return result

    def score(
        self, logits: np.ndarray, token_ids: np.ndarray
    ) -> tuple[float, dict[str, float]]:
        # Same classify-then-delegate-then-cache pattern; the base class
        # builds the trace inside score(), so we must rebuild and cache
        # explicitly to keep the cache consistent across both entry points.
        self._apply_phase_weights()
        logits_2d = _ensure_2d_logits(logits)
        token_ids_arr = np.asarray(token_ids)
        if token_ids_arr.ndim == 2:
            token_ids_arr = token_ids_arr[-1]
        trace = CoherenceTrace.from_logits(logits_2d, token_ids_arr)
        result = super().score_from_trace(trace)
        self._last_traces = [trace]
        return result

    # ------------------------------------------------------------------
    # Batch entry point (used by parent factory and agentic callers).
    # ------------------------------------------------------------------

    def compute(
        self, traces: list[CoherenceTrace]
    ) -> list[float]:
        """Batch scoring entry point for phase-adaptive composite.

        Accepts a list of pre-built CoherenceTrace objects, classifies
        the phase from the ``_last_traces`` cache (populated by a prior
        call or manually), shifts weights, then scores each trace and
        caches the full batch for the next invocation.

        This is a sibling of ``score_from_trace`` that handles the
        batch-level protocol used by TRL-style reward functions.
        """
        self._apply_phase_weights()
        scores: list[float] = []
        for trace in traces:
            score_val, _components = super().score_from_trace(trace)
            scores.append(float(score_val))
        # Cache the whole batch so the next compute() sees the current R.
        self._last_traces = list(traces)
        return scores

    # ------------------------------------------------------------------
    # Introspection -- read-only views into current phase + weights.
    # ------------------------------------------------------------------

    @property
    def current_weights(self) -> tuple[float, float, float]:
        """Most recently applied (w_multiscale, w_cloud, w_discontinuity)."""
        return self._last_weights

    @property
    def current_phase(self) -> str:
        """Detected phase name: 'gaseous' | 'liquid' | 'crystalline' | 'unknown'."""
        R = self._current_R()
        if R is None:
            return "unknown"
        if R < KURAMOTO_R_GASEOUS_MAX:
            return "gaseous"
        if R < KURAMOTO_R_LIQUID_MAX:
            return "liquid"
        return "crystalline"


# ---------------------------------------------------------------------------
# TRL-compatible factory
# ---------------------------------------------------------------------------


def make_carl_reward(
    model: Any,
    tokenizer: Any,
    vocab_size: int = 128000,
    active_after_step: int = 0,
    max_length: int = 512,
    reward_class: str = "static",
) -> Any:
    """Factory returning a TRL-compatible CARL reward function.

    The returned closure:
    1. Tokenizes each completion text
    2. Runs a torch.no_grad() forward pass to get logits
    3. Constructs a CoherenceTrace from the logits (computed ONCE)
    4. Derives CARL composite from the trace (no triple compute_phi)
    5. Stores traces for CoherenceTraceCallback pickup

    Fixes applied:
      L1 -- eval/train mode toggle
      L2 -- thread-safe _last_metrics with threading.Lock
      L8 -- VRAM cleanup (del + empty_cache)
      L9 -- log OOM to stderr
      WS-T3 -- reward clamping to [-100, 100] with NaN/inf coercion
      WS-T4 -- logits shape guard (2D required; 3D auto-collapsed)

    Args:
        model: The model (captured in closure for forward pass).
        tokenizer: The tokenizer (captured in closure).
        vocab_size: Vocabulary size for entropy normalization.
        active_after_step: Return 0.0 before this step (cascade integration).
        max_length: Max token length for CARL forward pass.
        reward_class: ``"static"`` -> CARLReward (constant 50/30/20 weights).
            ``"phase_adaptive"`` -> PhaseAdaptiveCARLReward (weights shift
            with detected Kuramoto-R phase). Unknown values raise
            ``ValueError`` immediately — this is surfaced from YAML so a
            typo should fail fast with a clear message.
    """
    import torch

    if reward_class == "static":
        carl: CARLReward = CARLReward()
    elif reward_class == "phase_adaptive":
        carl = PhaseAdaptiveCARLReward()
    else:
        raise ValueError(
            "reward_class must be 'static' or 'phase_adaptive', got "
            f"{reward_class!r}"
        )
    _step_counter = [0]
    _last_metrics: list[Any] = [None]
    _last_components: list[Any] = [None]
    _last_traces: list[Any] = [None]  # stores list[CoherenceTrace] for last batch
    _metrics_lock = threading.Lock()

    @torch.no_grad()
    def carl_composite_reward(completions: list, **kwargs: Any) -> list[float]:
        _step_counter[0] += 1

        # Cascade: return zeros before activation step
        if _step_counter[0] < active_after_step:
            return [0.0] * len(completions)

        rewards: list[float] = []
        batch_metrics: list[dict[str, float]] = []
        batch_traces: list[CoherenceTrace] = []

        for idx, completion in enumerate(completions):
            text = extract_text(completion)
            if not text.strip() or len(text) < 10:
                rewards.append(0.0)
                continue

            try:
                inputs = tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_length,
                    padding=False,
                )
                inputs = {k: v.to(model.device) for k, v in inputs.items()}

                # Fix L1: eval/train mode toggle
                was_training = model.training
                model.eval()
                try:
                    outputs = model(**inputs)
                finally:
                    if was_training:
                        model.train()

                # WS-T4: normalize logits rank up front. Auto-collapse if
                # the model emitted a batch-dim.
                raw_logits = outputs.logits
                if raw_logits.dim() == 3:
                    logits_t = raw_logits[0]  # [T, V] — canonical per-sample view
                elif raw_logits.dim() == 2:
                    logits_t = raw_logits
                else:
                    raise ValidationError(
                        "unexpected logits rank from model",
                        code="carl.logits_shape",
                        context={"shape": list(raw_logits.shape)},
                    )
                token_ids_t = inputs["input_ids"][0]  # [T]

                logits_np = logits_t.cpu().float().numpy()
                token_ids_np = token_ids_t.cpu().numpy()

                # Fix L8: VRAM cleanup
                del outputs, logits_t, inputs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Build trace ONCE — all metrics derive from it
                trace = CoherenceTrace.from_logits(
                    logits_np,
                    token_ids_np,
                    step=_step_counter[0],
                    sample_idx=idx,
                )
                # Store token IDs for TTT micro-updates (dynamic attr, not in dataclass)
                trace._token_ids_for_ttt = token_ids_np.tolist()  # type: ignore[attr-defined]
                batch_traces.append(trace)

                score, components = carl.score_from_trace(trace)
                # WS-T3: final hard clamp before the optimizer sees the value.
                # score_from_trace already clamps, but we re-apply here so any
                # future refactor downstream of `score` still lands safe.
                if not math.isfinite(score):
                    score = 0.0
                score = max(-100.0, min(100.0, float(score)))
                batch_metrics.append(components)
                rewards.append(round(score, 4))

            except ValidationError as ve:
                # Shape/schema errors are loud — they are bugs upstream, not OOMs.
                print(
                    f"[CARL] Invalid logits: {ve} (context={ve.context})",
                    file=sys.stderr,
                )
                rewards.append(0.0)
            except Exception as e:
                # Fix L9: log OOM / generic forward failures to stderr
                print(f"[CARL] Forward pass failed: {e}", file=sys.stderr)
                rewards.append(0.0)

        # Fix L2: thread-safe metrics + trace storage
        if batch_metrics:
            with _metrics_lock:
                _last_metrics[0] = (_step_counter[0], batch_metrics)
                _last_components[0] = [
                    RewardComponents(
                        coherence=m.get("multiscale", 0.0),
                        cloud_quality=m.get("cloud_quality", 0.0),
                        discontinuity=m.get("discontinuity", 0.0),
                    )
                    for m in batch_metrics
                ]
                _last_traces[0] = batch_traces

        return rewards

    # Expose internal state for monitoring callbacks
    carl_composite_reward._last_metrics = _last_metrics  # type: ignore[attr-defined]
    carl_composite_reward._last_components = _last_components  # type: ignore[attr-defined]
    carl_composite_reward._last_traces = _last_traces  # type: ignore[attr-defined]
    carl_composite_reward._metrics_lock = _metrics_lock  # type: ignore[attr-defined]
    carl_composite_reward._step = _step_counter  # type: ignore[attr-defined]
    return carl_composite_reward
