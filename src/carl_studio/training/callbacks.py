"""CoherenceMonitorCallback -- logs crystal dynamics metrics per training step.

Ported from CrystalMonitorCallback with fixes:
  L2    -- thread-safe read of _last_metrics via threading.Lock
  A4    -- stale metrics detection via step comparison
  WS-T4 -- empty-batch guard on every len(X) division
  WS-T6 -- `with lock:` context manager instead of manual acquire/release;
           callback-level try/except isolation so a failing callback cannot
           crash the training loop.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable

try:
    from transformers import TrainerCallback
except Exception:

    class TrainerCallback:  # type: ignore[no-redef]
        """Fallback base when transformers is unavailable."""

        pass


from carl_core.constants import KAPPA, SIGMA

from carl_studio.training.rewards.multiscale import clamp_counts, reset_clamp_counts


logger = logging.getLogger(__name__)


def _safe_mean(values: Iterable[float], default: float = 0.0) -> float:
    """Mean that never divides by zero.

    Returns ``default`` for empty or non-iterable inputs, and silently skips
    non-numeric values. Used everywhere a callback aggregates across a batch.
    """
    try:
        vals = [float(v) for v in values]
    except (TypeError, ValueError):
        return default
    if not vals:
        return default
    return sum(vals) / len(vals)


class CoherenceMonitorCallback(TrainerCallback):
    """Logs crystal dynamics metrics every training step.

    Reads _last_metrics from the CARL reward function closure in a
    thread-safe manner (fix L2) and skips stale data (fix A4).

    Logged keys:
      coherence/phi_mean            -- mean order parameter
      coherence/cloud_quality       -- mean cloud quality
      coherence/discontinuity_density -- alias for defect density (not yet computed here)
      coherence/cryst_to_melt_ratio -- crystallizations / max(meltings, 1)
      coherence/coherence_scale_{j} -- per-scale coherence (first 8 scales)
      coherence/reward_clamp_total / _nonfinite / _overflow
                                    -- per-period clamp-telemetry counts
    """

    def __init__(self, carl_reward_fn: Any) -> None:
        self.carl_fn = carl_reward_fn

    def on_log(
        self,
        args: Any,
        state: Any,
        control: Any,
        logs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        try:
            self._on_log_impl(args, state, control, logs, **kwargs)
        except Exception as exc:
            # WS-T6: never propagate — training loop must survive buggy callback.
            logger.warning(
                "callback %s failed: %s", type(self).__name__, exc
            )

    def on_epoch_end(
        self,
        args: Any,
        state: Any,
        control: Any,
        **kwargs: Any,
    ) -> None:
        try:
            # Emit clamp telemetry summary and reset the counters at the epoch boundary.
            counts = clamp_counts()
            if counts.get("total", 0):
                logger.info(
                    "coherence reward clamps this epoch: total=%d nonfinite=%d overflow=%d",
                    counts.get("total", 0),
                    counts.get("nonfinite", 0),
                    counts.get("overflow", 0),
                )
            reset_clamp_counts()
        except Exception as exc:
            logger.warning(
                "callback %s.on_epoch_end failed: %s", type(self).__name__, exc
            )

    def on_step_end(
        self,
        args: Any,
        state: Any,
        control: Any,
        **kwargs: Any,
    ) -> None:
        # Hook retained for symmetry; all per-step telemetry currently emits
        # through on_log. Wrapped in try/except so subclasses or future logic
        # cannot break the training loop.
        try:
            return None
        except Exception as exc:
            logger.warning(
                "callback %s.on_step_end failed: %s", type(self).__name__, exc
            )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _on_log_impl(
        self,
        args: Any,
        state: Any,
        control: Any,
        logs: dict[str, Any] | None,
        **kwargs: Any,
    ) -> None:
        if logs is None:
            return

        # Log crystal constants on first step
        if state.global_step <= 1:
            logs["coherence/kappa"] = KAPPA
            logs["coherence/sigma"] = SIGMA
            logs["coherence/kappa_x_sigma"] = KAPPA * SIGMA

        # Fix L2 / WS-T6: thread-safe read using `with lock:`
        lock = getattr(self.carl_fn, "_metrics_lock", None)
        metrics_ref = getattr(self.carl_fn, "_last_metrics", None)
        if metrics_ref is None:
            return

        if lock is not None:
            with lock:
                data = metrics_ref[0]
        else:
            data = metrics_ref[0]

        if data is None:
            return

        stored_step, batch_metrics = data

        # Fix A4: stale metrics detection via step comparison
        if abs(stored_step - state.global_step) > 1:
            return  # stale, skip

        if not batch_metrics:
            return

        # batch_metrics is a list of component dicts. Two supported schemas:
        #   new    -- {"multiscale", "cloud_quality", "discontinuity"} from CARLReward.score()
        #   legacy -- {"phi_mean", "cloud_quality_mean", ...} from _logits_to_crystal_metrics
        sample = batch_metrics[0]
        is_legacy = "phi_mean" in sample

        if is_legacy:
            phi_means = [m.get("phi_mean", 0.0) for m in batch_metrics]
            cq_means = [m.get("cloud_quality_mean", 0.0) for m in batch_metrics]
            dd_means = [m.get("defect_density", 0.0) for m in batch_metrics]
            cryst = sum(m.get("n_crystallizations", 0) for m in batch_metrics)
            melts = sum(m.get("n_meltings", 0) for m in batch_metrics)

            # WS-T4: every aggregator uses _safe_mean so empty batches never
            # ZeroDivisionError.
            logs["coherence/phi_mean"] = _safe_mean(phi_means)
            logs["coherence/cloud_quality"] = _safe_mean(cq_means)
            logs["coherence/discontinuity_density"] = _safe_mean(dd_means)
            logs["coherence/cryst_to_melt_ratio"] = cryst / max(melts, 1)

            # Per-scale coherence
            all_scales: set[int] = set()
            for m in batch_metrics:
                sc = m.get("scale_coherence", {}) or {}
                all_scales.update(sc.keys())
            for j in sorted(all_scales)[:8]:
                vals = [
                    (m.get("scale_coherence") or {}).get(j, 0.0)
                    for m in batch_metrics
                ]
                logs[f"coherence/coherence_scale_{j}"] = _safe_mean(vals)
        else:
            ms_vals = [m.get("multiscale", 0.0) for m in batch_metrics]
            cq_vals = [m.get("cloud_quality", 0.0) for m in batch_metrics]
            disc_vals = [m.get("discontinuity", 0.0) for m in batch_metrics]

            logs["coherence/phi_mean"] = _safe_mean(ms_vals)
            logs["coherence/cloud_quality"] = _safe_mean(cq_vals)
            logs["coherence/discontinuity_density"] = _safe_mean(disc_vals)
            composite_vals = [
                0.5 * m.get("multiscale", 0.0)
                + 0.3 * m.get("cloud_quality", 0.0)
                + 0.2 * m.get("discontinuity", 0.0)
                for m in batch_metrics
            ]
            logs["coherence/cryst_to_melt_ratio"] = _safe_mean(composite_vals)

        # Clamp-telemetry: always emit current running counts for observability.
        counts = clamp_counts()
        logs["coherence/reward_clamp_total"] = counts.get("total", 0)
        logs["coherence/reward_clamp_nonfinite"] = counts.get("nonfinite", 0)
        logs["coherence/reward_clamp_overflow"] = counts.get("overflow", 0)
