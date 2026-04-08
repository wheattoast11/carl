"""CoherenceMonitorCallback -- logs crystal dynamics metrics per training step.

Ported from CrystalMonitorCallback with fixes:
  L2  -- thread-safe read of _last_metrics via threading.Lock
  A4  -- stale metrics detection via step comparison
"""

from __future__ import annotations

from typing import Any

from transformers import TrainerCallback

from carl_studio.primitives.constants import KAPPA, SIGMA


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
        if logs is None:
            return

        # Log crystal constants on first step
        if state.global_step <= 1:
            logs["coherence/kappa"] = KAPPA
            logs["coherence/sigma"] = SIGMA
            logs["coherence/kappa_x_sigma"] = KAPPA * SIGMA

        # Fix L2: thread-safe read of latest crystal metrics
        lock = getattr(self.carl_fn, "_metrics_lock", None)
        metrics_ref = getattr(self.carl_fn, "_last_metrics", None)
        if metrics_ref is None:
            return

        if lock is not None:
            lock.acquire()
            try:
                data = metrics_ref[0]
            finally:
                lock.release()
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

        # batch_metrics is a list of component dicts from CARLReward.score():
        # {"multiscale": float, "cloud_quality": float, "discontinuity": float}
        #
        # For backward compatibility, also handle legacy dict format with
        # "phi_mean", "cloud_quality_mean", etc.

        # Detect format: new (from CARLReward.score) vs legacy (from _logits_to_crystal_metrics)
        sample = batch_metrics[0]
        is_legacy = "phi_mean" in sample

        if is_legacy:
            # Legacy format from _logits_to_crystal_metrics
            phi_means = [m["phi_mean"] for m in batch_metrics]
            cq_means = [m["cloud_quality_mean"] for m in batch_metrics]
            dd_means = [m.get("defect_density", 0.0) for m in batch_metrics]
            cryst = sum(m.get("n_crystallizations", 0) for m in batch_metrics)
            melts = sum(m.get("n_meltings", 0) for m in batch_metrics)

            logs["coherence/phi_mean"] = sum(phi_means) / len(phi_means)
            logs["coherence/cloud_quality"] = sum(cq_means) / len(cq_means)
            logs["coherence/discontinuity_density"] = sum(dd_means) / len(dd_means)
            logs["coherence/cryst_to_melt_ratio"] = cryst / max(melts, 1)

            # Per-scale coherence
            all_scales: set[int] = set()
            for m in batch_metrics:
                sc = m.get("scale_coherence", {})
                all_scales.update(sc.keys())
            for j in sorted(all_scales)[:8]:
                vals = [m.get("scale_coherence", {}).get(j, 0.0) for m in batch_metrics]
                logs[f"coherence/coherence_scale_{j}"] = sum(vals) / len(vals)
        else:
            # New format from CARLReward.score() components dict
            ms_vals = [m.get("multiscale", 0.0) for m in batch_metrics]
            cq_vals = [m.get("cloud_quality", 0.0) for m in batch_metrics]
            disc_vals = [m.get("discontinuity", 0.0) for m in batch_metrics]

            logs["coherence/phi_mean"] = sum(ms_vals) / len(ms_vals)
            logs["coherence/cloud_quality"] = sum(cq_vals) / len(cq_vals)
            logs["coherence/discontinuity_density"] = sum(disc_vals) / len(disc_vals)
            # Composite score as proxy for cryst_to_melt_ratio in new format
            composite_vals = [
                0.5 * m.get("multiscale", 0.0)
                + 0.3 * m.get("cloud_quality", 0.0)
                + 0.2 * m.get("discontinuity", 0.0)
                for m in batch_metrics
            ]
            logs["coherence/cryst_to_melt_ratio"] = (
                sum(composite_vals) / len(composite_vals)
            )
