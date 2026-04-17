"""CoherenceTraceCallback — logs per-token coherence fields during training.

Reads CoherenceTrace objects from the CARL reward function's _last_traces
storage. Samples representative traces per step, logs rich metrics, and
optionally writes full traces to an artifact JSONL file for post-hoc analysis.

This replaces scalar-only logging with field-level observability:
  - Per-step Phi statistics from the actual per-token field
  - Defect counts (crystallizations, meltings) from delta_phi
  - Scale coherence decomposition
  - Sampled trace sparklines for quick visual inspection
  - Full trace serialization for detailed analysis
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

try:
    from transformers import TrainerCallback
except Exception:

    class TrainerCallback:  # type: ignore[no-redef]
        """Fallback base when transformers is unavailable."""

        pass


from carl_core.coherence_trace import CoherenceTrace, select_traces
from carl_core.constants import KAPPA, SIGMA


class CoherenceTraceCallback(TrainerCallback):
    """TRL TrainerCallback that captures and logs CoherenceTrace fields.

    Reads traces from the CARL reward function's _last_traces attribute
    (populated by make_carl_reward). Logs both scalar summaries and
    optional full-trace artifacts.

    Args:
        carl_reward_fn: The CARL reward closure from make_carl_reward().
            Must have _last_traces, _metrics_lock, _step attributes.
        trace_dir: Directory to write trace artifacts. None to disable.
        traces_per_step: Number of traces to sample per step (default 4:
            best, worst, median, random by CARL reward).
    """

    def __init__(
        self,
        carl_reward_fn: Any,
        trace_dir: Optional[str] = None,
        traces_per_step: int = 4,
    ) -> None:
        self.carl_fn = carl_reward_fn
        self.traces_per_step = traces_per_step
        self._trace_file = None

        if trace_dir is not None:
            path = Path(trace_dir)
            path.mkdir(parents=True, exist_ok=True)
            self._trace_file = path / "coherence_traces.jsonl"

    def _get_traces(self, state: Any) -> Optional[list[CoherenceTrace]]:
        """Thread-safe read of latest traces from CARL reward function."""
        lock = getattr(self.carl_fn, "_metrics_lock", None)
        traces_ref = getattr(self.carl_fn, "_last_traces", None)
        step_ref = getattr(self.carl_fn, "_step", None)

        if traces_ref is None:
            return None

        if lock is not None:
            lock.acquire()
            try:
                traces = traces_ref[0]
                step = step_ref[0] if step_ref else 0
            finally:
                lock.release()
        else:
            traces = traces_ref[0]
            step = step_ref[0] if step_ref else 0

        if traces is None:
            return None

        # Stale check
        if abs(step - state.global_step) > 1:
            return None

        return traces

    def on_log(
        self,
        args: Any,
        state: Any,
        control: Any,
        logs: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        if logs is None:
            return

        # Constants on first step
        if state.global_step <= 1:
            logs["coherence/kappa"] = KAPPA
            logs["coherence/sigma"] = SIGMA

        traces = self._get_traces(state)
        if not traces:
            return

        # --- Aggregate metrics from all traces in the batch ---
        phi_means = [t.phi_mean for t in traces]
        phi_stds = [t.phi_std for t in traces]
        cq_vals = [t.cloud_quality for t in traces]
        ms_vals = [t.multiscale_coherence for t in traces]
        disc_vals = [t.discontinuity_score for t in traces]
        carl_vals = [t.carl_reward() for t in traces]
        cryst_total = sum(t.n_crystallizations for t in traces)
        melt_total = sum(t.n_meltings for t in traces)
        entropy_means = [t.entropy_mean for t in traces]
        surprisal_means = [t.surprisal_mean for t in traces]
        defect_densities = [t.defect_density for t in traces]

        n = len(traces)

        # Core metrics
        logs["trace/phi_mean"] = sum(phi_means) / n
        logs["trace/phi_std"] = sum(phi_stds) / n
        logs["trace/cloud_quality"] = sum(cq_vals) / n
        logs["trace/multiscale_coherence"] = sum(ms_vals) / n
        logs["trace/discontinuity_score"] = sum(disc_vals) / n
        logs["trace/carl_reward"] = sum(carl_vals) / n
        logs["trace/entropy_mean"] = sum(entropy_means) / n
        logs["trace/surprisal_mean"] = sum(surprisal_means) / n
        logs["trace/defect_density"] = sum(defect_densities) / n

        # Defect dynamics
        logs["trace/crystallizations"] = cryst_total
        logs["trace/meltings"] = melt_total
        logs["trace/cryst_to_melt_ratio"] = cryst_total / max(melt_total, 1)

        # Spread metrics (variance within batch — measures diversity)
        if n > 1:
            import numpy as np

            logs["trace/phi_spread"] = float(np.std(phi_means))
            logs["trace/carl_spread"] = float(np.std(carl_vals))

        # Scale coherence decomposition (first 8 scales)
        all_scales: set[int] = set()
        for t in traces:
            all_scales.update(t.scale_coherence.keys())
        for j in sorted(all_scales)[:8]:
            vals = [t.scale_coherence.get(j, 0.0) for t in traces]
            logs[f"trace/scale_{j}"] = sum(vals) / len(vals)

        # --- Backward compat keys (for existing dashboards) ---
        logs["coherence/phi_mean"] = logs["trace/phi_mean"]
        logs["coherence/cloud_quality"] = logs["trace/cloud_quality"]
        logs["coherence/cryst_to_melt_ratio"] = logs["trace/cryst_to_melt_ratio"]

        # --- Sample and persist traces ---
        sampled = select_traces(traces, k=self.traces_per_step)

        if self._trace_file is not None:
            with open(self._trace_file, "a") as f:
                for trace in sampled:
                    line = json.dumps(trace.to_dict(include_arrays=True))
                    f.write(line + "\n")
