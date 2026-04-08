"""
coherence_observer.py -- Observability pipeline for coherence-aligned training.

Ported from zero-rl-pipeline/phase1/crystal_observer.py (CrystalObserver).

Renames:
  CrystalObserver   -> CoherenceObserver
  CrystalSnapshot   -> CoherenceSnapshot
  crystal health    -> coherence health
  defect density    -> discontinuity density
  crystallizations  -> commitment events
  meltings          -> dissolution events

Accumulates CoherenceSnapshot outputs from CoherenceProbe. Periodically sends
a metrics window to Claude for interpretation. Returns structured assessments
of coherence health.

Usage:
    from carl_studio.primitives import CoherenceProbe, CoherenceObserver

    probe = CoherenceProbe(vocab_size=128000)
    observer = CoherenceObserver(api_key="sk-ant-...", observe_every=50)

    for step, batch in enumerate(training_loop):
        logits = model.generate_with_logits(batch)       # [T, V]
        token_ids = logits.argmax(-1)                     # [T]

        snapshot = probe.measure(logits, token_ids, step=step)

        assessment = observer.ingest(snapshot)
        if assessment:
            print(assessment["diagnosis"])

Tej Desai x Claude Opus 4.6 -- April 1, 2026
Intuition Labs LLC x Anthropic
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional

from .coherence_probe import CoherenceSnapshot
from .constants import DEFECT_THRESHOLD, KAPPA, SIGMA


# ============================================================
# System prompt -- teaches Claude what to look for.
# Distilled watchpoint list from the CARL analysis.
# ============================================================

OBSERVER_SYSTEM_PROMPT = """You are a Coherence Dynamics Observer for an RL training run based on Crystal-Aligned Reinforcement Learning (CARL).

You receive periodic batches of CoherenceSnapshot metrics from a training run. Your job is to diagnose the coherence health and flag specific events.

## Key Constants
- kappa = 64/3 ~ 21.33 (conservation constant)
- sigma = 3/16 = 0.1875 (semantic quantum -- minimum meaningful signal)
- Discontinuity threshold: |delta_phi| > 0.03

## What to Watch For

### 1. PHASE TRANSITION (Critical Event)
The Kuramoto theory predicts a discrete phase transition where discontinuity choreography suddenly snaps into alignment. Signal: sudden drop in discontinuity density variance across batches, accompanied by a jump in mean phi. If you see this, flag it prominently -- it means a coherence transition has occurred and training is working.

### 2. SCALE-SEPARATED CONVERGENCE
Fine scales (j=0,1,2) should improve before coarse scales (j=8+). If coarse improves first, flag as WARNING: "hallucination factory risk -- global coherence without local structure."

### 3. DISCONTINUITY CHOREOGRAPHY EMERGENCE
Early in training: discontinuities are random (commitment events and dissolution events roughly equal, no positional pattern). As training works: commitment events should exceed dissolution events (the model learns to commit more than it collapses). If dissolution events consistently exceed commitment events, flag as WARNING: "coherence destabilizing."

### 4. CLOUD QUALITY TREND
Cloud quality = P(selected) x phi. Should increase over training. If it plateaus while overall reward increases, the model is getting lucky (right tokens from noisy distributions) rather than genuinely improving.

### 5. NOISE FLOOR (sigma = 0.1875)
If advantage_above_sigma is provided, it shows what fraction of per-token advantages exceed the semantic quantum. If this drops below 30%, the reward signal is mostly sub-semantic noise -- learning rate or reward scale may need adjustment.

### 6. KL STEPPING
If KL divergence data is available: look for a stepped (staircase) pattern rather than smooth monotonic rise. Steps indicate the model learning new discontinuity patterns. Smooth rise suggests drift without structural learning.

### 7. ENTROPY-STRUCTURE CORRELATION
The phi trajectory should develop a characteristic shape correlated with content structure over training. Signal: phi_std should initially rise (the model develops more varied confidence levels) then stabilize at a higher value than baseline.

## Response Format
Respond with a JSON object (no markdown, no backticks):
{
  "status": "HEALTHY" | "WARNING" | "CRITICAL" | "PHASE_TRANSITION",
  "diagnosis": "1-3 sentence summary of coherence state",
  "signals": [
    {"name": "...", "status": "ok|watch|alert", "detail": "..."}
  ],
  "recommendations": ["actionable suggestion 1", ...],
  "metrics_summary": {
    "phi_trend": "rising|stable|falling",
    "defect_trend": "improving|stable|degrading",
    "cloud_trend": "tightening|stable|loosening",
    "scale_alignment": "correct|inverted|unclear"
  }
}"""


class CoherenceObserver:
    """
    Accumulates CoherenceSnapshots and periodically sends them to Claude
    for interpretation. Returns structured assessments.

    Gracefully degrades when no API key is available -- returns a warning
    assessment instead of crashing.

    Parameters:
        api_key:        Anthropic API key (or set ANTHROPIC_API_KEY env var)
        observe_every:  How many snapshots to accumulate before calling Claude
        model:          Claude model to use for observation
        window_size:    How many recent snapshots to include in each observation
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        observe_every: int = 50,
        model: str = "claude-sonnet-4-20250514",
        window_size: int = 100,
    ) -> None:
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.observe_every = observe_every
        self.model = model
        self.window_size = window_size

        self._buffer: List[CoherenceSnapshot] = []
        self._history: List[Dict[str, Any]] = []  # Past assessments
        self._since_last_observe: int = 0

    def ingest(self, snapshot: CoherenceSnapshot) -> Optional[Dict[str, Any]]:
        """
        Ingest a snapshot. Returns an assessment dict if it's time to observe,
        otherwise returns None.
        """
        self._buffer.append(snapshot)
        if len(self._buffer) > self.window_size:
            self._buffer = self._buffer[-self.window_size:]

        self._since_last_observe += 1

        if self._since_last_observe >= self.observe_every:
            self._since_last_observe = 0
            return self._observe()

        return None

    def force_observe(self) -> Dict[str, Any]:
        """Force an observation right now, regardless of schedule."""
        return self._observe()

    def _observe(self) -> Dict[str, Any]:
        """Send the current window to Claude and return the assessment."""
        window = self._buffer[-self.window_size:]
        if not window:
            return {
                "status": "HEALTHY",
                "diagnosis": "No data yet.",
                "signals": [],
                "recommendations": [],
            }

        # Compress the window into a summary that fits in a single message.
        summary = self._compress_window(window)

        # Build the user message
        user_message = json.dumps(summary, indent=2)

        # Call Claude
        try:
            assessment = self._call_claude(user_message)
        except Exception as e:
            assessment = {
                "status": "WARNING",
                "diagnosis": f"Observer API call failed: {e}",
                "signals": [],
                "recommendations": ["Check API connectivity"],
            }

        # Store in history
        self._history.append({
            "timestamp": time.time(),
            "step_range": [window[0].step, window[-1].step],
            "assessment": assessment,
        })

        return assessment

    def _compress_window(self, window: List[CoherenceSnapshot]) -> Dict[str, Any]:
        """
        Compress a window of snapshots into a summary for Claude.
        Goal: fit in ~2K tokens while preserving all diagnostic signals.
        """
        steps = [s.step for s in window]
        n = len(window)

        # Aggregate time series (sample evenly across window)
        sample_indices = list(range(0, n, max(1, n // 20)))  # <=20 samples
        sampled = [window[i] for i in sample_indices]

        # Compute trends
        phi_means = [s.phi_mean for s in window]
        defect_densities = [s.defect_density for s in window]
        cloud_qualities = [s.cloud_quality_mean for s in window]
        surprisals = [s.surprisal_mean for s in window]

        def trend(values: List[float]) -> float:
            if len(values) < 2:
                return 0.0
            half = len(values) // 2
            first_half = sum(values[:half]) / max(half, 1)
            second_half = sum(values[half:]) / max(len(values) - half, 1)
            return second_half - first_half

        # Scale coherence: average across window, per scale
        all_scales: set[int] = set()
        for s in window:
            all_scales.update(s.scale_coherence.keys())

        scale_means: Dict[int, float] = {}
        scale_trends: Dict[int, float] = {}
        for j in sorted(all_scales):
            vals = [s.scale_coherence.get(j, 0.0) for s in window]
            scale_means[j] = sum(vals) / len(vals)
            scale_trends[j] = trend(vals)

        return {
            "window": {
                "step_start": steps[0],
                "step_end": steps[-1],
                "n_snapshots": n,
            },
            "phi": {
                "current": phi_means[-1] if phi_means else 0,
                "mean": sum(phi_means) / max(len(phi_means), 1),
                "trend": trend(phi_means),
                "std_mean": sum(s.phi_std for s in window) / n,
            },
            "defects": {
                "current_density": defect_densities[-1] if defect_densities else 0,
                "mean_density": sum(defect_densities) / max(len(defect_densities), 1),
                "trend": trend(defect_densities),
                "total_crystallizations": sum(s.n_crystallizations for s in window),
                "total_meltings": sum(s.n_meltings for s in window),
                "cryst_to_melt_ratio": (
                    sum(s.n_crystallizations for s in window)
                    / max(sum(s.n_meltings for s in window), 1)
                ),
            },
            "cloud": {
                "current": cloud_qualities[-1] if cloud_qualities else 0,
                "mean": sum(cloud_qualities) / max(len(cloud_qualities), 1),
                "trend": trend(cloud_qualities),
            },
            "surprisal": {
                "current": surprisals[-1] if surprisals else 0,
                "mean": sum(surprisals) / max(len(surprisals), 1),
                "trend": trend(surprisals),
            },
            "entropy": {
                "mean": sum(s.entropy_mean for s in window) / n,
                "std_mean": sum(s.entropy_std for s in window) / n,
            },
            "top_k_mass": {
                "mean": sum(s.top_k_mass for s in window) / n,
                "trend": trend([s.top_k_mass for s in window]),
            },
            "scale_coherence": {
                "per_scale_mean": {str(j): round(v, 4) for j, v in scale_means.items()},
                "per_scale_trend": {str(j): round(v, 4) for j, v in scale_trends.items()},
            },
            "advantage_signal": {
                "mean_magnitude": (
                    sum(s.advantage_mean for s in window if s.advantage_mean is not None)
                    / max(sum(1 for s in window if s.advantage_mean is not None), 1)
                )
                if any(s.advantage_mean is not None for s in window)
                else None,
                "fraction_above_sigma": (
                    sum(s.advantage_above_sigma for s in window if s.advantage_above_sigma is not None)
                    / max(sum(1 for s in window if s.advantage_above_sigma is not None), 1)
                )
                if any(s.advantage_above_sigma is not None for s in window)
                else None,
            },
            "constants": {
                "kappa": KAPPA,
                "sigma": SIGMA,
            },
            # Include a few raw phi trajectories for pattern detection
            "sample_phi_trajectories": [
                {"step": s.step, "phi": s.phi_trajectory}
                for s in sampled[:5]
            ],
            # History context: previous assessments (last 3)
            "prior_assessments": [
                {
                    "step_range": h["step_range"],
                    "status": h["assessment"].get("status", "unknown"),
                    "diagnosis": h["assessment"].get("diagnosis", ""),
                }
                for h in self._history[-3:]
            ],
        }

    def _call_claude(self, user_message: str) -> Dict[str, Any]:
        """Call the Anthropic API and parse the response."""
        if not self.api_key:
            return {
                "status": "WARNING",
                "diagnosis": "No ANTHROPIC_API_KEY available. Observer running in degraded mode -- metrics are collected but not interpreted.",
                "signals": [],
                "recommendations": [
                    "Set ANTHROPIC_API_KEY environment variable or pass api_key to CoherenceObserver"
                ],
            }

        import anthropic

        client = anthropic.Anthropic(api_key=self.api_key)

        response = client.messages.create(
            model=self.model,
            max_tokens=1000,
            system=OBSERVER_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )

        # Extract text from response
        text = ""
        for block in response.content:
            if hasattr(block, "text"):
                text += block.text

        # Parse JSON response
        text = text.strip()
        # Strip markdown code fences if present
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {
                "status": "WARNING",
                "diagnosis": text[:500],
                "signals": [],
                "recommendations": ["Observer returned non-JSON -- check system prompt"],
            }

    @property
    def history(self) -> List[Dict[str, Any]]:
        """All past assessments."""
        return self._history

    @property
    def buffer(self) -> List[CoherenceSnapshot]:
        """Current snapshot buffer."""
        return self._buffer
