"""Data sources for the observe dashboard.

Each source yields ``ObserveFrame`` objects that the TUI renders.
Two backends:
  - FileSource:    tail a JSONL log file (works offline, any training backend)
  - TrackioSource: poll Trackio API (works with ``report_to="trackio"`` in config)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator


@dataclass
class ObserveFrame:
    """One snapshot of training state."""

    step: int = 0
    phi: float = 0.0
    loss: float = 0.0
    reward_mean: float = 0.0
    rewards: dict[str, float] = field(default_factory=dict)
    completion_sample: str = ""
    phase_transition: bool = False
    timestamp: float = field(default_factory=time.time)

    # Trace-level data (optional — populated when trace artifacts available)
    phi_sparkline: str = ""        # Per-token Phi field sparkline
    entropy_sparkline: str = ""    # Per-token entropy field sparkline
    defect_map: str = ""           # Crystallization/melting map (+/-/·)
    trace_carl_reward: float = 0.0
    trace_n_tokens: int = 0
    trace_crystallizations: int = 0
    trace_meltings: int = 0


class FileSource:
    """Tail a JSONL metrics log file.

    Expected format per line (flexible — extracts what it finds)::

        {"step": 10, "phi": 0.85, "loss": 1.2, "reward_mean": 0.6, ...}

    Usage::

        source = FileSource("training_log.jsonl")
        for frame in source.poll():
            dashboard.update(frame)
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._offset = 0

    def poll(self) -> list[ObserveFrame]:
        """Read new lines since last poll. Returns list of new frames."""
        if not self.path.exists():
            return []

        with open(self.path) as f:
            f.seek(self._offset)
            new_lines = f.readlines()
            self._offset = f.tell()

        frames = []
        for line in new_lines:
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                continue

            frames.append(ObserveFrame(
                step=raw.get("step", raw.get("global_step", 0)),
                phi=raw.get("phi", raw.get("phi_mean", 0.0)),
                loss=raw.get("loss", raw.get("train_loss", 0.0)),
                reward_mean=raw.get("reward_mean", raw.get("mean_reward", 0.0)),
                rewards={k: v for k, v in raw.items()
                         if k.startswith("reward_") and isinstance(v, (int, float))},
                completion_sample=raw.get("completion", raw.get("sample", "")),
                phase_transition=raw.get("phase_transition", False),
            ))

        return frames

    def stream(self, interval: float = 2.0) -> Iterator[ObserveFrame]:
        """Blocking iterator — yields frames as they appear."""
        while True:
            frames = self.poll()
            yield from frames
            if not frames:
                time.sleep(interval)


class TraceFileSource:
    """Read CoherenceTrace artifacts from coherence_traces.jsonl.

    These are the full per-token traces written by CoherenceTraceCallback.
    Produces ObserveFrames with trace-level visualization data populated.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._offset = 0

    def poll(self) -> list[ObserveFrame]:
        if not self.path.exists():
            return []

        with open(self.path) as f:
            f.seek(self._offset)
            new_lines = f.readlines()
            self._offset = f.tell()

        frames = []
        for line in new_lines:
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Reconstruct trace for sparklines
            try:
                from carl_studio.primitives.coherence_trace import CoherenceTrace
                trace = CoherenceTrace.from_dict(raw)
                phi_spark = trace.sparkline(width=60)
                entropy_spark = trace.entropy_sparkline(width=60)
                defect = trace.defect_map(width=60)
                carl_r = trace.carl_reward()
            except Exception:
                phi_spark = ""
                entropy_spark = ""
                defect = ""
                carl_r = raw.get("carl_reward", 0.0)

            frames.append(ObserveFrame(
                step=raw.get("step", 0),
                phi=raw.get("phi_mean", 0.0),
                loss=0.0,
                reward_mean=carl_r,
                phi_sparkline=phi_spark,
                entropy_sparkline=entropy_spark,
                defect_map=defect,
                trace_carl_reward=carl_r,
                trace_n_tokens=raw.get("n_tokens", 0),
                trace_crystallizations=raw.get("n_crystallizations", 0),
                trace_meltings=raw.get("n_meltings", 0),
            ))

        return frames

    def stream(self, interval: float = 2.0) -> Iterator[ObserveFrame]:
        while True:
            frames = self.poll()
            yield from frames
            if not frames:
                time.sleep(interval)


class TrackioSource:
    """Poll Trackio API for live metrics.

    Requires ``trackio`` package and a running Trackio space.

    Usage::

        source = TrackioSource(space="wheattoast11-trackio", run="my-run")
        for frame in source.poll():
            dashboard.update(frame)
    """

    def __init__(
        self,
        space: str = "wheattoast11-trackio",
        run: str = "",
        token: str | None = None,
    ) -> None:
        self.space = space
        self.run = run
        self.token = token
        self._last_step = 0

    def _api_url(self) -> str:
        return f"https://{self.space}.hf.space/api/runs/{self.run}/metrics"

    def poll(self) -> list[ObserveFrame]:
        """Fetch new metrics from Trackio since last poll."""
        try:
            import urllib.request
            import urllib.error

            url = self._api_url()
            if self._last_step > 0:
                url += f"?after_step={self._last_step}"

            req = urllib.request.Request(url)
            if self.token:
                req.add_header("Authorization", f"Bearer {self.token}")

            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())

        except Exception:
            return []

        if not isinstance(data, list):
            data = data.get("metrics", []) if isinstance(data, dict) else []

        frames = []
        for entry in data:
            step = entry.get("step", 0)
            if step <= self._last_step:
                continue
            self._last_step = step

            frames.append(ObserveFrame(
                step=step,
                phi=entry.get("phi", entry.get("phi_mean", 0.0)),
                loss=entry.get("loss", 0.0),
                reward_mean=entry.get("reward_mean", 0.0),
                rewards={k: v for k, v in entry.items()
                         if k.startswith("reward_") and isinstance(v, (int, float))},
                phase_transition=entry.get("phase_transition", False),
            ))

        return frames

    def stream(self, interval: float = 5.0) -> Iterator[ObserveFrame]:
        """Blocking iterator — yields frames as they appear."""
        while True:
            frames = self.poll()
            yield from frames
            if not frames:
                time.sleep(interval)
