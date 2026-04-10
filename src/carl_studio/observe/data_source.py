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
from urllib.parse import urlparse


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
    phi_sparkline: str = ""  # Per-token Phi field sparkline
    entropy_sparkline: str = ""  # Per-token entropy field sparkline
    defect_map: str = ""  # Crystallization/melting map (+/-/·)
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

            frames.append(
                ObserveFrame(
                    step=raw.get("step", raw.get("global_step", 0)),
                    phi=raw.get("phi", raw.get("phi_mean", 0.0)),
                    loss=raw.get("loss", raw.get("train_loss", 0.0)),
                    reward_mean=raw.get("reward_mean", raw.get("mean_reward", 0.0)),
                    rewards={
                        k: v
                        for k, v in raw.items()
                        if k.startswith("reward_") and isinstance(v, (int, float))
                    },
                    completion_sample=raw.get("completion", raw.get("sample", "")),
                    phase_transition=raw.get("phase_transition", False),
                )
            )

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

            frames.append(
                ObserveFrame(
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
                )
            )

        return frames

    def stream(self, interval: float = 2.0) -> Iterator[ObserveFrame]:
        while True:
            frames = self.poll()
            yield from frames
            if not frames:
                time.sleep(interval)


class TrackioError(RuntimeError):
    """Base error for Trackio-backed observe failures."""


class TrackioConfigurationError(TrackioError):
    """Raised when the Trackio source input is incomplete or invalid."""


class TrackioFetchError(TrackioError):
    """Raised when Trackio data cannot be fetched from the Space API."""


def normalize_trackio_space(value: str) -> str:
    """Normalize Trackio space inputs to the hf.space subdomain slug.

    Accepts:
      - `wheattoast11-trackio`
      - `wheattoast11/trackio`
      - `https://wheattoast11-trackio.hf.space/`
      - `https://huggingface.co/spaces/wheattoast11/trackio`
    """
    candidate = value.strip().rstrip("/")
    if not candidate:
        raise TrackioConfigurationError("Trackio space is required.")

    if "://" not in candidate:
        if candidate.count("/") == 1:
            owner, space = candidate.split("/", 1)
            if owner and space:
                return f"{owner}-{space}"
        return candidate

    parsed = urlparse(candidate)
    host = parsed.netloc.lower()
    if host.endswith(".hf.space"):
        return host[: -len(".hf.space")]

    if host == "huggingface.co":
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) >= 3 and parts[0] == "spaces":
            owner = parts[1].strip()
            space = parts[2].strip()
            if owner and space:
                return f"{owner}-{space}"

    raise TrackioConfigurationError(
        "Unsupported Trackio URL. Use a Space slug like 'owner-space', an HF URL like "
        "'https://owner-space.hf.space/', or a browser URL like "
        "'https://huggingface.co/spaces/owner/space'."
    )


class TrackioSource:
    """Poll a Trackio dashboard Space for live metrics.

    For remote dashboards, CARL uses the same Gradio APIs that Trackio's own CLI
    uses (`/get_all_projects`, `/get_runs_for_project`, `/get_logs`).
    """

    def __init__(
        self,
        space: str = "wheattoast11-trackio",
        project: str = "",
        run: str = "",
        token: str | None = None,
    ) -> None:
        self.space = normalize_trackio_space(space)
        self.project = project.strip()
        self.run = run.strip()
        self.token = token
        self._last_step = 0
        self._client = None
        self._resolved_project: str | None = self.project or None
        self._resolved_run: str | None = self.run or None

    @property
    def resolved_project(self) -> str:
        return self._resolved_project or ""

    @property
    def resolved_run(self) -> str:
        return self._resolved_run or ""

    def _get_client(self):
        if self._client is not None:
            return self._client

        try:
            from gradio_client import Client
        except ImportError as exc:
            raise TrackioConfigurationError(
                "Remote Trackio observation requires gradio-client. Reinstall carl-studio "
                "or install gradio-client explicitly."
            ) from exc

        kwargs = {"verbose": False}
        if self.token:
            kwargs["hf_token"] = self.token

        try:
            self._client = Client(self.space, **kwargs)
        except Exception as exc:
            raise TrackioFetchError(
                f"Could not connect to Trackio Space '{self.space}'. Is it running? {exc}"
            ) from exc

        return self._client

    def _predict(self, *args, api_name: str):
        client = self._get_client()
        try:
            return client.predict(*args, api_name=api_name)
        except Exception as exc:
            message = str(exc)
            if "API Not Found" in message or "api_name" in message:
                raise TrackioFetchError(
                    f"Space '{self.space}' does not expose the Trackio API '{api_name}'. "
                    "Redeploy or resync the Trackio dashboard."
                ) from exc
            raise TrackioFetchError(
                f"Trackio request failed for space '{self.space}': {exc}"
            ) from exc

    def _resolve_project_name(self) -> str:
        if self._resolved_project:
            return self._resolved_project

        projects = self._predict(api_name="/get_all_projects")
        if not isinstance(projects, list) or not projects:
            raise TrackioFetchError(f"No Trackio projects found in space '{self.space}'.")
        if len(projects) > 1:
            listed = ", ".join(str(project) for project in projects)
            raise TrackioConfigurationError(
                f"Multiple Trackio projects found in '{self.space}'. Pass --project. "
                f"Available: {listed}"
            )

        self._resolved_project = str(projects[0])
        return self._resolved_project

    def _resolve_run_name(self, project: str) -> str:
        if self._resolved_run:
            return self._resolved_run

        runs = self._predict(project, api_name="/get_runs_for_project")
        if not isinstance(runs, list) or not runs:
            raise TrackioFetchError(
                f"No runs found for project '{project}' in space '{self.space}'."
            )
        if len(runs) > 1:
            listed = ", ".join(str(run) for run in runs)
            raise TrackioConfigurationError(
                f"Multiple runs found for project '{project}'. Pass --run. Available: {listed}"
            )

        self._resolved_run = str(runs[0])
        return self._resolved_run

    def poll(self) -> list[ObserveFrame]:
        """Fetch new metrics from Trackio since last poll."""
        project = self._resolve_project_name()
        run = self._resolve_run_name(project)
        data = self._predict(project, run, api_name="/get_logs")

        if not isinstance(data, list):
            raise TrackioFetchError(
                f"Unexpected response from Trackio logs API for '{project}/{run}'."
            )

        frames = []
        for entry in data:
            if not isinstance(entry, dict):
                continue

            step = int(entry.get("step", 0) or 0)
            if step <= self._last_step:
                continue
            self._last_step = step

            rewards = {
                k: float(v)
                for k, v in entry.items()
                if isinstance(v, (int, float))
                and (k.startswith("reward_") or k.startswith("rewards/"))
            }
            reward_mean = float(entry.get("reward_mean", entry.get("mean_reward", 0.0)) or 0.0)
            if reward_mean == 0.0 and rewards:
                reward_mean = sum(rewards.values()) / len(rewards)

            frames.append(
                ObserveFrame(
                    step=step,
                    phi=float(
                        entry.get("phi", entry.get("phi_mean", entry.get("trace/phi_mean", 0.0)))
                        or 0.0
                    ),
                    loss=float(entry.get("loss", entry.get("train_loss", 0.0)) or 0.0),
                    reward_mean=reward_mean,
                    rewards=rewards,
                    completion_sample=str(entry.get("completion", entry.get("sample", "")) or ""),
                    phase_transition=entry.get("phase_transition", False),
                )
            )

        return frames

    def stream(self, interval: float = 5.0) -> Iterator[ObserveFrame]:
        """Blocking iterator — yields frames as they appear."""
        while True:
            frames = self.poll()
            yield from frames
            if not frames:
                time.sleep(interval)
