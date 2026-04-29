"""Substrate health monitor for v0.19 anticipatory coherence (FREE tier).

Samples carl_core.SubstrateState from process-level signals (memory, cpu,
fd count). Lazy-imports psutil; degrades gracefully to a single
"runtime" channel when psutil is missing.

Substrate-cognition isomorphism per design doc §2.6: same algebra as
cognitive/conversational substrates — only the channel adapters differ.
"""
from __future__ import annotations

import time
from typing import TYPE_CHECKING

from carl_core import SubstrateChannel, SubstrateState

if TYPE_CHECKING:
    pass


def _psutil_available() -> bool:
    """Detect psutil without importing it (avoids reportUnusedImport)."""
    import importlib.util

    return importlib.util.find_spec("psutil") is not None


def _safe_health(value: float, hi: float = 1.0, lo: float = 0.0) -> float:
    """Clamp value into [lo, hi]."""
    return max(lo, min(hi, value))


class SubstrateMonitor:
    """Polls substrate state and accumulates a rolling history.

    On each ``sample()`` the monitor reads current process-level metrics
    via psutil (when present), computes per-channel drift derivatives
    against the prior sample, and appends a SubstrateState to history.
    """

    def __init__(self, *, max_history: int = 64) -> None:
        self._history: list[SubstrateState] = []
        self._max_history = max_history

    @property
    def history(self) -> list[SubstrateState]:
        return list(self._history)

    @property
    def psutil_available(self) -> bool:
        return _psutil_available()

    def sample(self, timestamp: float | None = None) -> SubstrateState:
        """Take a single substrate snapshot."""
        ts = float(timestamp) if timestamp is not None else time.time()
        channels = self._collect_channels()
        # overall_psi: average of all channel healths
        if channels:
            psi = sum(ch.health for ch in channels.values()) / len(channels)
        else:
            psi = 1.0
        state = SubstrateState(
            timestamp=ts, channels=channels, overall_psi=_safe_health(psi)
        )
        self._append(state)
        return state

    def reset(self) -> None:
        self._history.clear()

    def _append(self, state: SubstrateState) -> None:
        self._history.append(state)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history :]

    def _collect_channels(self) -> dict[str, SubstrateChannel]:
        if not _psutil_available():
            # Fallback: single "runtime" channel reporting steady-state
            return {
                "runtime": SubstrateChannel(
                    name="runtime",
                    health=1.0,
                    drift_rate=0.0,
                    drift_acceleration=0.0,
                )
            }

        import psutil

        proc = psutil.Process()
        # Memory: 1 - (rss / virtual total) -> health
        vmem = psutil.virtual_memory()
        mem_health = _safe_health(1.0 - (vmem.percent / 100.0))
        # CPU: 1 - (system cpu utilization / 100) -> health
        cpu_pct = psutil.cpu_percent(interval=None)  # non-blocking
        cpu_health = _safe_health(1.0 - (cpu_pct / 100.0))
        # File descriptors (POSIX): 1 - (open / soft_limit)
        fd_health = self._fd_health(proc)
        new_chans: dict[str, SubstrateChannel] = {
            "memory": self._channel_from_history("memory", mem_health),
            "compute": self._channel_from_history("compute", cpu_health),
            "fd": self._channel_from_history("fd", fd_health),
        }
        return new_chans

    @staticmethod
    def _fd_health(proc: object) -> float:
        try:
            import resource

            num_open = int(getattr(proc, "num_fds", lambda: 0)())
            soft, _hard = resource.getrlimit(resource.RLIMIT_NOFILE)
            if soft <= 0:
                return 1.0
            return _safe_health(1.0 - (num_open / soft))
        except (AttributeError, OSError, ImportError):
            return 1.0

    def _channel_from_history(self, name: str, current_health: float) -> SubstrateChannel:
        """Build channel with drift derivatives computed against history."""
        prev_healths = [
            s.channels[name].health for s in self._history if name in s.channels
        ]
        if len(prev_healths) >= 1:
            drift_rate = current_health - prev_healths[-1]
        else:
            drift_rate = 0.0
        if len(prev_healths) >= 2:
            prev_rate = prev_healths[-1] - prev_healths[-2]
            drift_acceleration = drift_rate - prev_rate
        else:
            drift_acceleration = 0.0
        return SubstrateChannel(
            name=name,
            health=current_health,
            drift_rate=drift_rate,
            drift_acceleration=drift_acceleration,
        )


__all__ = ["SubstrateMonitor"]
