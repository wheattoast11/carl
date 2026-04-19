"""Compat shim: `carl_studio.primitives.coherence_trace` -> `carl_core.coherence_trace`.

Use `from carl_core.coherence_trace import CoherenceTrace, select_traces` in new code.
"""
from carl_core.coherence_trace import *  # noqa: F401, F403
from carl_core.coherence_trace import CoherenceTrace, select_traces

__all__ = ["CoherenceTrace", "select_traces"]
