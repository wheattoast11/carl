"""Back-compat shim — CoherenceTrace lives in ``carl_core.coherence_trace``."""
from __future__ import annotations

from carl_core.coherence_trace import CoherenceTrace, select_traces

__all__ = ["CoherenceTrace", "select_traces"]
