"""Back-compat shim — CoherenceObserver lives in ``carl_core.coherence_observer``."""
from __future__ import annotations

from carl_core.coherence_observer import OBSERVER_SYSTEM_PROMPT, CoherenceObserver

__all__ = ["CoherenceObserver", "OBSERVER_SYSTEM_PROMPT"]
