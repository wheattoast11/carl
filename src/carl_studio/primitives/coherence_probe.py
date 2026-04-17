"""Back-compat shim — CoherenceProbe lives in ``carl_core.coherence_probe``."""
from __future__ import annotations

from carl_core.coherence_probe import CoherenceProbe, CoherenceSnapshot

__all__ = ["CoherenceProbe", "CoherenceSnapshot"]
