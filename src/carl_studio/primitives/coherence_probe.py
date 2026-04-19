"""Compat shim: `carl_studio.primitives.coherence_probe` -> `carl_core.coherence_probe`."""
from carl_core.coherence_probe import *  # noqa: F401, F403
from carl_core.coherence_probe import CoherenceProbe, CoherenceSnapshot

__all__ = ["CoherenceProbe", "CoherenceSnapshot"]
