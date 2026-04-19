"""Compat shim: `carl_studio.primitives.coherence_observer` -> `carl_core.coherence_observer`."""
from carl_core.coherence_observer import *  # noqa: F401, F403
from carl_core.coherence_observer import CoherenceObserver

__all__ = ["CoherenceObserver"]
