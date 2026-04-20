"""Training subpackage — lazy-loaded public surface.

All heavy symbols (CoherenceMonitorCallback, CascadeRewardManager,
ResonanceLRCallback, make_multi_environment, etc.) live in submodules
that depend on torch/transformers. Importing :mod:`carl_studio.training`
directly MUST NOT drag those modules in so that ``import carl_studio``
stays lightweight — the top-level package re-exports
:class:`PhaseTransitionGate` from :mod:`carl_studio.training.gates`, and
resolving that name must not also resolve trainer/cascade.

Public names continue to work through module-level ``__getattr__``:
``from carl_studio.training import CascadeRewardManager`` still succeeds,
it just loads the cascade module on first access.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # Type-only imports for static checkers; never executed at runtime.
    from carl_studio.training.callbacks import CoherenceMonitorCallback
    from carl_studio.training.cascade import CascadeCallback, CascadeRewardManager
    from carl_studio.training.gates import PhaseTransitionGate
    from carl_studio.training.lr_resonance import ResonanceLRCallback
    from carl_studio.training.multi_env import make_multi_environment
    from carl_studio.training.trace_callback import CoherenceTraceCallback

__all__ = [
    "CoherenceMonitorCallback",
    "CoherenceTraceCallback",
    "CascadeRewardManager",
    "CascadeCallback",
    "ResonanceLRCallback",
    "make_multi_environment",
    "PhaseTransitionGate",
]


_LAZY: dict[str, str] = {
    "CoherenceMonitorCallback": "carl_studio.training.callbacks",
    "CoherenceTraceCallback": "carl_studio.training.trace_callback",
    "CascadeRewardManager": "carl_studio.training.cascade",
    "CascadeCallback": "carl_studio.training.cascade",
    "ResonanceLRCallback": "carl_studio.training.lr_resonance",
    "make_multi_environment": "carl_studio.training.multi_env",
    "PhaseTransitionGate": "carl_studio.training.gates",
}


def __getattr__(name: str) -> Any:
    module_path = _LAZY.get(name)
    if module_path is None:
        raise AttributeError(
            f"module 'carl_studio.training' has no attribute {name!r}"
        )
    import importlib

    mod = importlib.import_module(module_path)
    value = getattr(mod, name)
    globals()[name] = value
    return value
