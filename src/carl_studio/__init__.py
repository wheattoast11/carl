"""CARL Studio -- Coherence-Aware Reinforcement Learning.

The entire paradigm in one import::

    from carl_studio import KAPPA, SIGMA, CoherenceProbe, PhaseTransitionGate
    from carl_studio import CARLTrainer, TrainingConfig

See also: carl.py (300-line seed crystal at the repo root)
"""

__version__ = "0.18.2"

from typing import TYPE_CHECKING

from carl_core.constants import KAPPA, SIGMA, DEFECT_THRESHOLD
from carl_core import CoherenceProbe
from carl_studio.types.config import TrainingConfig
from . import observe

if TYPE_CHECKING:  # Type-only import — runtime resolution goes through __getattr__.
    from carl_studio.training.gates import PhaseTransitionGate

__all__ = [
    "__version__",
    "KAPPA",
    "SIGMA",
    "DEFECT_THRESHOLD",
    "CoherenceProbe",
    "TrainingConfig",
    "observe",
    "PhaseTransitionGate",
    # Lazy: CascadeRewardManager, CARLTrainer, Bundler, ComputeTarget, TrainingMethod
    # Lazy: BenchConfig, BenchSuite, BenchReport, CTI
    # Lazy: SLOTOptimizer, SLOTResult, LoRAMicroUpdate (torch/peft dep)
]


# PhaseTransitionGate lives in carl_studio.training.gates. We lazy-load it
# on first attribute access so ``import carl_studio`` does NOT trigger
# ``carl_studio.training.__init__`` (which pulls cascade/trainer and their
# torch/transformers deps). The gates module itself is dep-light.
def __getattr__(name: str):
    """Lazy imports for objects that pull heavy deps (torch, transformers).

    PhaseTransitionGate is resolved lazily for the same reason — the
    ``carl_studio.training`` package ``__init__`` eagerly imports cascade
    and trainer modules that depend on torch/trl. Accessing
    ``carl_studio.PhaseTransitionGate`` triggers import of the
    ``carl_studio.training.gates`` submodule only on first use.
    """
    if name == "PhaseTransitionGate":
        import importlib

        mod = importlib.import_module("carl_studio.training.gates")
        gate = mod.PhaseTransitionGate
        # Cache on the package so subsequent accesses skip the __getattr__ hop.
        globals()["PhaseTransitionGate"] = gate
        return gate

    _lazy = {
        "CascadeRewardManager": "carl_studio.training.cascade",
        "ComputeTarget": "carl_studio.types.config",
        "TrainingMethod": "carl_studio.types.config",
        "CARLTrainer": "carl_studio.training.trainer",
        "observe": "carl_studio.observe",
        "Bundler": "carl_studio.bundler",
        "EvalConfig": "carl_studio.eval.runner",
        "EvalRunner": "carl_studio.eval.runner",
        "EvalReport": "carl_studio.eval.runner",
        "EvalGate": "carl_studio.eval.runner",
        "BenchConfig": "carl_studio.bench.suite",
        "BenchSuite": "carl_studio.bench.suite",
        "BenchReport": "carl_studio.bench.suite",
        "CTI": "carl_studio.bench.suite",
        "SLOTOptimizer": "carl_studio.ttt.slot",
        "SLOTResult": "carl_studio.ttt.slot",
        "LoRAMicroUpdate": "carl_studio.ttt.slot",
    }
    if name in _lazy:
        import importlib

        mod = importlib.import_module(_lazy[name])
        if name == "observe":
            return mod
        return getattr(mod, name)
    raise AttributeError(f"module 'carl_studio' has no attribute {name!r}")
