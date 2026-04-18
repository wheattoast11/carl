"""CARL Studio -- Coherence-Aware Reinforcement Learning.

The entire paradigm in one import::

    from carl_studio import KAPPA, SIGMA, CoherenceProbe, PhaseTransitionGate
    from carl_studio import CARLTrainer, TrainingConfig

See also: carl.py (300-line seed crystal at the repo root)
"""

__version__ = "0.4.0"

from carl_core.constants import KAPPA, SIGMA, DEFECT_THRESHOLD
from carl_core import CoherenceProbe
from carl_studio.types.config import TrainingConfig
from . import observe

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


# PhaseTransitionGate is lightweight (no torch dep) — import eagerly
# This is the core abstraction: the windowed gate that detects crystallization
def _get_gate():
    """Import from carl.py seed if available, otherwise inline."""
    import importlib.util
    import pathlib

    seed = pathlib.Path(__file__).parent.parent.parent / "carl.py"
    if seed.exists():
        try:
            spec = importlib.util.spec_from_file_location("carl_seed", seed)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            gate_cls = mod.PhaseTransitionGate
            required_methods = (
                "on_init_end",
                "on_train_begin",
                "on_train_end",
                "on_step_begin",
                "on_step_end",
                "on_log",
                "on_epoch_begin",
                "on_epoch_end",
                "on_save",
                "on_evaluate",
                "on_predict",
            )
            if all(hasattr(gate_cls, method) for method in required_methods):
                return gate_cls
        except Exception:
            pass
    # Fallback: inline minimal gate (inherits TrainerCallback if available)
    try:
        from transformers import TrainerCallback as _Base
    except Exception:
        _Base = object

    class _Gate(_Base):
        def __init__(self, threshold=0.99, window=5, min_above=3):
            if _Base is not object:
                super().__init__()
            self.threshold, self.window, self.min_above = threshold, window, min_above
            self._recent, self.triggered, self.trigger_step = [], False, -1
            self.peak_entropy, self.peak_entropy_step = 0.0, -1

        def check(self, value, entropy=0.0, step=0):
            if entropy > self.peak_entropy:
                self.peak_entropy, self.peak_entropy_step = entropy, step
            self._recent.append(value)
            if len(self._recent) > self.window:
                self._recent.pop(0)
            if len(self._recent) >= self.min_above and not self.triggered:
                if sum(1 for v in self._recent if v >= self.threshold) >= self.min_above:
                    self.triggered, self.trigger_step = True, step
                    return True
            return False

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and self.check(
                logs.get("mean_token_accuracy", 0), logs.get("entropy", 0), state.global_step
            ):
                control.should_training_stop = True

        def on_init_end(self, *args, **kwargs):
            pass

        def on_train_begin(self, *args, **kwargs):
            pass

        def on_train_end(self, *args, **kwargs):
            pass

        def on_step_begin(self, *args, **kwargs):
            pass

        def on_step_end(self, *args, **kwargs):
            pass

        def on_epoch_begin(self, *args, **kwargs):
            pass

        def on_epoch_end(self, *args, **kwargs):
            pass

        def on_save(self, *args, **kwargs):
            pass

        def on_evaluate(self, *args, **kwargs):
            pass

        def on_predict(self, *args, **kwargs):
            pass

    return _Gate


PhaseTransitionGate = _get_gate()


def __getattr__(name: str):
    """Lazy imports for objects that pull heavy deps (torch, transformers)."""
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
