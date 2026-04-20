"""Phase-transition crystallization gate.

Hosts :class:`PhaseTransitionGate` — the windowed callback that stops
training once ``mean_token_accuracy`` crosses a crystallization threshold
over a rolling window.

The class is resolved at import time via :func:`_get_gate`, which
prefers the canonical implementation in ``carl.py`` (the seed crystal at
the repo root) when it exposes the full ``transformers.TrainerCallback``
surface. If the seed is absent or its class is incomplete, we fall back
to an inline implementation that inherits from
:class:`transformers.TrainerCallback` when available and ``object``
otherwise.

Kept out of :mod:`carl_studio.__init__` to preserve import-time
lightness — nothing here needs to run at ``import carl_studio`` unless
the top-level package re-exports :class:`PhaseTransitionGate`, which it
does via a thin ``from ... import`` only.
"""

from __future__ import annotations

import importlib.util
import pathlib
from typing import Any


def _get_gate() -> type:
    """Resolve :class:`PhaseTransitionGate` — prefer ``carl.py`` seed.

    The seed crystal (``carl.py`` at the repo root) carries the canonical
    implementation. We load it via :mod:`importlib.util` under a private
    module name (``carl_seed``) so it stays out of :data:`sys.modules`
    cache collisions with the ``carl`` package on PyPI. If the seed is
    missing, unreadable, or the class lacks required callback methods,
    we fall back to an inline implementation.
    """
    seed = pathlib.Path(__file__).parent.parent.parent.parent / "carl.py"
    if seed.exists():
        try:
            spec = importlib.util.spec_from_file_location("carl_seed", seed)
            if spec is not None and spec.loader is not None:
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
            # Seed is advisory — fall through to the inline fallback.
            pass

    try:
        from transformers import TrainerCallback as _Base
    except Exception:
        _Base = object  # type: ignore[assignment,misc]

    class _Gate(_Base):  # type: ignore[misc,valid-type]
        def __init__(
            self,
            threshold: float = 0.99,
            window: int = 5,
            min_above: int = 3,
        ) -> None:
            if _Base is not object:
                super().__init__()  # pyright: ignore[reportUnknownMemberType]
            self.threshold = threshold
            self.window = window
            self.min_above = min_above
            self._recent: list[float] = []
            self.triggered = False
            self.trigger_step = -1
            self.peak_entropy = 0.0
            self.peak_entropy_step = -1

        def check(
            self,
            value: float,
            entropy: float = 0.0,
            step: int = 0,
        ) -> bool:
            if entropy > self.peak_entropy:
                self.peak_entropy = entropy
                self.peak_entropy_step = step
            self._recent.append(value)
            if len(self._recent) > self.window:
                self._recent.pop(0)
            if len(self._recent) >= self.min_above and not self.triggered:
                above = sum(1 for v in self._recent if v >= self.threshold)
                if above >= self.min_above:
                    self.triggered = True
                    self.trigger_step = step
                    return True
            return False

        def on_log(
            self,
            args: Any,
            state: Any,
            control: Any,
            logs: Any = None,
            **kwargs: Any,
        ) -> None:
            if logs and self.check(
                logs.get("mean_token_accuracy", 0),
                logs.get("entropy", 0),
                state.global_step,
            ):
                control.should_training_stop = True

        def on_init_end(self, *args: Any, **kwargs: Any) -> None:
            pass

        def on_train_begin(self, *args: Any, **kwargs: Any) -> None:
            pass

        def on_train_end(self, *args: Any, **kwargs: Any) -> None:
            pass

        def on_step_begin(self, *args: Any, **kwargs: Any) -> None:
            pass

        def on_step_end(self, *args: Any, **kwargs: Any) -> None:
            pass

        def on_epoch_begin(self, *args: Any, **kwargs: Any) -> None:
            pass

        def on_epoch_end(self, *args: Any, **kwargs: Any) -> None:
            pass

        def on_save(self, *args: Any, **kwargs: Any) -> None:
            pass

        def on_evaluate(self, *args: Any, **kwargs: Any) -> None:
            pass

        def on_predict(self, *args: Any, **kwargs: Any) -> None:
            pass

    return _Gate


PhaseTransitionGate: type = _get_gate()


__all__ = ["PhaseTransitionGate"]
