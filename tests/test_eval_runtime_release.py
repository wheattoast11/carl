"""Evaluation releases cyclic model storage before the next GPU owner runs."""

import gc
import sys
import weakref
from types import SimpleNamespace

import pytest

from carl_studio.eval.runner import EvalConfig, EvalRunner


@pytest.mark.parametrize("fails", [False, True])
def test_single_turn_releases_model_and_purges_allocator_on_each_exit(monkeypatch, fails):
    references = []
    purges = []

    class CyclicModel:
        def __init__(self):
            self.cycle = self

    def load():
        model = CyclicModel()
        references.append(weakref.ref(model))
        return model, object()

    def purge():
        assert references[0]() is None
        purges.append("released")

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True, empty_cache=purge)),
    )
    runner = EvalRunner(EvalConfig(checkpoint="fixture", phase="1", device="cpu"))
    monkeypatch.setattr(runner, "_load_dataset", lambda: [{"id": "heldout"}])
    monkeypatch.setattr(runner, "_load_model_simple", load)

    def generate(*args):
        if fails:
            raise RuntimeError("generation fixture failure")
        return ["answer"]

    monkeypatch.setattr(runner, "_generate_single_turn", generate)
    monkeypatch.setattr(runner, "_compute_metrics", lambda *args: {"chain_completion_rate": 1.0})
    monkeypatch.setattr(
        runner, "_compute_coherence", lambda *args: {"phi_mean": 0.4, "discontinuity_score": 0.5}
    )
    enabled = gc.isenabled()
    gc.disable()
    try:
        if fails:
            with pytest.raises(RuntimeError, match="generation fixture failure"):
                runner._run_single_turn_phase()
        else:
            assert runner._run_single_turn_phase().n_samples == 1
        assert references[0]() is None
        assert purges == ["released"]
    finally:
        if enabled:
            gc.enable()
        gc.collect()
