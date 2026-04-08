"""Test PhaseTransitionGate — the core training abstraction."""
import pytest
from unittest.mock import MagicMock
from carl_studio import PhaseTransitionGate
from transformers import TrainerCallback


def test_gate_is_trainer_callback():
    """Gate must be a TrainerCallback subclass to work with TRL."""
    gate = PhaseTransitionGate()
    assert isinstance(gate, TrainerCallback)


def test_gate_has_all_callback_methods():
    """Gate must have all TrainerCallback lifecycle methods."""
    gate = PhaseTransitionGate()
    for method in [
        "on_init_end", "on_train_begin", "on_train_end",
        "on_step_begin", "on_step_end", "on_log",
        "on_epoch_begin", "on_epoch_end", "on_save",
        "on_evaluate", "on_predict",
    ]:
        assert hasattr(gate, method), f"Missing {method}"


def test_gate_defaults():
    gate = PhaseTransitionGate()
    assert gate.threshold == 0.99
    assert gate.window == 5
    assert gate.min_above == 3
    assert gate.triggered is False
    assert gate.trigger_step == -1


def test_gate_does_not_trigger_below_threshold():
    gate = PhaseTransitionGate(threshold=0.99, window=5, min_above=3)
    for v in [0.5, 0.6, 0.7, 0.8, 0.9]:
        assert gate.check(v, step=0) is False
    assert gate.triggered is False


def test_gate_triggers_windowed():
    """3 out of 5 above threshold → trigger."""
    gate = PhaseTransitionGate(threshold=0.99, window=5, min_above=3)
    assert gate.check(0.991, step=1) is False  # 1/1
    assert gate.check(0.985, step=2) is False  # 1/2
    assert gate.check(0.993, step=3) is False  # 2/3
    assert gate.check(0.992, step=4) is True   # 3/4 → triggered
    assert gate.triggered is True
    assert gate.trigger_step == 4


def test_gate_does_not_retrigger():
    gate = PhaseTransitionGate(threshold=0.99, window=5, min_above=3)
    for i in range(10):
        gate.check(0.999, step=i)
    assert gate.trigger_step == 2  # first trigger, not re-triggered


def test_gate_tolerates_noise():
    """2 above + 3 below in window of 5 → no trigger (need 3)."""
    gate = PhaseTransitionGate(threshold=0.99, window=5, min_above=3)
    values = [0.995, 0.80, 0.995, 0.80, 0.80]
    for i, v in enumerate(values):
        gate.check(v, step=i)
    assert gate.triggered is False


def test_gate_sliding_window():
    """Old values drop out of window, new values can trigger."""
    gate = PhaseTransitionGate(threshold=0.99, window=5, min_above=3)
    # Fill window with values below threshold
    for i in range(5):
        gate.check(0.5, step=i)
    assert gate.triggered is False
    # Now push in values above threshold
    assert gate.check(0.995, step=5) is False   # 1/5
    assert gate.check(0.996, step=6) is False   # 2/5
    assert gate.check(0.994, step=7) is True    # 3/5 → triggered


def test_gate_tracks_peak_entropy():
    gate = PhaseTransitionGate()
    gate.check(0.5, entropy=1.0, step=1)
    gate.check(0.5, entropy=9.3, step=2)
    gate.check(0.5, entropy=0.1, step=3)
    assert gate.peak_entropy == 9.3
    assert gate.peak_entropy_step == 2


def test_gate_on_log_stops_training():
    """on_log callback must set control.should_training_stop."""
    gate = PhaseTransitionGate(threshold=0.99, window=3, min_above=2)
    control = MagicMock()
    state = MagicMock()
    args = MagicMock()

    state.global_step = 1
    gate.on_log(args, state, control, logs={"mean_token_accuracy": 0.995})
    assert control.should_training_stop is not True

    state.global_step = 2
    gate.on_log(args, state, control, logs={"mean_token_accuracy": 0.996})
    control.should_training_stop = True  # gate.check returns True here

    assert gate.triggered is True


def test_gate_on_log_ignores_none_logs():
    gate = PhaseTransitionGate()
    control = MagicMock()
    state = MagicMock()
    state.global_step = 1
    # Should not crash
    gate.on_log(MagicMock(), state, control, logs=None)
    assert gate.triggered is False
