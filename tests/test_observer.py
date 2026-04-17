"""Tests for CoherenceObserver -- no API key needed for accumulation."""

import numpy as np
import pytest
from carl_core import CoherenceProbe, CoherenceObserver


def test_observer_accumulates():
    probe = CoherenceProbe(vocab_size=1000)
    observer = CoherenceObserver(observe_every=5)

    results = []
    for step in range(10):
        logits = np.random.randn(32, 1000).astype(np.float32)
        snap = probe.measure(logits, logits.argmax(-1), step=step)
        result = observer.ingest(snap)
        if result is not None:
            results.append(result)

    # All 10 snapshots should be in the buffer
    assert len(observer.buffer) == 10


def test_observer_buffer_window():
    observer = CoherenceObserver(observe_every=100, window_size=5)
    probe = CoherenceProbe(vocab_size=100)

    for step in range(10):
        logits = np.random.randn(16, 100).astype(np.float32)
        snap = probe.measure(logits, logits.argmax(-1), step=step)
        observer.ingest(snap)

    # Buffer should be capped at window_size
    assert len(observer.buffer) <= 5


def test_observer_no_api_key_graceful():
    """Without API key, observer should not crash on ingest."""
    # Ensure ANTHROPIC_API_KEY is not set for this test
    import os
    old_key = os.environ.pop("ANTHROPIC_API_KEY", None)
    try:
        observer = CoherenceObserver(api_key=None, observe_every=2)
        probe = CoherenceProbe(vocab_size=100)

        for step in range(5):
            logits = np.random.randn(16, 100).astype(np.float32)
            snap = probe.measure(logits, logits.argmax(-1), step=step)
            result = observer.ingest(snap)
            # Should either return None or a degraded assessment, never crash
            if result is not None:
                # Degraded mode returns a dict with WARNING status
                assert result.get("status") in ("WARNING", "HEALTHY")
    finally:
        if old_key is not None:
            os.environ["ANTHROPIC_API_KEY"] = old_key


def test_observer_no_api_key_returns_degraded_assessment():
    """Without API key, force_observe should return a degraded assessment."""
    import os
    old_key = os.environ.pop("ANTHROPIC_API_KEY", None)
    try:
        observer = CoherenceObserver(api_key=None, observe_every=100)
        probe = CoherenceProbe(vocab_size=100)

        logits = np.random.randn(16, 100).astype(np.float32)
        snap = probe.measure(logits, logits.argmax(-1), step=0)
        observer.ingest(snap)

        assessment = observer.force_observe()
        assert assessment["status"] == "WARNING"
        assert "ANTHROPIC_API_KEY" in assessment["diagnosis"]
    finally:
        if old_key is not None:
            os.environ["ANTHROPIC_API_KEY"] = old_key


def test_observer_observe_every_triggers():
    """Observer should trigger observation at the correct interval."""
    import os
    old_key = os.environ.pop("ANTHROPIC_API_KEY", None)
    try:
        observer = CoherenceObserver(api_key=None, observe_every=3)
        probe = CoherenceProbe(vocab_size=100)

        results = []
        for step in range(7):
            logits = np.random.randn(8, 100).astype(np.float32)
            snap = probe.measure(logits, logits.argmax(-1), step=step)
            result = observer.ingest(snap)
            if result is not None:
                results.append(result)

        # observe_every=3 => triggers at step 2 (3rd ingest) and step 5 (6th ingest)
        assert len(results) == 2
    finally:
        if old_key is not None:
            os.environ["ANTHROPIC_API_KEY"] = old_key


def test_observer_since_last_observe_resets():
    """Internal counter should reset after each observation."""
    import os
    old_key = os.environ.pop("ANTHROPIC_API_KEY", None)
    try:
        observer = CoherenceObserver(api_key=None, observe_every=3)
        probe = CoherenceProbe(vocab_size=100)

        for step in range(4):
            logits = np.random.randn(8, 100).astype(np.float32)
            snap = probe.measure(logits, logits.argmax(-1), step=step)
            observer.ingest(snap)

        # After 4 ingests with observe_every=3: triggered at 3, counter reset, now at 1
        assert observer._since_last_observe == 1
    finally:
        if old_key is not None:
            os.environ["ANTHROPIC_API_KEY"] = old_key


def test_observer_history_grows():
    """Each observation should append to history."""
    import os
    old_key = os.environ.pop("ANTHROPIC_API_KEY", None)
    try:
        observer = CoherenceObserver(api_key=None, observe_every=2)
        probe = CoherenceProbe(vocab_size=100)

        for step in range(6):
            logits = np.random.randn(8, 100).astype(np.float32)
            snap = probe.measure(logits, logits.argmax(-1), step=step)
            observer.ingest(snap)

        # observe_every=2 => triggers at step 1, 3, 5 => 3 history entries
        assert len(observer.history) == 3
    finally:
        if old_key is not None:
            os.environ["ANTHROPIC_API_KEY"] = old_key
