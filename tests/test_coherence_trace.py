"""Test CoherenceTrace — the foundational CARL primitive."""
import json

import numpy as np
import pytest

from carl_studio.primitives.coherence_trace import CoherenceTrace, select_traces
from carl_studio.primitives.constants import DEFECT_THRESHOLD


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_random_logits(T: int = 64, V: int = 1000, seed: int = 42):
    rng = np.random.default_rng(seed)
    logits = rng.standard_normal((T, V)).astype(np.float32)
    token_ids = logits.argmax(axis=-1)
    return logits, token_ids


def _make_sharp_logits(T: int = 64, V: int = 1000, seed: int = 42):
    rng = np.random.default_rng(seed)
    logits = np.zeros((T, V), dtype=np.float32)
    logits[np.arange(T), rng.integers(0, V, T)] = 10.0
    token_ids = logits.argmax(axis=-1)
    return logits, token_ids


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestFromLogits:
    def test_basic_shape(self):
        logits, ids = _make_random_logits(T=64, V=1000)
        trace = CoherenceTrace.from_logits(logits, ids)
        assert trace.n_tokens == 64
        assert trace.vocab_size == 1000
        assert trace.phi.shape == (64,)
        assert trace.entropy.shape == (64,)
        assert trace.selected_prob.shape == (64,)
        assert trace.delta_phi.shape == (63,)  # T-1

    def test_phi_range(self):
        logits, ids = _make_random_logits()
        trace = CoherenceTrace.from_logits(logits, ids)
        assert np.all(trace.phi >= 0)
        assert np.all(trace.phi <= 1)

    def test_selected_prob_range(self):
        logits, ids = _make_random_logits()
        trace = CoherenceTrace.from_logits(logits, ids)
        assert np.all(trace.selected_prob >= 0)
        assert np.all(trace.selected_prob <= 1)

    def test_sharp_logits_higher_phi(self):
        random_logits, random_ids = _make_random_logits()
        sharp_logits, sharp_ids = _make_sharp_logits()

        trace_random = CoherenceTrace.from_logits(random_logits, random_ids)
        trace_sharp = CoherenceTrace.from_logits(sharp_logits, sharp_ids)

        assert trace_sharp.phi_mean > trace_random.phi_mean

    def test_cross_batch_continuity(self):
        logits, ids = _make_random_logits(T=32)
        trace1 = CoherenceTrace.from_logits(logits, ids, step=0)
        last_phi = float(trace1.phi[-1])

        logits2, ids2 = _make_random_logits(T=32, seed=99)
        trace2 = CoherenceTrace.from_logits(logits2, ids2, step=1, prev_phi=last_phi)

        # With prev_phi, delta_phi has T elements (boundary + T-1 internal)
        assert trace2.delta_phi.shape == (32,)
        # First element should be phi[0] - prev_phi
        expected_boundary = trace2.phi[0] - last_phi
        assert abs(trace2.delta_phi[0] - expected_boundary) < 1e-6

    def test_metadata(self):
        logits, ids = _make_random_logits()
        trace = CoherenceTrace.from_logits(logits, ids, step=42, sample_idx=3)
        assert trace.step == 42
        assert trace.sample_idx == 3


class TestFromEntropy:
    def test_matches_from_logits(self):
        """from_entropy should produce identical phi and derived metrics."""
        import math
        logits, ids = _make_random_logits(T=64, V=1000)

        # Compute what from_logits would compute
        trace_full = CoherenceTrace.from_logits(logits, ids)

        # Simulate TRL's output: entropy + selected logprobs
        entropy = trace_full.entropy
        selected_logprobs = np.log(trace_full.selected_prob + 1e-10)

        trace_trl = CoherenceTrace.from_entropy(
            entropy=entropy,
            selected_logprobs=selected_logprobs,
            vocab_size=1000,
        )

        # Phi should be identical
        np.testing.assert_allclose(trace_trl.phi, trace_full.phi, atol=1e-6)

        # Derived metrics should be very close
        assert abs(trace_trl.phi_mean - trace_full.phi_mean) < 1e-6
        assert abs(trace_trl.multiscale_coherence - trace_full.multiscale_coherence) < 1e-6
        assert abs(trace_trl.discontinuity_score - trace_full.discontinuity_score) < 1e-6

    def test_selected_prob_from_logprobs(self):
        logits, ids = _make_random_logits()
        trace_full = CoherenceTrace.from_logits(logits, ids)

        selected_logprobs = np.log(trace_full.selected_prob + 1e-10)
        trace_trl = CoherenceTrace.from_entropy(
            entropy=trace_full.entropy,
            selected_logprobs=selected_logprobs,
            vocab_size=1000,
        )

        np.testing.assert_allclose(
            trace_trl.selected_prob, trace_full.selected_prob, atol=1e-5
        )


# ---------------------------------------------------------------------------
# Derived metrics
# ---------------------------------------------------------------------------

class TestMetrics:
    def test_cloud_quality_range(self):
        logits, ids = _make_random_logits()
        trace = CoherenceTrace.from_logits(logits, ids)
        assert 0 <= trace.cloud_quality <= 1

    def test_multiscale_coherence_range(self):
        logits, ids = _make_random_logits(T=128)
        trace = CoherenceTrace.from_logits(logits, ids)
        assert 0 <= trace.multiscale_coherence <= 1

    def test_discontinuity_score_range(self):
        logits, ids = _make_random_logits()
        trace = CoherenceTrace.from_logits(logits, ids)
        assert 0 <= trace.discontinuity_score <= 1

    def test_carl_reward_range(self):
        logits, ids = _make_random_logits()
        trace = CoherenceTrace.from_logits(logits, ids)
        reward = trace.carl_reward()
        assert 0 <= reward <= 1

    def test_carl_reward_custom_weights(self):
        logits, ids = _make_random_logits()
        trace = CoherenceTrace.from_logits(logits, ids)
        r_default = trace.carl_reward()
        r_custom = trace.carl_reward(w_coherence=1.0, w_cloud=0.0, w_discontinuity=0.0)
        assert r_custom == trace.multiscale_coherence

    def test_defect_counts(self):
        logits, ids = _make_random_logits(T=256)
        trace = CoherenceTrace.from_logits(logits, ids)
        assert trace.n_defects == trace.n_crystallizations + trace.n_meltings
        assert trace.defect_density == trace.n_defects / 256

    def test_scale_coherence_dict(self):
        logits, ids = _make_random_logits(T=128)
        trace = CoherenceTrace.from_logits(logits, ids)
        sc = trace.scale_coherence
        assert isinstance(sc, dict)
        assert all(0 <= v <= 1 for v in sc.values())
        assert 0 in sc  # scale 0 should always exist for T >= 1

    def test_surprisal_positive(self):
        logits, ids = _make_random_logits()
        trace = CoherenceTrace.from_logits(logits, ids)
        assert trace.surprisal_mean > 0

    def test_caching(self):
        """Accessing the same property twice should return identical values."""
        logits, ids = _make_random_logits()
        trace = CoherenceTrace.from_logits(logits, ids)
        r1 = trace.carl_reward()
        r2 = trace.carl_reward()
        assert r1 == r2
        assert trace._cache  # cache should be populated


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

class TestSerialization:
    def test_round_trip(self):
        logits, ids = _make_random_logits()
        trace = CoherenceTrace.from_logits(logits, ids, step=5, sample_idx=2)

        d = trace.to_dict(include_arrays=True)
        restored = CoherenceTrace.from_dict(d)

        np.testing.assert_allclose(restored.phi, trace.phi, atol=1e-10)
        np.testing.assert_allclose(restored.entropy, trace.entropy, atol=1e-10)
        np.testing.assert_allclose(restored.selected_prob, trace.selected_prob, atol=1e-10)
        np.testing.assert_allclose(restored.delta_phi, trace.delta_phi, atol=1e-10)
        assert restored.step == 5
        assert restored.sample_idx == 2
        assert restored.vocab_size == trace.vocab_size

    def test_compact_dict_no_arrays(self):
        logits, ids = _make_random_logits()
        trace = CoherenceTrace.from_logits(logits, ids)
        d = trace.to_dict(include_arrays=False)
        assert "phi" not in d
        assert "entropy" not in d
        assert "phi_mean" in d
        assert "carl_reward" in d

    def test_json_serializable(self):
        logits, ids = _make_random_logits()
        trace = CoherenceTrace.from_logits(logits, ids)
        d = trace.to_dict(include_arrays=True)
        s = json.dumps(d)
        assert isinstance(s, str)
        parsed = json.loads(s)
        assert parsed["n_tokens"] == 64


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------

class TestBackwardCompat:
    def test_to_snapshot(self):
        logits, ids = _make_random_logits()
        trace = CoherenceTrace.from_logits(logits, ids, step=10)
        snap = trace.to_snapshot()

        assert snap.step == 10
        assert snap.n_tokens == 64
        assert abs(snap.phi_mean - trace.phi_mean) < 1e-6
        assert abs(snap.cloud_quality_mean - trace.cloud_quality) < 1e-6
        assert snap.n_defects == trace.n_defects

    def test_probe_measure_returns_snapshot(self):
        """CoherenceProbe.measure() still returns CoherenceSnapshot."""
        from carl_studio.primitives.coherence_probe import CoherenceProbe
        logits, ids = _make_random_logits()
        probe = CoherenceProbe(vocab_size=1000)
        snap = probe.measure(logits, ids, step=0)
        assert hasattr(snap, "phi_mean")
        assert hasattr(snap, "scale_coherence")

    def test_probe_measure_trace(self):
        """CoherenceProbe.measure_trace() returns CoherenceTrace."""
        from carl_studio.primitives.coherence_probe import CoherenceProbe
        logits, ids = _make_random_logits()
        probe = CoherenceProbe(vocab_size=1000)
        trace = probe.measure_trace(logits, ids, step=0)
        assert isinstance(trace, CoherenceTrace)
        assert trace.n_tokens == 64


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

class TestVisualization:
    def test_sparkline(self):
        logits, ids = _make_random_logits()
        trace = CoherenceTrace.from_logits(logits, ids)
        spark = trace.sparkline(width=40)
        assert len(spark) <= 40
        assert len(spark) > 0

    def test_entropy_sparkline(self):
        logits, ids = _make_random_logits()
        trace = CoherenceTrace.from_logits(logits, ids)
        spark = trace.entropy_sparkline(width=40)
        assert len(spark) <= 40

    def test_defect_map(self):
        logits, ids = _make_random_logits()
        trace = CoherenceTrace.from_logits(logits, ids)
        dm = trace.defect_map(width=40)
        assert len(dm) <= 40
        # Should contain only +, -, and middle dot
        valid_chars = {"+", "-", "\u00b7"}
        assert all(c in valid_chars for c in dm)

    def test_repr(self):
        logits, ids = _make_random_logits()
        trace = CoherenceTrace.from_logits(logits, ids)
        r = repr(trace)
        assert "CoherenceTrace" in r
        assert "T=64" in r


# ---------------------------------------------------------------------------
# Batch utilities
# ---------------------------------------------------------------------------

class TestSelectTraces:
    def test_small_batch(self):
        logits, ids = _make_random_logits(seed=0)
        traces = [CoherenceTrace.from_logits(logits, ids)]
        selected = select_traces(traces, k=4)
        assert len(selected) == 1  # batch < k

    def test_selects_k(self):
        traces = []
        for i in range(10):
            logits, ids = _make_random_logits(seed=i)
            traces.append(CoherenceTrace.from_logits(logits, ids, sample_idx=i))
        selected = select_traces(traces, k=4)
        assert len(selected) == 4

    def test_includes_best_and_worst(self):
        traces = []
        for i in range(10):
            logits, ids = _make_random_logits(seed=i)
            traces.append(CoherenceTrace.from_logits(logits, ids, sample_idx=i))

        selected = select_traces(traces, k=4)
        rewards = [t.carl_reward() for t in traces]
        selected_rewards = [t.carl_reward() for t in selected]

        assert min(rewards) in selected_rewards
        assert max(rewards) in selected_rewards
