"""Test CoherenceProbe and constants."""
import numpy as np
import pytest
from carl_core import KAPPA, SIGMA, DEFECT_THRESHOLD, T_STAR
from carl_core import CoherenceProbe


def test_conservation_constant():
    assert abs(KAPPA * SIGMA - 4.0) < 1e-10

def test_t_star_triadic():
    # d = 3072 = 3 * 2^10 -> T* = 2^16 = 65536
    assert T_STAR(3072) == 65536

def test_t_star_non_triadic():
    # d = 4096 -> T* = int(kappa * 4096) = 87381
    assert T_STAR(4096) == int(KAPPA * 4096)

def test_probe_basic():
    np.random.seed(42)
    probe = CoherenceProbe(vocab_size=1000)
    logits = np.random.randn(64, 1000).astype(np.float32)
    token_ids = logits.argmax(axis=-1)
    snap = probe.measure(logits, token_ids, step=0)
    assert 0 <= snap.phi_mean <= 1
    assert snap.cloud_quality_mean >= 0
    assert snap.n_tokens == 64

def test_probe_sharp_logits_higher_phi():
    np.random.seed(42)
    probe = CoherenceProbe(vocab_size=1000)

    # Random logits
    random_logits = np.random.randn(64, 1000).astype(np.float32)
    random_ids = random_logits.argmax(axis=-1)
    snap_random = probe.measure(random_logits, random_ids, step=0)

    # Sharp logits (one-hot-ish)
    sharp_logits = np.zeros((64, 1000), dtype=np.float32)
    sharp_logits[np.arange(64), np.random.randint(0, 1000, 64)] = 10.0
    sharp_ids = sharp_logits.argmax(axis=-1)
    snap_sharp = probe.measure(sharp_logits, sharp_ids, step=1)

    assert snap_sharp.phi_mean > snap_random.phi_mean
    assert snap_sharp.cloud_quality_mean > snap_random.cloud_quality_mean

def test_probe_scale_coherence_clamped():
    np.random.seed(42)
    probe = CoherenceProbe(vocab_size=1000)
    logits = np.random.randn(128, 1000).astype(np.float32)
    token_ids = logits.argmax(axis=-1)
    snap = probe.measure(logits, token_ids)
    for j, c in snap.scale_coherence.items():
        assert 0.0 <= c <= 1.0, f"Scale {j} coherence {c} out of [0,1]"
