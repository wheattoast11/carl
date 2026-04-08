"""Shared math primitives for coherence computation."""
from __future__ import annotations
import math
import numpy as np


def compute_phi(logits: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute order parameter Phi from logits.

    Args:
        logits: [T, V] raw logits

    Returns:
        (phi, probs, entropy) where:
        - phi: [T] order parameter, 1 - H/log|V|
        - probs: [T, V] softmax probabilities
        - entropy: [T] per-token Shannon entropy
    """
    T, V = logits.shape
    log_vocab = math.log(V)

    logits_shifted = logits - logits.max(axis=-1, keepdims=True)
    exp_logits = np.exp(logits_shifted)
    probs = exp_logits / exp_logits.sum(axis=-1, keepdims=True)

    log_probs = np.log(probs + 1e-10)
    entropy = -np.sum(probs * log_probs, axis=-1)

    phi = 1.0 - (entropy / log_vocab)

    return phi, probs, entropy
