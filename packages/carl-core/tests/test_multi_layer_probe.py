"""Tests for CoherenceProbe.measure_multi_layer + LayeredTrace (SEM-004).

Covers the residual-stream / attention-entropy probe added to observe the
internal trajectory — not just the terminal distribution.
"""
from __future__ import annotations

import math
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from carl_core.coherence_probe import (
    CARL_LAYER_PROBE_ENABLED,
    CoherenceProbe,
    CoherenceSnapshot,
)
from carl_core.coherence_trace import CoherenceTrace, LayeredTrace


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


VOCAB_SIZE = 16
T = 6
D = 8
N_LAYERS = 4


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(12345)


@pytest.fixture
def probe() -> CoherenceProbe:
    return CoherenceProbe(vocab_size=VOCAB_SIZE)


@pytest.fixture
def hidden_states(rng: np.random.Generator) -> list[np.ndarray]:
    # Start from a base and evolve with small residual updates so adjacent
    # layers have high but non-degenerate cosine similarity.
    base = rng.standard_normal((T, D)).astype(np.float64)
    layers = [base]
    for _ in range(N_LAYERS - 1):
        perturb = 0.1 * rng.standard_normal((T, D)).astype(np.float64)
        layers.append(layers[-1] + perturb)
    return layers


@pytest.fixture
def logits(rng: np.random.Generator) -> np.ndarray:
    return rng.standard_normal((T, VOCAB_SIZE)).astype(np.float64)


@pytest.fixture
def token_ids(rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, VOCAB_SIZE, size=T).astype(np.int64)


@pytest.fixture
def attentions(rng: np.random.Generator) -> list[np.ndarray]:
    # Shape [H, T, T] per layer, softmaxed over the last axis.
    H = 3
    out: list[np.ndarray] = []
    for _ in range(N_LAYERS):
        raw = rng.standard_normal((H, T, T))
        exp = np.exp(raw - raw.max(axis=-1, keepdims=True))
        out.append(exp / exp.sum(axis=-1, keepdims=True))
    return out


# ---------------------------------------------------------------------------
# core contract
# ---------------------------------------------------------------------------


def test_measure_multi_layer_returns_layered_trace(
    probe: CoherenceProbe,
    hidden_states: list[np.ndarray],
    logits: np.ndarray,
    token_ids: np.ndarray,
) -> None:
    result = probe.measure_multi_layer(hidden_states, logits, token_ids)
    assert isinstance(result, LayeredTrace)
    assert isinstance(result.final, CoherenceTrace)
    assert result.n_layers == N_LAYERS - 1


def test_layered_trace_residual_cos_length_matches_layer_count(
    probe: CoherenceProbe,
    hidden_states: list[np.ndarray],
    logits: np.ndarray,
    token_ids: np.ndarray,
) -> None:
    result = probe.measure_multi_layer(hidden_states, logits, token_ids)
    # Pairs are (h0,h1), (h1,h2), (h2,h3) => len == N_LAYERS - 1
    assert len(result.layer_residual_cos) == len(hidden_states) - 1
    assert result.n_layers == len(result.layer_residual_cos)
    # Every cosine is in [-1, 1] (+ numerical slack).
    for c in result.layer_residual_cos:
        assert -1.0 - 1e-6 <= c <= 1.0 + 1e-6


def test_layered_trace_residual_mean_is_mean_of_per_layer(
    probe: CoherenceProbe,
    hidden_states: list[np.ndarray],
    logits: np.ndarray,
    token_ids: np.ndarray,
) -> None:
    result = probe.measure_multi_layer(hidden_states, logits, token_ids)
    expected_mean = sum(result.layer_residual_cos) / len(result.layer_residual_cos)
    assert result.residual_mean == pytest.approx(expected_mean)
    assert result.residual_min == pytest.approx(min(result.layer_residual_cos))
    assert result.residual_trajectory == list(result.layer_residual_cos)
    # residual_trajectory is a copy, not the same list object.
    assert result.residual_trajectory is not result.layer_residual_cos


def test_measure_multi_layer_with_attentions_populates_attention_entropy(
    probe: CoherenceProbe,
    hidden_states: list[np.ndarray],
    logits: np.ndarray,
    token_ids: np.ndarray,
    attentions: list[np.ndarray],
) -> None:
    result = probe.measure_multi_layer(
        hidden_states,
        logits,
        token_ids,
        include_attention_entropy=True,
        attentions=attentions,
    )
    assert result.layer_attention_entropy is not None
    assert len(result.layer_attention_entropy) == len(attentions)
    # Entropy is non-negative and bounded by log(T) for a length-T distribution.
    upper = math.log(T) + 1e-6
    for e in result.layer_attention_entropy:
        assert 0.0 <= e <= upper


def test_measure_multi_layer_without_attentions_entropy_is_none(
    probe: CoherenceProbe,
    hidden_states: list[np.ndarray],
    logits: np.ndarray,
    token_ids: np.ndarray,
) -> None:
    # Flag on but attentions omitted — entropy stays None.
    result_a = probe.measure_multi_layer(
        hidden_states,
        logits,
        token_ids,
        include_attention_entropy=True,
        attentions=None,
    )
    assert result_a.layer_attention_entropy is None

    # Default path — entropy stays None.
    result_b = probe.measure_multi_layer(hidden_states, logits, token_ids)
    assert result_b.layer_attention_entropy is None


def test_measure_multi_layer_handles_numpy_inputs(
    probe: CoherenceProbe,
    hidden_states: list[np.ndarray],
    logits: np.ndarray,
    token_ids: np.ndarray,
) -> None:
    # All inputs are numpy — should work without conversion hiccups.
    result = probe.measure_multi_layer(hidden_states, logits, token_ids)
    assert isinstance(result, LayeredTrace)
    assert result.final.n_tokens == T


def test_measure_multi_layer_handles_torch_tensors_if_available(
    probe: CoherenceProbe,
    hidden_states: list[np.ndarray],
    logits: np.ndarray,
    token_ids: np.ndarray,
    attentions: list[np.ndarray],
) -> None:
    torch = pytest.importorskip("torch")

    torch_hidden = [torch.from_numpy(h.copy()).float() for h in hidden_states]
    torch_logits = torch.from_numpy(logits.copy()).float()
    torch_tokens = torch.from_numpy(token_ids.copy()).long()
    torch_attentions = [torch.from_numpy(a.copy()).float() for a in attentions]

    result = probe.measure_multi_layer(
        torch_hidden,
        torch_logits,
        torch_tokens,
        include_attention_entropy=True,
        attentions=torch_attentions,
    )
    assert isinstance(result, LayeredTrace)
    assert result.n_layers == len(hidden_states) - 1
    assert result.layer_attention_entropy is not None
    assert len(result.layer_attention_entropy) == len(attentions)

    # Torch path should match numpy path exactly (modulo float32 precision).
    numpy_result = probe.measure_multi_layer(
        hidden_states,
        logits,
        token_ids,
        include_attention_entropy=True,
        attentions=attentions,
    )
    for torch_c, numpy_c in zip(
        result.layer_residual_cos, numpy_result.layer_residual_cos
    ):
        assert torch_c == pytest.approx(numpy_c, abs=1e-5)


# ---------------------------------------------------------------------------
# backward-compat guarantee
# ---------------------------------------------------------------------------


def test_legacy_measure_unchanged(
    probe: CoherenceProbe,
    logits: np.ndarray,
    token_ids: np.ndarray,
) -> None:
    # The pre-existing fast path still returns a CoherenceSnapshot with all
    # its required fields. Calling it must NOT require hidden_states or any
    # new parameters.
    snap = probe.measure(logits, token_ids, step=7)
    assert isinstance(snap, CoherenceSnapshot)
    assert snap.step == 7
    assert snap.n_tokens == T
    # Optional advantage path still works.
    snap_with_adv = probe.measure(
        logits, token_ids, step=7, advantages=np.zeros(T)
    )
    assert snap_with_adv.advantage_mean == pytest.approx(0.0)
    # The trace path also still works.
    trace = probe.measure_trace(logits, token_ids, step=7)
    assert isinstance(trace, CoherenceTrace)
    assert trace.n_tokens == T


# ---------------------------------------------------------------------------
# dataclass invariants
# ---------------------------------------------------------------------------


def test_layered_trace_frozen_dataclass_immutable(
    probe: CoherenceProbe,
    hidden_states: list[np.ndarray],
    logits: np.ndarray,
    token_ids: np.ndarray,
) -> None:
    result = probe.measure_multi_layer(hidden_states, logits, token_ids)
    with pytest.raises(FrozenInstanceError):
        result.n_layers = 999  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        result.layer_residual_cos = []  # type: ignore[misc]


# ---------------------------------------------------------------------------
# validation + edge cases
# ---------------------------------------------------------------------------


def test_measure_multi_layer_rejects_fewer_than_two_layers(
    probe: CoherenceProbe,
    hidden_states: list[np.ndarray],
    logits: np.ndarray,
    token_ids: np.ndarray,
) -> None:
    with pytest.raises(ValueError, match="at least 2 residual layers"):
        probe.measure_multi_layer([hidden_states[0]], logits, token_ids)


def test_measure_multi_layer_rejects_non_list_hidden_states(
    probe: CoherenceProbe,
    hidden_states: list[np.ndarray],
    logits: np.ndarray,
    token_ids: np.ndarray,
) -> None:
    stacked = np.stack(hidden_states, axis=0)
    with pytest.raises(TypeError, match="hidden_states must be a list"):
        probe.measure_multi_layer(stacked, logits, token_ids)  # type: ignore[arg-type]


def test_measure_multi_layer_rejects_mismatched_shapes(
    probe: CoherenceProbe,
    hidden_states: list[np.ndarray],
    logits: np.ndarray,
    token_ids: np.ndarray,
    rng: np.random.Generator,
) -> None:
    bad = list(hidden_states)
    bad[1] = rng.standard_normal((T, D + 1))
    with pytest.raises(ValueError, match="does not match previous layer shape"):
        probe.measure_multi_layer(bad, logits, token_ids)


def test_measure_multi_layer_rejects_non_2d_layer(
    probe: CoherenceProbe,
    hidden_states: list[np.ndarray],
    logits: np.ndarray,
    token_ids: np.ndarray,
    rng: np.random.Generator,
) -> None:
    bad = list(hidden_states)
    bad[0] = rng.standard_normal((T,))
    with pytest.raises(ValueError, match="hidden_states\\[0\\] must be 2-D"):
        probe.measure_multi_layer(bad, logits, token_ids)


def test_env_flag_is_module_level_boolean() -> None:
    # The opt-in gate exists and is a concrete bool (not None, not a string).
    assert isinstance(CARL_LAYER_PROBE_ENABLED, bool)


def test_identical_hidden_states_give_cosine_one(
    probe: CoherenceProbe,
    logits: np.ndarray,
    token_ids: np.ndarray,
    rng: np.random.Generator,
) -> None:
    h = rng.standard_normal((T, D))
    layers = [h.copy(), h.copy(), h.copy()]
    result = probe.measure_multi_layer(layers, logits, token_ids)
    for c in result.layer_residual_cos:
        assert c == pytest.approx(1.0, abs=1e-9)
    assert result.residual_min == pytest.approx(1.0, abs=1e-9)
    assert result.residual_mean == pytest.approx(1.0, abs=1e-9)


def test_antiparallel_hidden_states_give_cosine_negative_one(
    probe: CoherenceProbe,
    logits: np.ndarray,
    token_ids: np.ndarray,
    rng: np.random.Generator,
) -> None:
    h = rng.standard_normal((T, D))
    layers = [h, -h]
    result = probe.measure_multi_layer(layers, logits, token_ids)
    assert result.layer_residual_cos[0] == pytest.approx(-1.0, abs=1e-9)
