"""Tests for carl_core.resonant — perceive/cognize/act triple built on EML."""
from __future__ import annotations

import math

import numpy as np
import pytest

from carl_core.eml import MAX_DEPTH, EMLTree
from carl_core.errors import ValidationError
from carl_core.resonant import (
    Resonant,
    compose_resonants,
    make_resonant,
)


# ---------------------------------------------------------------------------
# Factory / basic construction
# ---------------------------------------------------------------------------


def test_make_resonant_happy_path() -> None:
    tree = EMLTree.exp_single()
    proj = np.eye(2, 3)
    readout = np.eye(4, 2)
    r = make_resonant(tree, proj, readout)
    assert r.observation_dim == 3
    assert r.latent_dim == 2
    assert r.action_dim == 4
    assert isinstance(r.identity, str)
    assert len(r.identity) == 64  # sha256 hex


def test_make_resonant_rejects_mismatched_matrices() -> None:
    tree = EMLTree.exp_single()
    proj = np.eye(2, 3)  # k=2
    readout = np.eye(4, 5)  # expects k=5
    with pytest.raises(ValidationError) as excinfo:
        make_resonant(tree, proj, readout)
    assert excinfo.value.code == "carl.eml.domain_error"


def test_make_resonant_rejects_wrong_ndim() -> None:
    tree = EMLTree.exp_single()
    proj = np.array([1.0, 2.0])  # 1D
    readout = np.eye(2, 2)
    with pytest.raises(ValidationError):
        make_resonant(tree, proj, readout)


# ---------------------------------------------------------------------------
# Resonant-specific test 1: Round-trip to_dict / from_dict
# ---------------------------------------------------------------------------


def test_roundtrip_to_dict_from_dict() -> None:
    rng = np.random.default_rng(0)
    tree = EMLTree.exp_single()
    proj = rng.normal(size=(3, 4))
    readout = rng.normal(size=(2, 3))
    metadata = {"name": "test", "version": 1}
    r = make_resonant(tree, proj, readout, metadata=metadata)
    d = r.to_dict()
    r2 = Resonant.from_dict(d)
    assert r2 == r
    assert r2.identity == r.identity
    assert r2.metadata == r.metadata
    np.testing.assert_allclose(r2.projection, r.projection, atol=1e-12)
    np.testing.assert_allclose(r2.readout, r.readout, atol=1e-12)


def test_from_dict_detects_identity_tamper() -> None:
    tree = EMLTree.exp_single()
    proj = np.eye(2, 3)
    readout = np.eye(2, 2)
    r = make_resonant(tree, proj, readout)
    d = r.to_dict()
    d["identity"] = "00" * 32  # tamper
    with pytest.raises(ValidationError) as excinfo:
        Resonant.from_dict(d)
    assert excinfo.value.code == "carl.eml.decode_error"


def test_from_dict_missing_keys() -> None:
    with pytest.raises(ValidationError) as excinfo:
        Resonant.from_dict({"tree": "00"})
    assert excinfo.value.code == "carl.eml.decode_error"


# ---------------------------------------------------------------------------
# Resonant-specific test 2: Identity sha256 is stable across builds
# ---------------------------------------------------------------------------


def test_identity_stable_for_equivalent_inputs() -> None:
    tree_a = EMLTree.exp_single()
    tree_b = EMLTree.exp_single()
    proj_a = np.eye(2, 3)
    proj_b = np.eye(2, 3)
    readout_a = np.eye(2, 2)
    readout_b = np.eye(2, 2)
    r_a = make_resonant(tree_a, proj_a, readout_a)
    r_b = make_resonant(tree_b, proj_b, readout_b)
    assert r_a.identity == r_b.identity


def test_identity_changes_with_matrix_values() -> None:
    tree = EMLTree.exp_single()
    proj_a = np.eye(2, 3)
    proj_b = np.eye(2, 3) * 2.0
    readout = np.eye(2, 2)
    r_a = make_resonant(tree, proj_a, readout)
    r_b = make_resonant(tree, proj_b, readout)
    assert r_a.identity != r_b.identity


def test_identity_changes_with_tree() -> None:
    tree_a = EMLTree.exp_single()
    tree_b = EMLTree.ln_single()
    proj = np.eye(2, 3)
    readout = np.eye(2, 2)
    r_a = make_resonant(tree_a, proj, readout)
    r_b = make_resonant(tree_b, proj, readout)
    assert r_a.identity != r_b.identity


def test_identity_ignores_metadata() -> None:
    tree = EMLTree.exp_single()
    proj = np.eye(2, 3)
    readout = np.eye(2, 2)
    r_a = make_resonant(tree, proj, readout, metadata={"a": 1})
    r_b = make_resonant(tree, proj, readout, metadata={"b": 2})
    assert r_a.identity == r_b.identity


# ---------------------------------------------------------------------------
# Resonant-specific test 3: Forward pipeline shapes
# ---------------------------------------------------------------------------


def test_perceive_1d() -> None:
    tree = EMLTree.exp_single()
    proj = np.eye(2, 3)
    readout = np.eye(2, 2)
    r = make_resonant(tree, proj, readout)
    obs = np.array([1.0, 2.0, 3.0])
    latent = r.perceive(obs)
    assert latent.shape == (2,)


def test_perceive_batch() -> None:
    tree = EMLTree.exp_single()
    proj = np.eye(2, 3)
    readout = np.eye(2, 2)
    r = make_resonant(tree, proj, readout)
    obs = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    latent = r.perceive(obs)
    assert latent.shape == (2, 2)


def test_cognize_applies_tree_per_dim() -> None:
    tree = EMLTree.exp_single()
    proj = np.eye(2, 3)
    readout = np.eye(2, 2)
    r = make_resonant(tree, proj, readout)
    lat = np.array([0.0, 1.0])
    cog = r.cognize(lat)
    # exp(0) = 1, exp(1) = e
    assert math.isclose(float(cog[0]), math.exp(0.0), rel_tol=1e-10)
    assert math.isclose(float(cog[1]), math.exp(1.0), rel_tol=1e-10)


def test_act_shape() -> None:
    tree = EMLTree.exp_single()
    proj = np.eye(2, 3)
    readout = np.eye(4, 2)
    r = make_resonant(tree, proj, readout)
    lat = np.array([1.0, 2.0])
    act = r.act(lat)
    assert act.shape == (4,)


def test_forward_full_pipeline() -> None:
    tree = EMLTree.exp_single()
    proj = np.eye(2, 3)
    readout = np.eye(2, 2)
    r = make_resonant(tree, proj, readout)
    obs = np.array([0.0, 1.0, 0.0])
    out = r.forward(obs)
    assert out.shape == (2,)
    # proj selects obs[0]=0 and obs[1]=1 → cognize(exp) → readout identity → [1, e].
    assert math.isclose(float(out[0]), math.exp(0.0), rel_tol=1e-10)
    assert math.isclose(float(out[1]), math.exp(1.0), rel_tol=1e-10)


def test_perceive_dim_mismatch_raises() -> None:
    tree = EMLTree.exp_single()
    proj = np.eye(2, 3)
    readout = np.eye(2, 2)
    r = make_resonant(tree, proj, readout)
    with pytest.raises(ValidationError):
        r.perceive(np.array([1.0, 2.0]))  # wrong dim


def test_cognize_dim_mismatch_raises() -> None:
    tree = EMLTree.exp_single()
    proj = np.eye(2, 3)
    readout = np.eye(2, 2)
    r = make_resonant(tree, proj, readout)
    with pytest.raises(ValidationError):
        r.cognize(np.array([1.0, 2.0, 3.0]))  # wrong dim


def test_act_dim_mismatch_raises() -> None:
    tree = EMLTree.exp_single()
    proj = np.eye(2, 3)
    readout = np.eye(2, 2)
    r = make_resonant(tree, proj, readout)
    with pytest.raises(ValidationError):
        r.act(np.array([1.0, 2.0, 3.0]))


# ---------------------------------------------------------------------------
# Resonant-specific test 4: Composition shapes align
# ---------------------------------------------------------------------------


def _make(tree: EMLTree, obs_dim: int, latent_dim: int, action_dim: int) -> Resonant:
    rng = np.random.default_rng(1)
    proj = rng.normal(size=(latent_dim, obs_dim))
    readout = rng.normal(size=(action_dim, latent_dim))
    return make_resonant(tree, proj, readout)


def test_compose_matching_shapes() -> None:
    r1 = _make(EMLTree.exp_single(), 3, 2, 4)
    r2 = _make(EMLTree.identity(1), 3, 2, 4)
    composed = compose_resonants(r1, r2)
    assert composed.observation_dim == 3
    assert composed.latent_dim == 2
    assert composed.action_dim == 4
    # Composed tree is eml(exp_single, identity) = depth 1 + max(1, 0) = 2.
    assert composed.tree.depth() == 2


def test_compose_mismatched_latent_rejected() -> None:
    r1 = _make(EMLTree.exp_single(), 3, 2, 4)
    r2 = _make(EMLTree.identity(1), 3, 5, 4)
    with pytest.raises(ValidationError) as excinfo:
        compose_resonants(r1, r2)
    assert excinfo.value.code == "carl.eml.domain_error"


def test_compose_mismatched_observation_rejected() -> None:
    r1 = _make(EMLTree.exp_single(), 3, 2, 4)
    r2 = _make(EMLTree.identity(1), 7, 2, 4)
    with pytest.raises(ValidationError) as excinfo:
        compose_resonants(r1, r2)
    assert excinfo.value.code == "carl.eml.domain_error"


def test_compose_mismatched_action_rejected() -> None:
    r1 = _make(EMLTree.exp_single(), 3, 2, 4)
    r2 = _make(EMLTree.identity(1), 3, 2, 7)
    with pytest.raises(ValidationError) as excinfo:
        compose_resonants(r1, r2)
    assert excinfo.value.code == "carl.eml.domain_error"


def test_compose_depth_exceeded() -> None:
    r1 = _make(EMLTree.identity_deep(1), 3, 2, 4)  # depth 4
    r2 = _make(EMLTree.identity_deep(1), 3, 2, 4)  # depth 4
    # eml(depth4, depth4) => depth 5 > MAX_DEPTH
    with pytest.raises(ValidationError) as excinfo:
        compose_resonants(r1, r2)
    assert excinfo.value.code == "carl.eml.depth_exceeded"


def test_compose_metadata_records_provenance() -> None:
    r1 = _make(EMLTree.exp_single(), 3, 2, 4)
    r2 = _make(EMLTree.identity(1), 3, 2, 4)
    composed = compose_resonants(r1, r2)
    assert "composed_of" in composed.metadata
    assert composed.metadata["composed_of"] == [r1.identity, r2.identity]
    assert composed.metadata["composition_depth"] == composed.tree.depth()


def test_compose_respects_max_depth_constant() -> None:
    assert MAX_DEPTH == 4  # sanity — if this changes, adjust compose tests above


# ---------------------------------------------------------------------------
# Resonant equality & hashability
# ---------------------------------------------------------------------------


def test_resonant_equality_by_identity() -> None:
    r_a = _make(EMLTree.exp_single(), 3, 2, 4)
    r_b = _make(EMLTree.exp_single(), 3, 2, 4)
    assert r_a.identity == r_b.identity
    # Equality uses identity + metadata.
    assert r_a == r_b


def test_resonant_set_membership() -> None:
    r_a = _make(EMLTree.exp_single(), 3, 2, 4)
    r_b = _make(EMLTree.exp_single(), 3, 2, 4)
    seen = {r_a}
    assert r_b in seen


def test_resonant_inequality_with_non_resonant() -> None:
    r = _make(EMLTree.exp_single(), 3, 2, 4)
    assert (r == "not a resonant") is False


# ---------------------------------------------------------------------------
# Numerical: forward with non-identity matrices
# ---------------------------------------------------------------------------


def test_forward_batch_consistency_with_per_row() -> None:
    rng = np.random.default_rng(7)
    tree = EMLTree.exp_single()
    proj = rng.normal(size=(2, 3))
    readout = rng.normal(size=(2, 2))
    r = make_resonant(tree, proj, readout)

    obs_batch = rng.normal(size=(5, 3))
    batch_out = r.forward(obs_batch)
    for i in range(5):
        row_out = r.forward(obs_batch[i])
        np.testing.assert_allclose(batch_out[i], row_out, atol=1e-10)


def test_dict_serialization_tree_bytes_present() -> None:
    r = _make(EMLTree.exp_single(), 3, 2, 4)
    d = r.to_dict()
    assert "tree" in d
    assert isinstance(d["tree"], str)  # hex encoding
    # Valid hex.
    bytes.fromhex(d["tree"])
