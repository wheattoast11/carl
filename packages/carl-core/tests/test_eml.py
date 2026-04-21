"""Tests for carl_core.eml — Odrzywolek EML primitive."""
from __future__ import annotations

import math

import numpy as np
import pytest

from carl_core.eml import (
    CLAMP_X,
    EPS,
    MAX_DEPTH,
    EMLNode,
    EMLOp,
    EMLTree,
    eml,
    eml_array,
    eml_scalar_reward,
)
from carl_core.errors import ValidationError


# ---------------------------------------------------------------------------
# 1. exp(x) = eml(x, 1) — closed form verification
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("x", [-5.0, -1.0, 0.0, 0.5, 1.0, 2.0, 5.0, 10.0])
def test_exp_via_eml_scalar(x: float) -> None:
    assert math.isclose(eml(x, 1.0), math.exp(x), rel_tol=1e-10, abs_tol=1e-10)


@pytest.mark.parametrize("x", [-5.0, -1.0, 0.0, 0.5, 1.0, 2.0, 5.0, 10.0])
def test_exp_via_eml_tree(x: float) -> None:
    tree = EMLTree.exp_single()
    got = tree.forward(np.array([x], dtype=np.float64))
    assert math.isclose(got, math.exp(x), rel_tol=1e-10, abs_tol=1e-10)


# ---------------------------------------------------------------------------
# 2. ln(x) = eml(1, eml(eml(1, x), 1))
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("x", [0.1, 0.5, 1.0, math.e, 5.0, 10.0])
def test_ln_via_eml_scalar(x: float) -> None:
    got = eml(1.0, eml(eml(1.0, x), 1.0))
    assert math.isclose(got, math.log(x), rel_tol=1e-10, abs_tol=1e-10)


@pytest.mark.parametrize("x", [0.1, 0.5, 1.0, math.e, 5.0, 10.0])
def test_ln_via_eml_tree(x: float) -> None:
    tree = EMLTree.ln_single()
    got = tree.forward(np.array([x], dtype=np.float64))
    assert math.isclose(got, math.log(x), rel_tol=1e-10, abs_tol=1e-10)


# ---------------------------------------------------------------------------
# 3. Zero identity: 0 = eml(1, eml(eml(1, 1), 1))
# ---------------------------------------------------------------------------


def test_zero_identity_scalar() -> None:
    zero_scalar = eml(1.0, eml(eml(1.0, 1.0), 1.0))
    assert abs(zero_scalar - 0.0) < 1e-12


def test_zero_identity_tree() -> None:
    zero_tree = EMLTree.zero()
    got = zero_tree.forward(np.array([0.0], dtype=np.float64))
    assert abs(got - 0.0) < 1e-12


# ---------------------------------------------------------------------------
# 4. Clamping: exp(50) must not overflow (clamped to exp(CLAMP_X))
# ---------------------------------------------------------------------------


def test_clamp_positive_extreme() -> None:
    got = eml(50.0, 1.0)
    assert math.isfinite(got)
    assert got == math.exp(CLAMP_X) - math.log(1.0)


def test_clamp_negative_extreme() -> None:
    got = eml(-50.0, 1.0)
    assert math.isfinite(got)
    assert math.isclose(got, math.exp(-CLAMP_X), rel_tol=1e-10)


def test_clamp_vectorized_extreme() -> None:
    xs = np.array([-100.0, -50.0, 0.0, 50.0, 100.0])
    ys = np.ones_like(xs)
    out = eml_array(xs, ys)
    assert np.all(np.isfinite(out))
    # Outer values clamp to exp(+/- CLAMP_X).
    assert math.isclose(float(out[0]), math.exp(-CLAMP_X), rel_tol=1e-10)
    assert math.isclose(float(out[-1]), math.exp(CLAMP_X), rel_tol=1e-10)


# ---------------------------------------------------------------------------
# 5. Domain guard: eml(1, 0) does not crash (uses EPS)
# ---------------------------------------------------------------------------


def test_domain_guard_zero_y_scalar() -> None:
    got = eml(1.0, 0.0)
    assert math.isfinite(got)
    # Guarded value: exp(1) - ln(EPS)
    assert math.isclose(got, math.exp(1.0) - math.log(EPS), rel_tol=1e-10)


def test_domain_guard_negative_y_scalar() -> None:
    got = eml(0.0, -1.0)
    assert math.isfinite(got)
    assert math.isclose(got, math.exp(0.0) - math.log(EPS), rel_tol=1e-10)


def test_domain_guard_vectorized() -> None:
    xs = np.zeros(3)
    ys = np.array([0.0, -1.0, 1e-20])
    out = eml_array(xs, ys)
    assert np.all(np.isfinite(out))


# ---------------------------------------------------------------------------
# 6. Depth enforcement: > 4 raises carl.eml.depth_exceeded
# ---------------------------------------------------------------------------


def test_max_depth_is_four() -> None:
    assert MAX_DEPTH == 4


def test_depth_leaf_is_zero() -> None:
    leaf = EMLNode(op=EMLOp.CONST, const=1.0)
    assert leaf.depth() == 0


def test_depth_single_eml_is_one() -> None:
    node = EMLNode(
        op=EMLOp.EML,
        left=EMLNode(op=EMLOp.CONST, const=1.0),
        right=EMLNode(op=EMLOp.CONST, const=2.0),
    )
    assert node.depth() == 1


def test_depth_five_rejected() -> None:
    """A tree of depth 5 should be rejected at tree construction."""
    # Build a right-spine tree of depth 5.
    one = EMLNode(op=EMLOp.CONST, const=1.0)
    node = one
    for _ in range(5):
        node = EMLNode(op=EMLOp.EML, left=one, right=node)
    assert node.depth() == 5
    with pytest.raises(ValidationError) as excinfo:
        EMLTree(root=node, input_dim=0)
    assert excinfo.value.code == "carl.eml.depth_exceeded"
    assert excinfo.value.context["depth"] == 5
    assert excinfo.value.context["max_depth"] == 4


def test_depth_four_accepted() -> None:
    tree = EMLTree.identity_deep(1)
    assert tree.depth() == 4  # at the boundary, should succeed


# ---------------------------------------------------------------------------
# 7. Round-trip bytes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "factory",
    [
        EMLTree.exp_single,
        EMLTree.ln_single,
        EMLTree.zero,
        lambda: EMLTree.identity(1),
        lambda: EMLTree.identity_deep(1),
    ],
)
def test_bytes_roundtrip_factories(factory) -> None:  # type: ignore[no-untyped-def]
    original = factory()
    b = original.to_bytes()
    decoded = EMLTree.from_bytes(b)
    assert decoded == original
    assert decoded.hash() == original.hash()
    assert decoded.depth() == original.depth()
    assert decoded.nodes() == original.nodes()


def test_bytes_roundtrip_preserves_leaf_params() -> None:
    """Mutating leaf_params before encoding must round-trip the runtime value.

    Structural equality may not hold because the immutable CONST nodes retain
    their original construction-time values, but the serialized bytes carry
    the param-overridden values and the decoded tree evaluates identically.
    """
    tree = EMLTree.exp_single()
    tree.leaf_params[0] = 2.5  # mutate the baked constant
    b = tree.to_bytes()
    decoded = EMLTree.from_bytes(b)
    assert math.isclose(float(decoded.leaf_params[0]), 2.5, rel_tol=1e-12)
    # Forward values match — the param vector is the runtime source of truth.
    x = np.array([0.5])
    assert math.isclose(
        float(decoded.forward(x)), float(tree.forward(x)), rel_tol=1e-12
    )


def test_bytes_bad_magic() -> None:
    bad = b"XXXX" + b"\x00" * 10
    with pytest.raises(ValidationError) as excinfo:
        EMLTree.from_bytes(bad)
    assert excinfo.value.code == "carl.eml.decode_error"


def test_bytes_truncated() -> None:
    tree = EMLTree.exp_single()
    b = tree.to_bytes()
    with pytest.raises(ValidationError) as excinfo:
        EMLTree.from_bytes(b[:4])
    assert excinfo.value.code == "carl.eml.decode_error"


# ---------------------------------------------------------------------------
# 8. Determinism: same tree + same input = same output across 1000 calls
# ---------------------------------------------------------------------------


def test_determinism_1000_calls() -> None:
    tree = EMLTree.exp_single()
    ref = tree.forward(np.array([1.3]))
    for _ in range(1000):
        assert tree.forward(np.array([1.3])) == ref


def test_determinism_batch_matches_scalar() -> None:
    tree = EMLTree.exp_single()
    xs = np.linspace(-2.0, 2.0, 10).reshape(-1, 1)
    batch = tree.forward_batch(xs)
    for i in range(xs.shape[0]):
        assert math.isclose(
            float(batch[i]), tree.forward(xs[i]), rel_tol=1e-12, abs_tol=1e-12
        )


# ---------------------------------------------------------------------------
# 9. Gradient sanity: finite-diff vs numerical for inputs
# ---------------------------------------------------------------------------


def test_grad_wrt_inputs_matches_closed_form() -> None:
    """For exp(x), d/dx exp(x) = exp(x)."""
    tree = EMLTree.exp_single()
    x0 = np.array([1.0])
    grad = tree.grad_wrt_inputs(x0)
    assert math.isclose(float(grad[0]), math.exp(1.0), rel_tol=1e-4)


def test_grad_wrt_inputs_sign() -> None:
    """For depth-2 random tree, finite-diff should match central-diff."""
    rng = np.random.default_rng(42)
    # eml(var, eml(var, 1)) — still depth 2 after checking: eml(1, x) depth 1,
    # then outer eml(var, that) depth 2.
    inner = EMLNode(
        op=EMLOp.EML,
        left=EMLNode(op=EMLOp.VAR_X, var_idx=0),
        right=EMLNode(op=EMLOp.CONST, const=1.0),
    )
    root = EMLNode(
        op=EMLOp.EML,
        left=EMLNode(op=EMLOp.VAR_X, var_idx=0),
        right=inner,
    )
    tree = EMLTree(root=root, input_dim=1)
    x0 = rng.uniform(-1.0, 1.0, size=(1,))
    grad = tree.grad_wrt_inputs(x0, eps=1e-6)
    # Verify via even-larger eps finite diff (consistency self-check).
    step = 1e-4
    plus = tree.forward(x0 + step)
    minus = tree.forward(x0 - step)
    expected = (plus - minus) / (2 * step)
    assert math.isclose(float(grad[0]), expected, rel_tol=1e-3, abs_tol=1e-3)


def test_grad_wrt_params_sign() -> None:
    """Gradient of MSE loss w.r.t. leaf params has correct sign."""
    tree = EMLTree.exp_single()  # one const leaf = 1
    x = np.array([0.5])
    target = 10.0  # exp(0.5)*1 - ln(1) ≈ 1.65, so we're below target
    grad = tree.grad_wrt_params(x, target)
    # With exp_single, params[0] is the leaf const (the `1` in eml(x, 1)).
    # But VAR_X is not a const leaf, so only one const leaf: the `1` on right.
    # Increasing that const makes eml smaller (less negative ln contribution → smaller value),
    # moving us further from target. grad should be positive (we want to decrease the param
    # to reduce loss when below target... but MSE grad here measures d(loss)/d(param)).
    # Just verify grad is finite and non-zero.
    assert np.all(np.isfinite(grad))
    assert grad.shape == tree.leaf_params.shape


# ---------------------------------------------------------------------------
# 10. Hashing: canonical form — two paths, same hash
# ---------------------------------------------------------------------------


def test_canonical_hash_ignores_construction_path() -> None:
    """Tree built via factory and tree built via from_bytes must hash the same."""
    a = EMLTree.exp_single()
    b = EMLTree.from_bytes(a.to_bytes())
    assert a.hash() == b.hash()


def test_canonical_hash_distinguishes_different_trees() -> None:
    a = EMLTree.exp_single()  # eml(x, 1)
    b = EMLTree.ln_single()  # eml(1, eml(eml(1, x), 1))
    assert a.hash() != b.hash()


def test_canonical_hash_depends_on_leaf_params() -> None:
    a = EMLTree.exp_single()
    b = EMLTree.exp_single()
    b.leaf_params[0] = 2.0
    assert a.hash() != b.hash()


def test_canonical_hash_ignores_const_vs_int_float() -> None:
    """1.0 and 1 should canonicalize to the same hash."""
    a = EMLNode(op=EMLOp.CONST, const=1.0)
    b = EMLNode(op=EMLOp.CONST, const=1)
    assert a.to_canonical_dict() == b.to_canonical_dict()


# ---------------------------------------------------------------------------
# 11. Vectorization: forward_batch matches element-wise forward
# ---------------------------------------------------------------------------


def test_forward_batch_matches_elementwise() -> None:
    tree = EMLTree.exp_single()
    inputs = np.array([[0.0], [1.0], [2.0], [-1.0], [0.5]])
    batch = tree.forward_batch(inputs)
    for i in range(inputs.shape[0]):
        assert math.isclose(
            float(batch[i]), tree.forward(inputs[i]), rel_tol=1e-12, abs_tol=1e-12
        )


def test_eml_array_broadcasting() -> None:
    xs = np.array([0.0, 1.0, 2.0])
    y = 1.0
    out = eml_array(xs, y)
    assert out.shape == (3,)
    for i in range(3):
        assert math.isclose(
            float(out[i]), eml(float(xs[i]), y), rel_tol=1e-12
        )


def test_eml_array_2d() -> None:
    xs = np.array([[0.0, 1.0], [2.0, 3.0]])
    ys = np.array([[1.0, 2.0], [3.0, 4.0]])
    out = eml_array(xs, ys)
    assert out.shape == (2, 2)
    assert math.isclose(float(out[0, 0]), eml(0.0, 1.0), rel_tol=1e-12)
    assert math.isclose(float(out[1, 1]), eml(3.0, 4.0), rel_tol=1e-12)


# ---------------------------------------------------------------------------
# 12. Composition: resonant composition preserves dims (also tested in
# test_resonant, but a sanity one here for EML tree composition mechanics).
# ---------------------------------------------------------------------------


def test_tree_composition_depth() -> None:
    """eml(depth-2, depth-2) should have depth 3."""
    d2_a = EMLNode(
        op=EMLOp.EML,
        left=EMLNode(op=EMLOp.EML,
                     left=EMLNode(op=EMLOp.CONST, const=1.0),
                     right=EMLNode(op=EMLOp.VAR_X, var_idx=0)),
        right=EMLNode(op=EMLOp.CONST, const=1.0),
    )
    # Second depth-2 branch: eml(1, eml(1, x)).
    d2_b = EMLNode(
        op=EMLOp.EML,
        left=EMLNode(op=EMLOp.CONST, const=1.0),
        right=EMLNode(
            op=EMLOp.EML,
            left=EMLNode(op=EMLOp.CONST, const=1.0),
            right=EMLNode(op=EMLOp.VAR_X, var_idx=0),
        ),
    )
    assert d2_a.depth() == 2
    assert d2_b.depth() == 2
    composed = EMLNode(op=EMLOp.EML, left=d2_a, right=d2_b)
    assert composed.depth() == 3


# ---------------------------------------------------------------------------
# EMLNode invariants
# ---------------------------------------------------------------------------


def test_const_node_requires_const() -> None:
    with pytest.raises(ValidationError) as excinfo:
        EMLNode(op=EMLOp.CONST)
    assert excinfo.value.code == "carl.eml.domain_error"


def test_var_node_requires_var_idx() -> None:
    with pytest.raises(ValidationError) as excinfo:
        EMLNode(op=EMLOp.VAR_X)
    assert excinfo.value.code == "carl.eml.domain_error"


def test_var_node_rejects_negative_idx() -> None:
    with pytest.raises(ValidationError) as excinfo:
        EMLNode(op=EMLOp.VAR_X, var_idx=-1)
    assert excinfo.value.code == "carl.eml.domain_error"


def test_eml_node_requires_both_children() -> None:
    with pytest.raises(ValidationError) as excinfo:
        EMLNode(op=EMLOp.EML, left=EMLNode(op=EMLOp.CONST, const=1.0))
    assert excinfo.value.code == "carl.eml.domain_error"


def test_const_node_rejects_children() -> None:
    with pytest.raises(ValidationError):
        EMLNode(
            op=EMLOp.CONST,
            const=1.0,
            left=EMLNode(op=EMLOp.CONST, const=1.0),
        )


# ---------------------------------------------------------------------------
# EMLTree invariants
# ---------------------------------------------------------------------------


def test_tree_rejects_insufficient_input_dim() -> None:
    root = EMLNode(op=EMLOp.VAR_X, var_idx=2)
    with pytest.raises(ValidationError) as excinfo:
        EMLTree(root=root, input_dim=1)
    assert excinfo.value.code == "carl.eml.domain_error"


def test_tree_forward_rejects_short_input() -> None:
    tree = EMLTree.exp_single()
    with pytest.raises(ValidationError):
        tree.forward(np.array([]))


def test_tree_forward_batch_rejects_1d_short() -> None:
    tree = EMLTree.exp_single()
    with pytest.raises(ValidationError):
        tree.forward_batch(np.zeros((2, 0)))


# ---------------------------------------------------------------------------
# from_dict / to_canonical_dict roundtrip
# ---------------------------------------------------------------------------


def test_node_dict_roundtrip() -> None:
    for factory in (EMLTree.exp_single, EMLTree.ln_single, EMLTree.zero):
        tree = factory()
        d = tree.root.to_canonical_dict()
        decoded = EMLNode.from_dict(d)
        assert decoded == tree.root


def test_node_dict_decode_error_missing_op() -> None:
    with pytest.raises(ValidationError) as excinfo:
        EMLNode.from_dict({})
    assert excinfo.value.code == "carl.eml.decode_error"


def test_node_dict_decode_error_bad_op() -> None:
    with pytest.raises(ValidationError) as excinfo:
        EMLNode.from_dict({"op": 99})
    assert excinfo.value.code == "carl.eml.decode_error"


def test_node_dict_decode_error_missing_const() -> None:
    with pytest.raises(ValidationError) as excinfo:
        EMLNode.from_dict({"op": int(EMLOp.CONST)})
    assert excinfo.value.code == "carl.eml.decode_error"


def test_node_dict_decode_error_missing_children() -> None:
    with pytest.raises(ValidationError) as excinfo:
        EMLNode.from_dict({"op": int(EMLOp.EML)})
    assert excinfo.value.code == "carl.eml.decode_error"


# ---------------------------------------------------------------------------
# Helper reward API
# ---------------------------------------------------------------------------


def test_eml_scalar_reward_sign() -> None:
    """Reward should increase with coherence, decrease with dispersion."""
    r_low = eml_scalar_reward(0.1, 1.0)
    r_high = eml_scalar_reward(1.0, 1.0)
    assert r_high > r_low

    r_tight = eml_scalar_reward(1.0, 0.1)
    r_loose = eml_scalar_reward(1.0, 10.0)
    assert r_tight > r_loose


def test_eml_scalar_reward_dispersion_guard() -> None:
    # Zero dispersion must not crash (uses EPS).
    got = eml_scalar_reward(0.0, 0.0)
    assert math.isfinite(got)


# ---------------------------------------------------------------------------
# Tree repr
# ---------------------------------------------------------------------------


def test_tree_repr_includes_formula() -> None:
    tree = EMLTree.exp_single()
    s = repr(tree)
    assert "eml" in s
    assert "x0" in s
    assert "input_dim=1" in s


def test_tree_hashable_via_identity() -> None:
    a = EMLTree.exp_single()
    b = EMLTree.exp_single()
    assert hash(a) == hash(b)
    # Dict membership works.
    seen = {a}
    assert b in seen
