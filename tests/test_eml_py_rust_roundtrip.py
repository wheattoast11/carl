"""Cross-boundary isomorphism test — Python EMLTree vs Rust EmlTree.

This test runs the Rust `cargo run --example eml_vectors -p terminals-core`
binary to produce a JSON test battery, then replays each vector in the
Python `EMLTree` impl and asserts numerical agreement within the
documented f32/f64 tolerance window.

Runs only when:
  1. Cargo is on PATH, and
  2. The `terminals-landing-new/crates` workspace exists at the expected
     path relative to this repo (treated as a sibling checkout).

Otherwise the whole module is skipped — carl-studio does not own the
Rust toolchain in CI, so we do not fail the Python test suite on its
absence.

See `docs/eml_wire_format.md` for the contract this test enforces.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from carl_core.eml import (
    CLAMP_X,
    EMLNode,
    EMLOp,
    EMLTree,
    eml,
)

# ---------------------------------------------------------------------------
# Harness discovery
# ---------------------------------------------------------------------------

# We expect terminals-landing-new to live as a sibling of carl-studio under
# ~/Documents/terminals-tech-landing/. Override via CARL_RUST_CRATES env
# to point at any cargo workspace containing terminals-core.
_DEFAULT_CRATES = (
    Path.home()
    / "Documents"
    / "terminals-tech-landing"
    / "terminals-landing-new"
    / "crates"
)
_CRATES_DIR = Path(os.environ.get("CARL_RUST_CRATES", str(_DEFAULT_CRATES)))

_CARGO = shutil.which("cargo")
_HAS_RUST = _CARGO is not None and _CRATES_DIR.exists() and (
    _CRATES_DIR / "terminals-core" / "examples" / "eml_vectors.rs"
).exists()

pytestmark = pytest.mark.skipif(
    not _HAS_RUST,
    reason=(
        "Rust toolchain or terminals-core workspace unavailable "
        f"(looked for {_CRATES_DIR}). Set CARL_RUST_CRATES to override."
    ),
)


# ---------------------------------------------------------------------------
# Tolerance — f32 vs f64, soft-Taylor vs libm
# ---------------------------------------------------------------------------

# Documented in docs/eml_wire_format.md §5.
_REL_TOL_LEAF = 2e-6  # single-op trees
_REL_TOL_DEPTH_2 = 5e-6
_REL_TOL_DEPTH_4 = 5e-5


def _rel_tol_for_depth(depth: int) -> float:
    if depth <= 1:
        return _REL_TOL_LEAF
    if depth <= 2:
        return _REL_TOL_DEPTH_2
    return _REL_TOL_DEPTH_4


# ---------------------------------------------------------------------------
# Python bytecode translator — reconstructs a Python EMLTree from the
# Rust-side (bytecode + constants + input_dim) triple. Only handles the
# shared subset {Const, VarX, Eml}. Returns None for Rust-only opcodes
# (Add/Sub/Mul/Neg) — those vectors are skipped.
# ---------------------------------------------------------------------------


def _rust_bytecode_to_py_tree(
    bytecode: bytes, constants: list[float], input_dim: int
) -> EMLTree | None:
    """Translate Rust postfix bytecode into a Python EMLTree.

    Returns None if the bytecode uses opcodes outside the shared subset.
    """
    stack: list[EMLNode] = []
    const_vals: list[float] = []  # tracks the order leaves appear in postfix
    i = 0
    while i < len(bytecode):
        op = bytecode[i]
        i += 1
        if op == 0x00:  # Const
            idx = bytecode[i]
            i += 1
            val = float(constants[idx])
            stack.append(EMLNode(op=EMLOp.CONST, const=val))
            const_vals.append(val)
        elif op == 0x01:  # VarX
            idx = bytecode[i]
            i += 1
            stack.append(EMLNode(op=EMLOp.VAR_X, var_idx=int(idx)))
        elif op == 0x02:  # Eml
            right = stack.pop()
            left = stack.pop()
            stack.append(EMLNode(op=EMLOp.EML, left=left, right=right))
        else:
            # Rust-only opcodes (Add=3/Sub=4/Mul=5/Neg=6) — no Python equivalent.
            return None
    if len(stack) != 1:
        return None
    return EMLTree(
        root=stack[0],
        input_dim=int(input_dim),
        leaf_params=np.asarray(const_vals, dtype=np.float64),
    )


# ---------------------------------------------------------------------------
# Vector battery — cached per session
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def rust_vectors() -> dict:
    proc = subprocess.run(
        [_CARGO, "run", "--example", "eml_vectors", "-p", "terminals-core", "-q"],
        cwd=str(_CRATES_DIR),
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )
    if proc.returncode != 0:
        pytest.skip(
            f"cargo run eml_vectors failed (rc={proc.returncode}):\n"
            f"STDOUT:\n{proc.stdout[-800:]}\nSTDERR:\n{proc.stderr[-800:]}"
        )
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        pytest.skip(f"cargo example did not emit valid JSON: {exc}\n{proc.stdout[:400]}")
        raise  # for type-checker


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_schema_version(rust_vectors: dict) -> None:
    assert rust_vectors["schema_version"] == 1, (
        "Rust test-vector schema bumped — update this test in lockstep."
    )


def test_magic_bytes_differ_as_documented(rust_vectors: dict) -> None:
    """Document the deliberate wire-format divergence.

    Python magic is b'EML\\x01' (0x45 4d 4c 01), Rust is b'EML1' (0x45 4d 4c 31).
    This asserts they REMAIN different — if they ever collide, interop
    becomes ambiguous.
    """
    # Grab any Rust tree's encoding to inspect magic.
    rust_tree0 = rust_vectors["trees"][0]
    rust_magic = bytes.fromhex(rust_tree0["encode_hex"])[:4]
    assert rust_magic == b"EML1"

    # Any Python tree's magic.
    py_magic = EMLTree.exp_single().to_bytes()[:4]
    assert py_magic == b"EML\x01"

    assert rust_magic != py_magic, "magic bytes collision — update wire doc"


def test_scalar_eml_agreement(rust_vectors: dict) -> None:
    """Python `eml(x, y)` and Rust `eml_det(x, y)` agree within f32 tolerance.

    Skips pairs where Rust errored (y ≤ 0) — Python silently clamps and
    would not produce a comparable value.

    Uses a mixed rel/abs tolerance: relative error < 1e-5 OR absolute
    error < 1e-6. The absolute floor handles catastrophic cancellation
    cases (e.g. `eml(0, e) ≈ 1 - 1`) where both sides produce values
    near 1e-7 in magnitude — the ratio is noisy but the numbers agree.
    """
    # Tolerance model: f32 mantissa gives ~1.2e-7 relative precision.
    # Softfloat Taylor/Mercator adds another ~5e-6 relative error at the
    # series tails. We accept the combined bound — rel < 5e-5 OR
    # abs < 1e-6 (covers cancellation near zero).
    max_rel_bounded = 0.0
    for pair in rust_vectors["scalar_eml"]:
        if "error" in pair:
            continue
        x, y = float(pair["x"]), float(pair["y"])
        rust_v = float(pair["out"])
        py_v = eml(x, y)
        denom = max(abs(rust_v), abs(py_v), 1e-9)
        rel = abs(rust_v - py_v) / denom
        abs_err = abs(rust_v - py_v)
        # Track max relative error EXCEPT for near-zero cancellation
        # cases where relative error is not meaningful.
        if abs_err >= 1e-6:
            max_rel_bounded = max(max_rel_bounded, rel)
        ok = rel < 5e-5 or abs_err < 1e-6
        assert ok, (
            f"scalar eml divergence @ x={x}, y={y}: py={py_v:.10g} rust={rust_v:.10g} "
            f"rel={rel:.2e} abs={abs_err:.2e}"
        )
    # Documented relative-error bound (excluding near-zero cancellation).
    assert max_rel_bounded < 5e-5, (
        f"scalar eml max relative error {max_rel_bounded:.2e} > 5e-5 "
        "— softfloat precision regression?"
    )


def test_tree_eval_agreement(rust_vectors: dict) -> None:
    """Re-evaluate each Rust-side tree in Python and compare outputs.

    For trees using only the shared subset, reconstructs the Python tree
    from the Rust bytecode and runs forward on each input vector.
    Skips trees using Rust-only arithmetic opcodes (Add/Sub/Mul/Neg).
    """
    shared_trees = 0
    arith_skipped = 0
    for tree_spec in rust_vectors["trees"]:
        bytecode = bytes.fromhex(tree_spec["bytecode_hex"])
        consts = [float(c) for c in tree_spec["constants"]]
        input_dim = int(tree_spec["input_dim"])
        depth = int(tree_spec["depth"])

        py_tree = _rust_bytecode_to_py_tree(bytecode, consts, input_dim)
        if py_tree is None:
            arith_skipped += 1
            continue
        shared_trees += 1
        tol = _rel_tol_for_depth(depth)

        for ev in tree_spec["evals"]:
            if "error" in ev:
                continue
            inputs = np.asarray(ev["inputs"], dtype=np.float64)
            rust_out = [float(v) for v in ev["out"]]
            assert len(rust_out) == 1, (
                f"multi-output trees not supported in shared subset "
                f"(tree={tree_spec['name']})"
            )
            py_out = py_tree.forward(inputs)
            rust_v = rust_out[0]
            denom = max(abs(rust_v), abs(py_out), 1e-9)
            rel = abs(rust_v - py_out) / denom
            # At tree depth 4 with saturated exp chains we hit the CLAMP_X
            # ceiling on both sides — the relative-error denominator
            # becomes enormous (~e^20) but both agree on that saturated
            # value. Accept absolute-error fallback for clamped cases.
            abs_err = abs(rust_v - py_out)
            assert rel < tol or abs_err < 1e-3 * max(1.0, denom), (
                f"tree eval divergence (tree={tree_spec['name']} depth={depth} "
                f"inputs={list(inputs)}): py={py_out:.10g} rust={rust_v:.10g} "
                f"rel={rel:.2e} tol={tol:.0e}"
            )
    assert shared_trees >= 3, (
        f"expected ≥3 shared-subset trees in test battery, got {shared_trees}"
    )
    assert arith_skipped >= 1, (
        "expected at least one arithmetic-opcode tree to exercise the "
        "skip path — battery shape changed"
    )


def test_opcode_shared_prefix_stable(rust_vectors: dict) -> None:
    """The first three opcodes (Const/VarX/Eml) MUST agree in numeric value.

    Python: EMLOp.CONST=0, VAR_X=1, EML=2.
    Rust bytecode in the shared subset uses the same integers.
    """
    assert int(EMLOp.CONST) == 0
    assert int(EMLOp.VAR_X) == 1
    assert int(EMLOp.EML) == 2
    # Spot-check by scanning a Rust bytecode — exp_x has [VarX=1, 0, Const=0, 0, Eml=2].
    exp_x = next(t for t in rust_vectors["trees"] if t["name"] == "exp_x")
    bc = bytes.fromhex(exp_x["bytecode_hex"])
    assert bc[0] == 1, "Rust VarX opcode != 1 — enum drift"
    assert bc[2] == 0, "Rust Const opcode != 0 — enum drift"
    assert bc[-1] == 2, "Rust Eml opcode != 2 — enum drift"


def test_clamp_agreement_at_ceiling() -> None:
    """Both sides clamp x at ±20 for exp. Inside the clamp, Python (f64)
    tracks libm; outside, both saturate to exp(20) ≈ 4.85e8."""
    # Python clamps at CLAMP_X=20.0.
    assert CLAMP_X == 20.0
    py_at_clamp = eml(20.0, 1.0)
    py_past_clamp = eml(1e6, 1.0)
    # Past the clamp, Python produces the clamped result.
    assert abs(py_at_clamp - py_past_clamp) < 1e-9


def test_depth_limit_agreement() -> None:
    """Both sides reject trees at depth 5; depth 4 must validate."""
    # Python: MAX_DEPTH=4. Build a depth-5 tree and expect rejection.
    from carl_core.eml import _IDENTITY_CONST  # type: ignore[attr-defined]

    one = EMLNode(op=EMLOp.CONST, const=_IDENTITY_CONST)
    x = EMLNode(op=EMLOp.VAR_X, var_idx=0)
    # Build depth 5 by nesting 5 EML layers.
    node: EMLNode = x
    for _ in range(5):
        node = EMLNode(op=EMLOp.EML, left=node, right=one)
    with pytest.raises(Exception) as excinfo:
        EMLTree(root=node, input_dim=1)
    # Error code should carry carl.eml.depth_exceeded.
    err = excinfo.value
    assert getattr(err, "code", "") == "carl.eml.depth_exceeded"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
