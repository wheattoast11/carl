#!/usr/bin/env python3
"""Generate shared EML test vectors from the Python reference implementation.

Writes ``test/vectors.json`` with 10 fixed tuples that both the TS codec and
the Python reference decode and verify against. Any impl that passes these is
protocol-compatible.

Usage:
    python3 scripts/gen_vectors.py
    # or via the npm script: `npm run gen-vectors`

Requires:
    - carl-core installed (editable or on PYTHONPATH).
    - numpy

Deterministic: every value is hand-picked; no RNG.
"""
from __future__ import annotations

import json
import os
import struct
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Bootstrap carl_core without requiring a full editable install.
_HERE = Path(__file__).resolve().parent
_PKG_ROOT = _HERE.parent.parent.parent  # repo root
_CARL_CORE_SRC = _PKG_ROOT / "packages" / "carl-core" / "src"
if str(_CARL_CORE_SRC) not in sys.path:
    sys.path.insert(0, str(_CARL_CORE_SRC))

from carl_core.eml import EMLNode, EMLOp, EMLTree  # noqa: E402
from carl_core.signing import (  # noqa: E402
    sign_platform_countersig,
    sign_tree_software,
)

USER_SECRET = b"x" * 16  # 16 bytes — minimum allowed
PLATFORM_SECRET = b"p" * 16

OUT = _HERE.parent / "test" / "vectors.json"


def _forward(tree: EMLTree, inputs: list[float]) -> float:
    """Deterministic scalar eval matching the TS reference evaluator."""
    arr = np.asarray(inputs, dtype=np.float64)
    # Pad if caller passed a shorter input vector.
    if arr.shape[0] < tree.input_dim:
        pad = np.zeros(tree.input_dim - arr.shape[0], dtype=np.float64)
        arr = np.concatenate([arr, pad])
    return float(tree.forward(arr))


def build_identity_var() -> EMLTree:
    """Depth-0 VAR_X leaf. input_dim=1."""
    return EMLTree.identity(1)


def build_const() -> EMLTree:
    """Depth-0 CONST leaf. input_dim=1 (no vars)."""
    root = EMLNode(op=EMLOp.CONST, const=2.718281828459045)
    return EMLTree(root=root, input_dim=1)


def build_simple_eml() -> EMLTree:
    """Depth-1 eml(x, 1) = exp_single. input_dim=1."""
    return EMLTree.exp_single()


def build_depth2() -> EMLTree:
    """Depth-2 eml(eml(x0, 1), x1). input_dim=2."""
    inner = EMLNode(
        op=EMLOp.EML,
        left=EMLNode(op=EMLOp.VAR_X, var_idx=0),
        right=EMLNode(op=EMLOp.CONST, const=1.0),
    )
    root = EMLNode(
        op=EMLOp.EML,
        left=inner,
        right=EMLNode(op=EMLOp.VAR_X, var_idx=1),
    )
    return EMLTree(root=root, input_dim=2)


def build_depth4() -> EMLTree:
    """Depth-4 exp(ln(x)) identity_deep. input_dim=1."""
    return EMLTree.identity_deep(1)


def build_zero() -> EMLTree:
    """Canonical zero expression, depth=3."""
    return EMLTree.zero()


def build_ln_single() -> EMLTree:
    """ln(x) depth-3 tree."""
    return EMLTree.ln_single()


def build_mixed_consts() -> EMLTree:
    """Depth-2 eml(CONST=0.5, eml(CONST=-1.25, x0)). input_dim=1."""
    inner = EMLNode(
        op=EMLOp.EML,
        left=EMLNode(op=EMLOp.CONST, const=-1.25),
        right=EMLNode(op=EMLOp.VAR_X, var_idx=0),
    )
    root = EMLNode(
        op=EMLOp.EML,
        left=EMLNode(op=EMLOp.CONST, const=0.5),
        right=inner,
    )
    return EMLTree(root=root, input_dim=1)


def build_multi_var() -> EMLTree:
    """eml(x0, x2) with input_dim=3 to exercise varIdx > 0."""
    root = EMLNode(
        op=EMLOp.EML,
        left=EMLNode(op=EMLOp.VAR_X, var_idx=0),
        right=EMLNode(op=EMLOp.VAR_X, var_idx=2),
    )
    return EMLTree(root=root, input_dim=3)


def build_neg_const() -> EMLTree:
    """Depth-1 eml(CONST=-0.75, x0) — stress float64 sign handling."""
    root = EMLNode(
        op=EMLOp.EML,
        left=EMLNode(op=EMLOp.CONST, const=-0.75),
        right=EMLNode(op=EMLOp.VAR_X, var_idx=0),
    )
    return EMLTree(root=root, input_dim=1)


CASES: list[tuple[str, EMLTree, list[float]]] = [
    ("identity_var", build_identity_var(), [1.5]),
    ("const_e", build_const(), [0.0]),
    ("simple_eml_exp", build_simple_eml(), [0.5]),
    ("depth2_eml", build_depth2(), [0.25, 0.75]),
    ("depth4_identity_deep", build_depth4(), [1.4]),
    ("zero_tree", build_zero(), [0.0]),
    ("ln_single", build_ln_single(), [2.0]),
    ("mixed_consts", build_mixed_consts(), [0.3]),
    ("multi_var", build_multi_var(), [0.1, 99.0, 0.8]),
    ("neg_const", build_neg_const(), [1.1]),
]


def encode_envelope(inner: bytes, sig: bytes) -> bytes:
    """Match terminals_runtime.eml.codec_impl.encode with include_signature=True."""
    return b"EMLT" + bytes([0x01]) + inner + sig


def main() -> None:
    records: list[dict[str, Any]] = []
    for name, tree, inputs in CASES:
        inner = tree.to_bytes()
        sig = sign_tree_software(inner, USER_SECRET)
        envelope_signed = encode_envelope(inner, sig)
        envelope_unsigned = b"EMLT" + bytes([0x01]) + inner
        out = _forward(tree, inputs)
        records.append(
            {
                "name": name,
                "depth": tree.depth(),
                "input_dim": tree.input_dim,
                "inner_bytes_hex": inner.hex(),
                "envelope_unsigned_hex": envelope_unsigned.hex(),
                "envelope_signed_hex": envelope_signed.hex(),
                "sig_hex": sig.hex(),
                "inputs": list(inputs),
                "expected_output": out,
            }
        )

    # Platform countersig vector — independent of tree payload.
    countersig_cases = []
    cs_payload = {
        "content_hash_hex": "a" * 64,
        "purchase_tx_id": "tx_12345",
        "buyer_user_id": "user_alice",
        "timestamp_ns": 1_700_000_000_000_000_000,
    }
    cs_sig = sign_platform_countersig(
        cs_payload["content_hash_hex"],
        cs_payload["purchase_tx_id"],
        cs_payload["buyer_user_id"],
        cs_payload["timestamp_ns"],
        PLATFORM_SECRET,
    )
    countersig_cases.append(
        {
            **cs_payload,
            "timestamp_ns": str(cs_payload["timestamp_ns"]),  # JSON int64-safe
            "platform_secret_hex": PLATFORM_SECRET.hex(),
            "sig_hex": cs_sig.hex(),
        }
    )

    out_doc = {
        "version": 1,
        "user_secret_hex": USER_SECRET.hex(),
        "cases": records,
        "countersig_cases": countersig_cases,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out_doc, indent=2) + "\n")
    print(f"wrote {len(records)} tree vectors + {len(countersig_cases)} countersig to {OUT}")


if __name__ == "__main__":
    main()
