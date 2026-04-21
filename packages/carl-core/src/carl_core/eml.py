"""EML primitive: the exp-minus-log magma.

Implements the single-operation algebra introduced by Odrzywolek (arXiv 2603.21852,
2026-03-23 v2 2026-04-04):

    eml(x, y) = exp(x) - ln(y)
    S -> 1 | eml(S, S)

This grammar generates closed-form expressions for e, pi, i, +, -, *, /, ^,
sin, cos, sqrt, log. EML is a MAGMA only (not associative, not commutative,
no identity). Adam trainability holds at depth <= 4 (100% at d=2, ~25% at
d=3-4, 0/448 at d=6).

This module is torch-free. Pure math on numpy/float. Other teams consume the
stable public API below.
"""
from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from carl_core.errors import ValidationError
from carl_core.hashing import content_hash

# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

EPS: float = 1e-12
"""Additive floor guarding `ln` against non-positive inputs."""

MAX_DEPTH: int = 4
"""Adam trainability ceiling from Odrzywolek 2026. depth > 4 fails to converge."""

CLAMP_X: float = 20.0
"""Exp overflow guard. exp(20) ~ 4.85e8, well below float64 overflow."""

_IDENTITY_CONST: float = 1.0
"""The distinguished constant `1` that the EML grammar generates from."""


# ---------------------------------------------------------------------------
# Error codes (namespaced under carl.eml.*)
# ---------------------------------------------------------------------------


def _depth_exceeded(depth: int) -> ValidationError:
    return ValidationError(
        f"EML tree depth {depth} exceeds MAX_DEPTH={MAX_DEPTH}",
        code="carl.eml.depth_exceeded",
        context={"depth": depth, "max_depth": MAX_DEPTH},
    )


def _decode_error(reason: str, **ctx: Any) -> ValidationError:
    return ValidationError(
        f"EML decode failed: {reason}",
        code="carl.eml.decode_error",
        context={"reason": reason, **ctx},
    )


# ---------------------------------------------------------------------------
# Core node type
# ---------------------------------------------------------------------------


class EMLOp(IntEnum):
    """Node operation kind. Values are stable; used in the byte encoding."""

    CONST = 0
    VAR_X = 1  # placeholder for input binding (inputs[var_idx])
    EML = 2  # binary eml(left, right) node


@dataclass(frozen=True, slots=True)
class EMLNode:
    """A node in an EML expression tree.

    Leaves (depth 0): op in {CONST, VAR_X}. EML nodes: op == EML with both
    `left` and `right` populated. Invariants are enforced in `__post_init__`.
    """

    op: EMLOp
    const: float | None = None
    var_idx: int | None = None
    left: EMLNode | None = None
    right: EMLNode | None = None

    def __post_init__(self) -> None:
        if self.op == EMLOp.CONST:
            if self.const is None:
                raise ValidationError(
                    "CONST node requires `const` value",
                    code="carl.eml.domain_error",
                )
            if self.var_idx is not None or self.left is not None or self.right is not None:
                raise ValidationError(
                    "CONST node must not set var_idx/left/right",
                    code="carl.eml.domain_error",
                )
        elif self.op == EMLOp.VAR_X:
            if self.var_idx is None or self.var_idx < 0:
                raise ValidationError(
                    "VAR_X node requires non-negative `var_idx`",
                    code="carl.eml.domain_error",
                )
            if self.const is not None or self.left is not None or self.right is not None:
                raise ValidationError(
                    "VAR_X node must not set const/left/right",
                    code="carl.eml.domain_error",
                )
        elif self.op == EMLOp.EML:
            if self.left is None or self.right is None:
                raise ValidationError(
                    "EML node requires both `left` and `right`",
                    code="carl.eml.domain_error",
                )
            if self.const is not None or self.var_idx is not None:
                raise ValidationError(
                    "EML node must not set const/var_idx",
                    code="carl.eml.domain_error",
                )
        else:  # pragma: no cover - IntEnum exhaustive
            raise ValidationError(
                f"Unknown EML op: {self.op}",
                code="carl.eml.domain_error",
            )

    # -- structural helpers -------------------------------------------------

    def depth(self) -> int:
        """Max EML-nesting depth. Leaves = 0, EML(l,r) = 1 + max(depth(l), depth(r))."""
        if self.op in (EMLOp.CONST, EMLOp.VAR_X):
            return 0
        # EML node: children non-None by invariant.
        left = cast(EMLNode, self.left)
        right = cast(EMLNode, self.right)
        return 1 + max(left.depth(), right.depth())

    def nodes(self) -> int:
        """Total node count (leaves + EML nodes)."""
        if self.op in (EMLOp.CONST, EMLOp.VAR_X):
            return 1
        left = cast(EMLNode, self.left)
        right = cast(EMLNode, self.right)
        return 1 + left.nodes() + right.nodes()

    def input_dim_required(self) -> int:
        """Smallest `input_dim` that satisfies all VAR_X references in this tree."""
        if self.op == EMLOp.CONST:
            return 0
        if self.op == EMLOp.VAR_X:
            return cast(int, self.var_idx) + 1
        left = cast(EMLNode, self.left)
        right = cast(EMLNode, self.right)
        return max(left.input_dim_required(), right.input_dim_required())

    # -- canonical serialization -------------------------------------------

    def to_canonical_dict(self) -> dict[str, Any]:
        """Canonical, structure-only dict used for hashing.

        Two trees that compute the same expression tree (same shape, same
        leaves) produce the same canonical dict regardless of how they were
        constructed.
        """
        if self.op == EMLOp.CONST:
            # Normalise floats via repr rules so 1.0 == 1 canonicalise equal.
            return {"op": int(self.op), "const": float(cast(float, self.const))}
        if self.op == EMLOp.VAR_X:
            return {"op": int(self.op), "var_idx": int(cast(int, self.var_idx))}
        left = cast(EMLNode, self.left)
        right = cast(EMLNode, self.right)
        return {
            "op": int(self.op),
            "left": left.to_canonical_dict(),
            "right": right.to_canonical_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EMLNode:
        """Inverse of `to_canonical_dict`. Raises `carl.eml.decode_error` on malformed input."""
        if "op" not in d:
            raise _decode_error("missing 'op' field")
        try:
            op = EMLOp(int(d["op"]))
        except (ValueError, TypeError) as exc:
            raise _decode_error("invalid op value", op=d.get("op")) from exc
        if op == EMLOp.CONST:
            if "const" not in d:
                raise _decode_error("CONST node missing 'const'")
            return cls(op=EMLOp.CONST, const=float(d["const"]))
        if op == EMLOp.VAR_X:
            if "var_idx" not in d:
                raise _decode_error("VAR_X node missing 'var_idx'")
            return cls(op=EMLOp.VAR_X, var_idx=int(d["var_idx"]))
        # EML
        if "left" not in d or "right" not in d:
            raise _decode_error("EML node missing 'left' or 'right'")
        return cls(
            op=EMLOp.EML,
            left=cls.from_dict(cast(dict[str, Any], d["left"])),
            right=cls.from_dict(cast(dict[str, Any], d["right"])),
        )


# ---------------------------------------------------------------------------
# Scalar + vector primitives
# ---------------------------------------------------------------------------


def eml(x: float, y: float) -> float:
    """Scalar EML: exp(clamp(x)) - ln(max(y, EPS)).

    Clamping keeps exp in float64 safe range; EPS keeps ln finite. Both
    guards are conservative — they never change the value when inputs are
    in a sensible range.
    """
    xc = CLAMP_X if x > CLAMP_X else (-CLAMP_X if x < -CLAMP_X else x)
    yc = y if y > EPS else EPS
    return math.exp(xc) - math.log(yc)


def eml_array(
    x: NDArray[np.floating[Any]] | float,
    y: NDArray[np.floating[Any]] | float,
) -> NDArray[np.float64]:
    """Vectorized EML for numpy arrays (broadcasting follows numpy rules)."""
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    xc = np.clip(xa, -CLAMP_X, CLAMP_X)
    yc = np.maximum(ya, EPS)
    return np.exp(xc) - np.log(yc)


def eml_scalar_reward(coherence: float, dispersion: float) -> float:
    """Depth-1 reward: exp(coherence) - ln(max(dispersion, EPS)).

    Baseline reward primitive other teams (T2 reward shaping) consume. The
    reward is monotone increasing in coherence and monotone decreasing in
    dispersion, which matches CARL's conservation-law sign convention.
    """
    return eml(coherence, max(dispersion, EPS))


# ---------------------------------------------------------------------------
# Tree evaluation + serialization
# ---------------------------------------------------------------------------


# Postfix byte-code tags for compact serialization.
_TAG_CONST: int = 0x01
_TAG_VAR_X: int = 0x02
_TAG_EML: int = 0x03
_MAGIC: bytes = b"EML\x01"  # 4-byte header, version 1


@dataclass
class EMLTree:
    """An EML expression tree with input binding and learnable leaf params.

    `leaf_params` collects the constants of every CONST leaf in in-order
    traversal. This gives Adam a flat parameter vector to optimize without
    mutating the immutable node structure.
    """

    root: EMLNode
    input_dim: int
    leaf_params: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(0, dtype=np.float64)
    )

    def __post_init__(self) -> None:
        d = self.root.depth()
        if d > MAX_DEPTH:
            raise _depth_exceeded(d)
        required = self.root.input_dim_required()
        if required > self.input_dim:
            raise ValidationError(
                f"input_dim={self.input_dim} insufficient for VAR_X references (need {required})",
                code="carl.eml.domain_error",
                context={"input_dim": self.input_dim, "required": required},
            )
        # Sync leaf_params against tree constants if caller did not pre-fill.
        consts = _collect_consts(self.root)
        if self.leaf_params.size == 0 and consts:
            # numpy assignment replaces the field's default via object.__setattr__
            # (dataclass is not frozen, so direct attribute set is fine).
            self.leaf_params = np.asarray(consts, dtype=np.float64)
        elif self.leaf_params.size != len(consts):
            raise ValidationError(
                f"leaf_params length {self.leaf_params.size} != constant count {len(consts)}",
                code="carl.eml.domain_error",
                context={"leaf_params": int(self.leaf_params.size), "expected": len(consts)},
            )

    # -- depth + structural --

    def depth(self) -> int:
        return self.root.depth()

    def nodes(self) -> int:
        return self.root.nodes()

    # -- evaluation ---------------------------------------------------------

    def forward(self, inputs: NDArray[np.floating[Any]]) -> float:
        """Evaluate on a single input vector of length `input_dim`."""
        arr = np.asarray(inputs, dtype=np.float64).reshape(-1)
        if arr.shape[0] < self.input_dim:
            raise ValidationError(
                f"inputs length {arr.shape[0]} < input_dim {self.input_dim}",
                code="carl.eml.domain_error",
                context={"got": int(arr.shape[0]), "expected": self.input_dim},
            )
        # Bind leaf constants from the flat parameter vector.
        counter = [0]
        result = _eval(self.root, arr, self.leaf_params, counter)
        return float(result)

    def forward_batch(
        self, inputs: NDArray[np.floating[Any]]
    ) -> NDArray[np.float64]:
        """Evaluate on a batch of input vectors, shape (B, input_dim)."""
        arr = np.asarray(inputs, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2:
            raise ValidationError(
                f"forward_batch expects 2D input, got ndim={arr.ndim}",
                code="carl.eml.domain_error",
            )
        if arr.shape[1] < self.input_dim:
            raise ValidationError(
                f"input trailing dim {arr.shape[1]} < input_dim {self.input_dim}",
                code="carl.eml.domain_error",
            )
        # Naive per-row loop over the tree; depth is tiny so this is fine.
        out = np.empty(arr.shape[0], dtype=np.float64)
        for i in range(arr.shape[0]):
            counter = [0]
            out[i] = _eval(self.root, arr[i], self.leaf_params, counter)
        return out

    # -- gradients ---------------------------------------------------------

    def grad_wrt_params(
        self,
        inputs: NDArray[np.floating[Any]],
        target: float,
        *,
        eps: float = 1e-5,
    ) -> NDArray[np.float64]:
        """Finite-difference gradient of 0.5*(forward - target)^2 w.r.t. leaf_params."""
        base = self.forward(inputs) - target
        grads = np.zeros_like(self.leaf_params)
        for i in range(self.leaf_params.size):
            saved = float(self.leaf_params[i])
            self.leaf_params[i] = saved + eps
            plus = self.forward(inputs) - target
            self.leaf_params[i] = saved - eps
            minus = self.forward(inputs) - target
            self.leaf_params[i] = saved
            grads[i] = base * (plus - minus) / (2.0 * eps)
        return grads

    def grad_wrt_inputs(
        self,
        inputs: NDArray[np.floating[Any]],
        *,
        eps: float = 1e-5,
    ) -> NDArray[np.float64]:
        """Finite-difference gradient of forward w.r.t. the input vector."""
        arr = np.asarray(inputs, dtype=np.float64).reshape(-1).copy()
        grads = np.zeros(self.input_dim, dtype=np.float64)
        for i in range(self.input_dim):
            saved = float(arr[i])
            arr[i] = saved + eps
            plus = self.forward(arr)
            arr[i] = saved - eps
            minus = self.forward(arr)
            arr[i] = saved
            grads[i] = (plus - minus) / (2.0 * eps)
        return grads

    # -- hashing -----------------------------------------------------------

    def hash(self) -> str:
        """Stable sha256 of the canonical tree structure + leaf params.

        Two trees with the same shape, same VAR_X indices, and same constant
        values hash identically — even if built through different AST paths.
        """
        payload: dict[str, Any] = {
            "root": self.root.to_canonical_dict(),
            "input_dim": int(self.input_dim),
            # Round leaf_params to 15 decimal places to avoid float jitter
            # killing determinism. 15 decimals ~ float64 precision.
            "leaf_params": [round(float(v), 15) for v in self.leaf_params],
        }
        return content_hash(payload)

    # -- bytes serialization ------------------------------------------------

    def to_bytes(self) -> bytes:
        """Compact postfix byte encoding. Suitable for on-chain / cheap storage.

        Layout: 4-byte magic | uint16 input_dim | postfix tag stream.
        CONST = 0x01 + float64 (8 bytes)
        VAR_X = 0x02 + uint16 var_idx (2 bytes)
        EML   = 0x03  (children already on the decode stack)
        """
        out = bytearray(_MAGIC)
        out += struct.pack("<H", self.input_dim)
        counter = [0]
        _encode_postfix(self.root, self.leaf_params, counter, out)
        return bytes(out)

    @classmethod
    def from_bytes(cls, b: bytes) -> EMLTree:
        """Inverse of `to_bytes`. Validates magic + structural sanity."""
        if len(b) < len(_MAGIC) + 2:
            raise _decode_error("buffer too short for header")
        if b[: len(_MAGIC)] != _MAGIC:
            raise _decode_error("bad magic", magic=b[: len(_MAGIC)].hex())
        cursor = len(_MAGIC)
        (input_dim,) = struct.unpack_from("<H", b, cursor)
        cursor += 2
        stack: list[EMLNode] = []
        leaf_values: list[float] = []
        while cursor < len(b):
            tag = b[cursor]
            cursor += 1
            if tag == _TAG_CONST:
                if cursor + 8 > len(b):
                    raise _decode_error("truncated CONST payload")
                (val,) = struct.unpack_from("<d", b, cursor)
                cursor += 8
                stack.append(EMLNode(op=EMLOp.CONST, const=float(val)))
                leaf_values.append(float(val))
            elif tag == _TAG_VAR_X:
                if cursor + 2 > len(b):
                    raise _decode_error("truncated VAR_X payload")
                (idx,) = struct.unpack_from("<H", b, cursor)
                cursor += 2
                stack.append(EMLNode(op=EMLOp.VAR_X, var_idx=int(idx)))
            elif tag == _TAG_EML:
                if len(stack) < 2:
                    raise _decode_error("EML tag but fewer than 2 stack entries")
                right = stack.pop()
                left = stack.pop()
                stack.append(EMLNode(op=EMLOp.EML, left=left, right=right))
            else:
                raise _decode_error(f"unknown tag 0x{tag:02x}", tag=tag)
        if len(stack) != 1:
            raise _decode_error(
                f"postfix stream left {len(stack)} nodes on stack (expected 1)",
                stack_size=len(stack),
            )
        root = stack[0]
        params = np.asarray(leaf_values, dtype=np.float64)
        return cls(root=root, input_dim=int(input_dim), leaf_params=params)

    # -- human repr --------------------------------------------------------

    def __repr__(self) -> str:
        return f"EMLTree({_format_node(self.root)}, input_dim={self.input_dim}, depth={self.depth()})"

    # -- equality ---------------------------------------------------------

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, EMLTree):
            return NotImplemented
        if self.input_dim != other.input_dim:
            return False
        if self.root != other.root:
            return False
        if self.leaf_params.shape != other.leaf_params.shape:
            return False
        return bool(np.allclose(self.leaf_params, other.leaf_params, atol=1e-12))

    def __hash__(self) -> int:
        return hash(self.hash())

    # -- factory constructors ---------------------------------------------

    @classmethod
    def identity(cls, input_dim: int) -> EMLTree:
        """Return the identity tree for input[0]: a single VAR_X leaf.

        The Odrzywolek grammar's natural identity is just the variable itself
        (depth 0). The paper's reference "identity trick" via exp(ln(x)) is
        available as `identity_deep`; it sits at depth 4 which is the max
        allowed, so it cannot be composed further.
        """
        if input_dim < 1:
            raise ValidationError(
                "identity requires input_dim >= 1",
                code="carl.eml.domain_error",
            )
        root = EMLNode(op=EMLOp.VAR_X, var_idx=0)
        return cls(root=root, input_dim=input_dim)

    @classmethod
    def identity_deep(cls, input_dim: int) -> EMLTree:
        """exp(ln(x)) == x, at depth 4.

        Right at MAX_DEPTH. Exists for Adam trainability benchmarking and
        to exercise the composition closure at the ceiling.
        """
        if input_dim < 1:
            raise ValidationError(
                "identity_deep requires input_dim >= 1",
                code="carl.eml.domain_error",
            )
        # ln(x) = eml(1, eml(eml(1, x), 1))
        one = EMLNode(op=EMLOp.CONST, const=_IDENTITY_CONST)
        x = EMLNode(op=EMLOp.VAR_X, var_idx=0)
        eml_1_x = EMLNode(op=EMLOp.EML, left=one, right=x)
        eml_eml1x_1 = EMLNode(
            op=EMLOp.EML, left=eml_1_x, right=EMLNode(op=EMLOp.CONST, const=_IDENTITY_CONST)
        )
        ln_x = EMLNode(
            op=EMLOp.EML,
            left=EMLNode(op=EMLOp.CONST, const=_IDENTITY_CONST),
            right=eml_eml1x_1,
        )
        # exp(ln(x)) = eml(ln(x), 1)
        root = EMLNode(
            op=EMLOp.EML, left=ln_x, right=EMLNode(op=EMLOp.CONST, const=_IDENTITY_CONST)
        )
        return cls(root=root, input_dim=input_dim)

    @classmethod
    def exp_single(cls) -> EMLTree:
        """exp(x) = eml(x, 1). input_dim=1, depth=1."""
        root = EMLNode(
            op=EMLOp.EML,
            left=EMLNode(op=EMLOp.VAR_X, var_idx=0),
            right=EMLNode(op=EMLOp.CONST, const=_IDENTITY_CONST),
        )
        return cls(root=root, input_dim=1)

    @classmethod
    def ln_single(cls) -> EMLTree:
        """ln(x) = eml(1, eml(eml(1, x), 1)). input_dim=1, depth=3."""
        one = EMLNode(op=EMLOp.CONST, const=_IDENTITY_CONST)
        x = EMLNode(op=EMLOp.VAR_X, var_idx=0)
        inner = EMLNode(op=EMLOp.EML, left=one, right=x)
        middle = EMLNode(
            op=EMLOp.EML, left=inner, right=EMLNode(op=EMLOp.CONST, const=_IDENTITY_CONST)
        )
        root = EMLNode(
            op=EMLOp.EML,
            left=EMLNode(op=EMLOp.CONST, const=_IDENTITY_CONST),
            right=middle,
        )
        return cls(root=root, input_dim=1)

    @classmethod
    def zero(cls) -> EMLTree:
        """The canonical 0 = eml(1, eml(eml(1, 1), 1))."""
        one = EMLNode(op=EMLOp.CONST, const=_IDENTITY_CONST)
        inner = EMLNode(
            op=EMLOp.EML,
            left=EMLNode(op=EMLOp.CONST, const=_IDENTITY_CONST),
            right=EMLNode(op=EMLOp.CONST, const=_IDENTITY_CONST),
        )
        middle = EMLNode(
            op=EMLOp.EML, left=inner, right=EMLNode(op=EMLOp.CONST, const=_IDENTITY_CONST)
        )
        root = EMLNode(op=EMLOp.EML, left=one, right=middle)
        # No variables, but keep input_dim=1 so it composes uniformly.
        return cls(root=root, input_dim=1)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _collect_consts(node: EMLNode) -> list[float]:
    """In-order traversal collecting CONST values."""
    if node.op == EMLOp.CONST:
        return [float(cast(float, node.const))]
    if node.op == EMLOp.VAR_X:
        return []
    left = cast(EMLNode, node.left)
    right = cast(EMLNode, node.right)
    return _collect_consts(left) + _collect_consts(right)


def _eval(
    node: EMLNode,
    inputs: NDArray[np.float64],
    params: NDArray[np.float64],
    counter: list[int],
) -> float:
    """Evaluate a node, consuming leaf constants from `params` in-order."""
    if node.op == EMLOp.CONST:
        idx = counter[0]
        counter[0] += 1
        if idx < params.size:
            return float(params[idx])
        return float(cast(float, node.const))
    if node.op == EMLOp.VAR_X:
        return float(inputs[cast(int, node.var_idx)])
    left = cast(EMLNode, node.left)
    right = cast(EMLNode, node.right)
    x = _eval(left, inputs, params, counter)
    y = _eval(right, inputs, params, counter)
    return eml(x, y)


def _encode_postfix(
    node: EMLNode,
    params: NDArray[np.float64],
    counter: list[int],
    out: bytearray,
) -> None:
    """Postorder write of the tree into `out`."""
    if node.op == EMLOp.CONST:
        idx = counter[0]
        counter[0] += 1
        if idx < params.size:
            val = float(params[idx])
        else:
            val = float(cast(float, node.const))
        out.append(_TAG_CONST)
        out += struct.pack("<d", val)
        return
    if node.op == EMLOp.VAR_X:
        out.append(_TAG_VAR_X)
        out += struct.pack("<H", int(cast(int, node.var_idx)))
        return
    left = cast(EMLNode, node.left)
    right = cast(EMLNode, node.right)
    _encode_postfix(left, params, counter, out)
    _encode_postfix(right, params, counter, out)
    out.append(_TAG_EML)


def _format_node(node: EMLNode) -> str:
    """Human-readable formula string."""
    if node.op == EMLOp.CONST:
        return str(node.const)
    if node.op == EMLOp.VAR_X:
        return f"x{node.var_idx}"
    left = cast(EMLNode, node.left)
    right = cast(EMLNode, node.right)
    return f"eml({_format_node(left)}, {_format_node(right)})"


__all__ = [
    "EPS",
    "MAX_DEPTH",
    "CLAMP_X",
    "EMLOp",
    "EMLNode",
    "EMLTree",
    "eml",
    "eml_array",
    "eml_scalar_reward",
]
