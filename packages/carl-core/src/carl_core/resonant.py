"""Resonant: a perception-cognition-action triple built on EML.

A `Resonant` wraps an EML expression tree with linear projection (observation
-> latent) and linear readout (latent -> action) matrices. The tree performs
the nonlinear "cognition" step per latent dimension; the matrices carry the
learnable basis change around it.

Closure under composition: `compose_resonants(r1, r2)` yields a new Resonant
whose cognition is `eml(r1.tree, r2.tree)`, provided the shape contract
holds. This preserves the EML magma structure at the Resonant level.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from carl_core.eml import EMLNode, EMLOp, EMLTree, MAX_DEPTH
from carl_core.errors import ValidationError
from carl_core.hashing import content_hash_bytes


def _depth_exceeded(depth: int) -> ValidationError:
    return ValidationError(
        f"Composed EML tree depth {depth} exceeds MAX_DEPTH={MAX_DEPTH}",
        code="carl.eml.depth_exceeded",
        context={"depth": depth, "max_depth": MAX_DEPTH},
    )


def _shape_error(msg: str, **ctx: Any) -> ValidationError:
    return ValidationError(
        msg, code="carl.eml.domain_error", context=dict(ctx)
    )


@dataclass(frozen=True)
class Resonant:
    """Perceive -> cognize -> act triple.

    Attributes:
        tree: EML tree applied per latent dimension during `cognize`.
        projection: (k, d) matrix mapping observation (d-dim) -> latent (k-dim).
        readout: (a, k) matrix mapping latent (k-dim) -> action (a-dim).
        identity: sha256 hex digest of (tree.hash, projection bytes, readout bytes).
            Stable identifier suitable for content-addressed storage.
        metadata: free-form annotations; excluded from identity.
    """

    tree: EMLTree
    projection: NDArray[np.float64]
    readout: NDArray[np.float64]
    identity: str
    metadata: dict[str, Any] = field(default_factory=lambda: {})  # noqa: C408

    # -- shape accessors --------------------------------------------------

    @property
    def observation_dim(self) -> int:
        return int(self.projection.shape[1])

    @property
    def latent_dim(self) -> int:
        return int(self.projection.shape[0])

    @property
    def action_dim(self) -> int:
        return int(self.readout.shape[0])

    # -- pipeline stages --------------------------------------------------

    def perceive(self, observation: NDArray[np.floating[Any]]) -> NDArray[np.float64]:
        """Project an observation into the latent space. obs shape: (d,) or (B, d)."""
        obs = np.asarray(observation, dtype=np.float64)
        if obs.ndim == 1:
            if obs.shape[0] != self.observation_dim:
                raise _shape_error(
                    f"observation dim {obs.shape[0]} != expected {self.observation_dim}",
                    got=int(obs.shape[0]),
                    expected=self.observation_dim,
                )
            return self.projection @ obs
        if obs.ndim == 2:
            if obs.shape[1] != self.observation_dim:
                raise _shape_error(
                    f"observation dim {obs.shape[1]} != expected {self.observation_dim}",
                    got=int(obs.shape[1]),
                    expected=self.observation_dim,
                )
            return obs @ self.projection.T
        raise _shape_error(f"perceive expects 1D or 2D input, got ndim={obs.ndim}")

    def cognize(self, latent: NDArray[np.floating[Any]]) -> NDArray[np.float64]:
        """Apply the EML tree per latent dimension.

        Latent shape: (k,) or (B, k). The tree takes each scalar latent value
        as its input[0] — so the tree must have input_dim >= 1.
        """
        lat = np.asarray(latent, dtype=np.float64)
        if self.tree.input_dim < 1:
            raise _shape_error("Resonant tree must have input_dim >= 1 for cognize")
        if lat.ndim == 1:
            if lat.shape[0] != self.latent_dim:
                raise _shape_error(
                    f"latent dim {lat.shape[0]} != expected {self.latent_dim}",
                    got=int(lat.shape[0]),
                    expected=self.latent_dim,
                )
            out = np.empty(self.latent_dim, dtype=np.float64)
            for i in range(self.latent_dim):
                # Pad with zeros if tree.input_dim > 1 (extra inputs unused).
                vec = np.zeros(self.tree.input_dim, dtype=np.float64)
                vec[0] = lat[i]
                out[i] = self.tree.forward(vec)
            return out
        if lat.ndim == 2:
            if lat.shape[1] != self.latent_dim:
                raise _shape_error(
                    f"latent dim {lat.shape[1]} != expected {self.latent_dim}",
                    got=int(lat.shape[1]),
                    expected=self.latent_dim,
                )
            out2 = np.empty_like(lat)
            for b in range(lat.shape[0]):
                for i in range(self.latent_dim):
                    vec = np.zeros(self.tree.input_dim, dtype=np.float64)
                    vec[0] = lat[b, i]
                    out2[b, i] = self.tree.forward(vec)
            return out2
        raise _shape_error(f"cognize expects 1D or 2D input, got ndim={lat.ndim}")

    def act(self, latent: NDArray[np.floating[Any]]) -> NDArray[np.float64]:
        """Project a (post-cognition) latent through the readout matrix."""
        lat = np.asarray(latent, dtype=np.float64)
        if lat.ndim == 1:
            if lat.shape[0] != self.latent_dim:
                raise _shape_error(
                    f"latent dim {lat.shape[0]} != expected {self.latent_dim}",
                )
            return self.readout @ lat
        if lat.ndim == 2:
            if lat.shape[1] != self.latent_dim:
                raise _shape_error(
                    f"latent dim {lat.shape[1]} != expected {self.latent_dim}",
                )
            return lat @ self.readout.T
        raise _shape_error(f"act expects 1D or 2D input, got ndim={lat.ndim}")

    def forward(self, observation: NDArray[np.floating[Any]]) -> NDArray[np.float64]:
        """Full pipeline: perceive -> cognize -> act."""
        return self.act(self.cognize(self.perceive(observation)))

    # -- serialization ----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "tree": self.tree.to_bytes().hex(),
            "input_dim": int(self.tree.input_dim),
            "projection": {
                "shape": list(self.projection.shape),
                "data": self.projection.astype(np.float64).tolist(),
            },
            "readout": {
                "shape": list(self.readout.shape),
                "data": self.readout.astype(np.float64).tolist(),
            },
            "identity": self.identity,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Resonant:
        if "tree" not in d or "projection" not in d or "readout" not in d:
            raise ValidationError(
                "Resonant.from_dict missing required keys",
                code="carl.eml.decode_error",
                context={"have": sorted(list(d.keys()))},
            )
        try:
            tree_bytes = bytes.fromhex(cast(str, d["tree"]))
            tree = EMLTree.from_bytes(tree_bytes)
        except Exception as exc:
            raise ValidationError(
                "Resonant.from_dict failed to decode tree",
                code="carl.eml.decode_error",
                cause=exc,
            ) from exc
        proj_data = d["projection"]["data"]
        read_data = d["readout"]["data"]
        projection = np.asarray(proj_data, dtype=np.float64)
        readout = np.asarray(read_data, dtype=np.float64)
        metadata = dict(d.get("metadata", {}))
        identity = _compute_identity(tree, projection, readout)
        # If caller provided an identity, it must match (tamper detection).
        given_id = d.get("identity")
        if isinstance(given_id, str) and given_id != identity:
            raise ValidationError(
                "Resonant identity mismatch on decode",
                code="carl.eml.decode_error",
                context={"expected": identity, "got": given_id},
            )
        return cls(
            tree=tree,
            projection=projection,
            readout=readout,
            identity=identity,
            metadata=metadata,
        )

    # -- equality (identity-keyed) ---------------------------------------

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Resonant):
            return NotImplemented
        return (
            self.identity == other.identity
            and self.metadata == other.metadata
        )

    def __hash__(self) -> int:
        return hash(self.identity)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def make_resonant(
    tree: EMLTree,
    projection: NDArray[np.floating[Any]],
    readout: NDArray[np.floating[Any]],
    *,
    metadata: dict[str, Any] | None = None,
) -> Resonant:
    """Build a Resonant, validating shapes and computing the identity."""
    proj = np.asarray(projection, dtype=np.float64)
    read = np.asarray(readout, dtype=np.float64)
    if proj.ndim != 2:
        raise _shape_error(
            f"projection must be 2D, got ndim={proj.ndim}", ndim=int(proj.ndim)
        )
    if read.ndim != 2:
        raise _shape_error(
            f"readout must be 2D, got ndim={read.ndim}", ndim=int(read.ndim)
        )
    if proj.shape[0] != read.shape[1]:
        raise _shape_error(
            f"projection rows ({proj.shape[0]}) must match readout cols ({read.shape[1]})",
            projection_rows=int(proj.shape[0]),
            readout_cols=int(read.shape[1]),
        )
    if tree.input_dim < 1:
        raise _shape_error("Resonant requires EMLTree.input_dim >= 1")
    ident = _compute_identity(tree, proj, read)
    return Resonant(
        tree=tree,
        projection=proj,
        readout=read,
        identity=ident,
        metadata=dict(metadata or {}),
    )


def _compute_identity(
    tree: EMLTree,
    projection: NDArray[np.float64],
    readout: NDArray[np.float64],
) -> str:
    """sha256(tree.hash || projection bytes || readout bytes).

    The matrices are hashed via their contiguous float64 bytes after rounding
    to 12 decimals — keeps identity stable across innocuous rewrites while
    still distinguishing real parameter changes.
    """
    proj_q = np.round(projection, 12).astype(np.float64, copy=True)
    read_q = np.round(readout, 12).astype(np.float64, copy=True)
    tree_hash = tree.hash()
    proj_hash = content_hash_bytes(np.ascontiguousarray(proj_q).tobytes())
    read_hash = content_hash_bytes(np.ascontiguousarray(read_q).tobytes())
    combined = f"{tree_hash}|{proj_hash}|{read_hash}".encode("utf-8")
    return content_hash_bytes(combined)


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------


def compose_resonants(r1: Resonant, r2: Resonant) -> Resonant:
    """Compose two Resonants under the EML magma.

    The composed Resonant cognizes with a new node `eml(r1.tree, r2.tree)`
    built over the two children. To stay a valid Resonant in its own right,
    the two operands must share the latent space that cognition runs on:

      - r1.latent_dim == r2.latent_dim  (both trees operate in the same latent)
      - depth(eml(r1.tree.root, r2.tree.root)) <= MAX_DEPTH

    The composed Resonant inherits r2's projection (the shared perception
    step that drives both children) and r1's readout (the downstream
    action), preserving the overall pipeline signature.
    """
    if r1.latent_dim != r2.latent_dim:
        raise _shape_error(
            "compose_resonants: latent dimensions must match",
            r1_latent_dim=r1.latent_dim,
            r2_latent_dim=r2.latent_dim,
        )
    if r1.observation_dim != r2.observation_dim:
        raise _shape_error(
            "compose_resonants: observation dimensions must match",
            r1_observation_dim=r1.observation_dim,
            r2_observation_dim=r2.observation_dim,
        )
    if r1.action_dim != r2.action_dim:
        raise _shape_error(
            "compose_resonants: action dimensions must match",
            r1_action_dim=r1.action_dim,
            r2_action_dim=r2.action_dim,
        )
    # Build the composed tree: eml(r1.root, r2.root). This is the Odrzywolek
    # closure — a new EML node over the two trees.
    composed_root = EMLNode(
        op=EMLOp.EML, left=r1.tree.root, right=r2.tree.root
    )
    new_depth = composed_root.depth()
    if new_depth > MAX_DEPTH:
        raise _depth_exceeded(new_depth)
    # Input dim of the composed tree is the max required by either child.
    input_dim = max(
        composed_root.input_dim_required(),
        r1.tree.input_dim,
        r2.tree.input_dim,
    )
    composed_tree = EMLTree(root=composed_root, input_dim=input_dim)
    # Average the projection/readout matrices — a neutral composition choice
    # that preserves both operands' shapes and lets downstream code learn
    # the mixing. The identity matrices are preserved exactly in this case.
    new_projection = 0.5 * (r1.projection + r2.projection)
    new_readout = 0.5 * (r1.readout + r2.readout)
    metadata = {
        "composed_of": [r1.identity, r2.identity],
        "composition_depth": int(new_depth),
    }
    return make_resonant(
        composed_tree, new_projection, new_readout, metadata=metadata
    )


__all__ = [
    "Resonant",
    "make_resonant",
    "compose_resonants",
]
