"""EML composite reward — third option alongside CARLReward and PhaseAdaptiveCARLReward.

The EML (Exponential-Minus-Log) operator ``eml(x, y) = exp(x) - ln(y)`` is a
non-linear binary combiner that is Adam-trainable up to depth 4 without
derivative explosion. At depth <= 3 we can approximate the linear CARL reward
``0.5*ms + 0.3*cq + 0.2*defect`` while leaving the tree free to learn
non-linear interactions between the three features.

This module is a *drop-in* reward head: ``score_from_trace`` matches the
signature used by ``CARLReward`` / ``PhaseAdaptiveCARLReward`` so the same
``CascadeRewardManager`` and monitoring callbacks work unchanged.

Built on top of ``carl_core.eml`` (Odrzywolek 2026): EMLTree, EMLNode, EMLOp,
``eml``, ``eml_scalar_reward``, MAX_DEPTH.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from carl_core.eml import (
    MAX_DEPTH,
    EMLNode,
    EMLOp,
    EMLTree,
    eml_scalar_reward,
)


# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

_REWARD_CLAMP_DEFAULT: float = 5.0
"""Default tanh-scaled output bound. Matches existing CARL reward order."""

_FEATURE_DIM: int = 3
"""Input dimensionality: (multiscale_coherence, cloud_quality, defect_score)."""

_ADAM_EPS: float = 1e-8
"""Adam numerical floor."""

_GRAD_FD_EPS: float = 1e-4
"""Finite-difference step for leaf_params gradients. Matches carl_core.eml
convention; leaf constants are O(1) so this is stable."""

# Pre-squash calibration. Derived empirically (see scripts/benchmark_eml_reward.py)
# by sweeping 100 random traces through the default tree and linear-regressing
# the raw EML output against CARLReward.score_from_trace. These constants
# bring the init-time output into the "drop-in" range (within ~0.3 abs of
# CARLReward for typical traces) AND preserve the sign of the correlation so
# that a "good" CARL trace stays "good" under EML scoring. Adam can then
# fine-tune the leaf CONSTs from there.
#
# Note the NEGATIVE scale: the default tree's raw output is anti-correlated
# with the linear CARL reward because exp(ms) in the inner position rewards
# disorder not order. The calibration flips the sign so downstream consumers
# see the conventional "higher = better" monotonicity.
_RAW_TO_LINEAR_SCALE: float = -0.035
_RAW_TO_LINEAR_SHIFT: float = 0.74

# Lower-bound for features entering the ``ln`` slot of an EML node. Without
# this floor, cq or defect values near 0 produce huge -ln() terms that blow
# up the tree's exp() arg and lose all numerical sanity.
_FEATURE_LN_FLOOR: float = 0.05


# ---------------------------------------------------------------------------
# Public helper — depth-1 baseline reward
# ---------------------------------------------------------------------------


def eml_reward_from_trace(trace: Any) -> float:
    """Depth-1 baseline reward: ``exp(coherence) - ln(dispersion)``.

    Uses ``trace.multiscale_coherence`` and ``(1 - trace.phi_mean)`` as the
    dispersion proxy (higher phi -> lower dispersion -> larger ln penalty).
    Output is tanh-bounded to ``[-_REWARD_CLAMP_DEFAULT, _REWARD_CLAMP_DEFAULT]``.
    """
    coherence = float(getattr(trace, "multiscale_coherence", 0.5))
    phi = float(getattr(trace, "phi_mean", 0.5))
    dispersion = max(1.0 - phi, 1e-8)
    raw = eml_scalar_reward(coherence, dispersion)
    half = _REWARD_CLAMP_DEFAULT
    return float(half * math.tanh(raw / max(half, 1e-6)))


# ---------------------------------------------------------------------------
# Default tree builder
# ---------------------------------------------------------------------------


def _default_tree() -> EMLTree:
    """Default tree approximating the CARL composite reward at depth 3.

    Topology (depth 3, 3 CONST + 3 VAR_X leaves -> 3 trainable params):

        root = eml(
            inner = eml(
                ms_shift  = eml(CONST_a, VAR_X[0]),   # ~= exp(a) - ln(ms)
                cq_shift  = eml(CONST_b, VAR_X[1]),   # ~= exp(b) - ln(cq)
            ),
            def_shift = eml(CONST_c, VAR_X[2]),        # exp(c) - ln(defect)
        )

    Depth accounting:
      ``eml(CONST, VAR)`` -> depth 1 (leaves at depth 0)
      ``eml(leaf1, leaf1)`` (inner) -> depth 2
      ``eml(inner, leaf1)`` (root) -> depth 3

    Which matches ``MAX_DEPTH - 1`` so there is headroom for one more level.
    The CONST leaves are Adam-trainable via ``leaf_params``. Initial values
    are zero so the tree reduces to pure feature transformation at init time;
    the ``_squash`` affine calibration handles the drop-in matching against
    the linear CARL reward, and Adam can learn per-feature shifts from there.
    """
    # ms_shift = eml(CONST_a, VAR_X[0]) = exp(a) - ln(ms)
    ms_shift = EMLNode(
        op=EMLOp.EML,
        left=EMLNode(op=EMLOp.CONST, const=0.0),
        right=EMLNode(op=EMLOp.VAR_X, var_idx=0),
    )
    # cq_shift = eml(CONST_b, VAR_X[1]) = exp(b) - ln(cq)
    cq_shift = EMLNode(
        op=EMLOp.EML,
        left=EMLNode(op=EMLOp.CONST, const=0.0),
        right=EMLNode(op=EMLOp.VAR_X, var_idx=1),
    )
    # def_shift = eml(CONST_c, VAR_X[2]) = exp(c) - ln(defect)
    def_shift = EMLNode(
        op=EMLOp.EML,
        left=EMLNode(op=EMLOp.CONST, const=0.0),
        right=EMLNode(op=EMLOp.VAR_X, var_idx=2),
    )
    # inner = eml(ms_shift, cq_shift)
    inner = EMLNode(op=EMLOp.EML, left=ms_shift, right=cq_shift)
    # root = eml(inner, def_shift)
    root = EMLNode(op=EMLOp.EML, left=inner, right=def_shift)
    return EMLTree(root=root, input_dim=_FEATURE_DIM)


# ---------------------------------------------------------------------------
# EMLCompositeReward — depth-3 learnable tree, drop-in for CARLReward
# ---------------------------------------------------------------------------


class EMLCompositeReward:
    """Depth-<=3 learnable EML tree — third reward-head option.

    Input features: ``[multiscale_coherence, cloud_quality, defect_score]``.
    Output: scalar reward, tanh-bounded into the configured clamp range.

    Leaf CONST params are Adam-trainable via ``fit_step``. The default tree
    approximates ``eml(eml(ms, cq), defect)`` at depth 3; max-depth is capped
    at ``MAX_DEPTH=4`` (hard-enforced against carl_core constants).

    The reward object is duck-compatible with ``CARLReward`` and
    ``PhaseAdaptiveCARLReward`` -- the same ``score_from_trace(trace) ->
    (reward, meta)`` signature.
    """

    def __init__(
        self,
        tree: EMLTree | None = None,
        max_depth: int = 3,
        clamp: tuple[float, float] = (-_REWARD_CLAMP_DEFAULT, _REWARD_CLAMP_DEFAULT),
    ) -> None:
        if max_depth > MAX_DEPTH:
            raise ValueError(
                f"max_depth {max_depth} exceeds EML MAX_DEPTH={MAX_DEPTH}"
            )
        self.max_depth: int = int(max_depth)
        self.tree: EMLTree = tree if tree is not None else _default_tree()
        actual_depth = self.tree.depth()
        if actual_depth > self.max_depth:
            raise ValueError(
                f"tree depth {actual_depth} exceeds max_depth {self.max_depth}"
            )
        # Ensure the tree's feature dim covers our 3-dim input.
        if self.tree.input_dim < _FEATURE_DIM:
            raise ValueError(
                f"tree.input_dim={self.tree.input_dim} < required {_FEATURE_DIM}"
            )
        self.clamp: tuple[float, float] = (float(clamp[0]), float(clamp[1]))
        # Adam moments — one per leaf param (CONST leaf in the tree).
        n_params = int(self.tree.leaf_params.size)
        self._adam_m: np.ndarray = np.zeros(n_params, dtype=np.float64)
        self._adam_v: np.ndarray = np.zeros(n_params, dtype=np.float64)
        self._adam_t: int = 0

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _features_from_trace(self, trace: Any) -> np.ndarray:
        ms = float(getattr(trace, "multiscale_coherence", 0.5))
        cq = float(getattr(trace, "cloud_quality", 0.5))
        defect = float(getattr(trace, "discontinuity_score", 0.5))
        return np.array([ms, cq, defect], dtype=np.float64)

    def _squash(self, raw: float) -> float:
        """Calibrate raw EML output to CARL's [0, 1]-ish range, then tanh-clamp.

        Two-stage pipeline:
          1. Affine calibration: linearized = RAW_TO_LINEAR_SCALE * raw + SHIFT
             (so the init-time output is a drop-in CARL approximation).
          2. Tanh squash into the configured clamp window so the reward can
             never runaway during training even if leaf_params drift.
        """
        lo, hi = self.clamp
        half = (hi - lo) / 2.0
        mid = (hi + lo) / 2.0
        if not math.isfinite(raw):
            return float(mid)
        if half <= 0:
            return float(mid)
        linearized = _RAW_TO_LINEAR_SCALE * raw + _RAW_TO_LINEAR_SHIFT
        # Center around the clamp midpoint before tanh. For the default clamp
        # (-5, 5) this is 0; users with shifted clamps still get a sensible
        # output.
        centered = linearized - mid
        return float(mid + half * math.tanh(centered / half))

    def forward(self, features: np.ndarray) -> float:
        """Evaluate the tree on a 3-dim feature vector and tanh-clamp."""
        feats = np.asarray(features, dtype=np.float64).reshape(-1)
        # Floor the two "dispersion-like" features (cq, defect) so they never
        # hit the ``ln`` slot with a near-zero value. Without this guard, the
        # tree's exp() argument blows up and the output saturates instantly.
        # Coherence (ms) is in the ``exp`` slot so the floor there is lighter.
        if feats.shape[0] >= 1:
            feats[0] = max(feats[0], 0.0)
        if feats.shape[0] >= 2:
            feats[1] = max(feats[1], _FEATURE_LN_FLOOR)
        if feats.shape[0] >= 3:
            feats[2] = max(feats[2], _FEATURE_LN_FLOOR)
        feats = np.clip(feats, 0.0, 1e6)
        raw = self.tree.forward(feats)
        return self._squash(raw)

    def score_from_trace(
        self, trace: Any
    ) -> tuple[float, dict[str, float]]:
        """Evaluate on (ms, cq, defect) -> (reward, metadata).

        Matches the signature of ``PhaseAdaptiveCARLReward.score_from_trace``
        so the TRL closure in ``make_carl_reward`` works unchanged.
        """
        feats = self._features_from_trace(trace)
        reward = self.forward(feats)
        meta: dict[str, float] = {
            "ms": float(feats[0]),
            "cq": float(feats[1]),
            "defect": float(feats[2]),
            "tree_depth": float(self.tree.depth()),
        }
        return reward, meta

    def __call__(
        self, trace: Any
    ) -> tuple[float, dict[str, float]]:
        return self.score_from_trace(trace)

    # ------------------------------------------------------------------
    # Training — Adam on leaf_params (CONST leaves)
    # ------------------------------------------------------------------

    def fit_step(
        self,
        inputs: np.ndarray,
        target: float,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = _ADAM_EPS,
    ) -> dict[str, float]:
        """Single Adam update on tree.leaf_params. Returns ``{loss, grad_norm}``.

        Loss: ``0.5 * (forward(inputs) - target) ** 2`` (in *pre-squash* space
        for a cleaner gradient signal; the tanh clamp is applied only at the
        inference boundary).

        Gradient: finite-difference directly on ``tree.leaf_params``. This
        matches the approach used by ``EMLTree.grad_wrt_params`` but we roll
        it locally so ``loss`` is reported consistently.
        """
        feats = np.asarray(inputs, dtype=np.float64).reshape(-1).copy()
        # Same feature floors as ``forward`` — must stay in sync so gradient
        # matches the loss surface the model actually sees at inference.
        if feats.shape[0] >= 1:
            feats[0] = max(feats[0], 0.0)
        if feats.shape[0] >= 2:
            feats[1] = max(feats[1], _FEATURE_LN_FLOOR)
        if feats.shape[0] >= 3:
            feats[2] = max(feats[2], _FEATURE_LN_FLOOR)
        feats = np.clip(feats, 0.0, 1e6)
        target = float(target)
        params = self.tree.leaf_params
        n = int(params.size)
        if n == 0:
            # No trainable leaves — just report current loss.
            raw = self.tree.forward(feats)
            return {"loss": 0.5 * (raw - target) ** 2, "grad_norm": 0.0}

        # Re-size Adam moments if tree was replaced externally.
        if self._adam_m.size != n:
            self._adam_m = np.zeros(n, dtype=np.float64)
            self._adam_v = np.zeros(n, dtype=np.float64)

        def _loss(raw: float) -> float:
            return 0.5 * (raw - target) ** 2

        h = _GRAD_FD_EPS
        grads = np.zeros(n, dtype=np.float64)
        for i in range(n):
            saved = float(params[i])
            params[i] = saved + h
            lp = _loss(self.tree.forward(feats))
            params[i] = saved - h
            lm = _loss(self.tree.forward(feats))
            params[i] = saved
            g = (lp - lm) / (2.0 * h)
            if not math.isfinite(g):
                g = 0.0
            grads[i] = g

        # Adam update
        self._adam_t += 1
        t = self._adam_t
        bc1 = 1.0 - beta1 ** t
        bc2 = 1.0 - beta2 ** t
        self._adam_m = beta1 * self._adam_m + (1 - beta1) * grads
        self._adam_v = beta2 * self._adam_v + (1 - beta2) * (grads * grads)
        m_hat = self._adam_m / bc1
        v_hat = self._adam_v / bc2
        updates = lr * m_hat / (np.sqrt(v_hat) + eps)
        # In-place update of the tree's leaf params
        params -= updates

        raw_after = self.tree.forward(feats)
        grad_norm = float(np.linalg.norm(grads))
        return {
            "loss": float(_loss(raw_after)),
            "grad_norm": grad_norm,
        }

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize tree + Adam state to a plain dict."""
        return {
            "version": 1,
            "max_depth": self.max_depth,
            "clamp": list(self.clamp),
            "root": self.tree.root.to_canonical_dict(),
            "input_dim": int(self.tree.input_dim),
            "leaf_params": [float(v) for v in self.tree.leaf_params],
            "adam_m": [float(v) for v in self._adam_m],
            "adam_v": [float(v) for v in self._adam_v],
            "adam_t": self._adam_t,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "EMLCompositeReward":
        """Reconstruct from ``to_dict`` output. Preserves reward output exactly."""
        root_dict = d.get("root")
        if root_dict is None:
            raise ValueError("EMLCompositeReward.from_dict: missing root")
        root = EMLNode.from_dict(root_dict)
        leaf_params = np.asarray(
            d.get("leaf_params", []), dtype=np.float64
        )
        tree = EMLTree(
            root=root,
            input_dim=int(d.get("input_dim", _FEATURE_DIM)),
            leaf_params=leaf_params,
        )
        clamp_raw = d.get(
            "clamp", [-_REWARD_CLAMP_DEFAULT, _REWARD_CLAMP_DEFAULT]
        )
        clamp_lo = float(clamp_raw[0])
        clamp_hi = float(clamp_raw[1])
        obj = cls(
            tree=tree,
            max_depth=int(d.get("max_depth", 3)),
            clamp=(clamp_lo, clamp_hi),
        )
        adam_m = np.asarray(d.get("adam_m", []), dtype=np.float64)
        adam_v = np.asarray(d.get("adam_v", []), dtype=np.float64)
        # Only adopt moments when they match the current param count — this
        # prevents a corrupt dict from silently desyncing Adam state.
        if adam_m.size == obj._adam_m.size:
            obj._adam_m = adam_m
        if adam_v.size == obj._adam_v.size:
            obj._adam_v = adam_v
        obj._adam_t = int(d.get("adam_t", 0))
        return obj


# ---------------------------------------------------------------------------
# Factory — matches the public surface of ``make_carl_reward``
# ---------------------------------------------------------------------------


def make_eml_reward(
    tree: EMLTree | None = None,
    max_depth: int = 3,
) -> EMLCompositeReward:
    """Factory for an ``EMLCompositeReward``.

    Mirrors the ergonomics of ``make_carl_reward`` but returns the reward
    object directly (not a TRL closure) -- the closure wrapping lives in
    ``composite.make_carl_reward`` which dispatches on ``reward_class``.
    """
    return EMLCompositeReward(tree=tree, max_depth=max_depth)


__all__ = [
    "EMLCompositeReward",
    "eml_reward_from_trace",
    "make_eml_reward",
]
