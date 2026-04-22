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

**Public / private split (v0.17 moat extraction).**

The depth-3 tree structure, composition rules, and ``score_from_trace``
interface are public (this file). The benchmark-tuned topology + pre-squash
calibration constants + feature-floor threshold — the 7 trained parameters
that yielded the +0.972 correlation with PhaseAdaptive on the 100-trace
benchmark (``scripts/eml_reward_benchmark.md``) — moved to
``resonance.rewards.eml_weights`` for v0.17. When the admin gate unlocks
and the private runtime resolves, :class:`EMLCompositeReward` pulls the
tuned coefficients through :func:`admin.load_private`. Without admin
unlock, it falls through to a PEDAGOGICAL random initialization whose
tree shape + calibration constants are sufficient to run the reward head
end-to-end with bounded output but whose coefficients are **not tuned**
against the paper's benchmark suite.

Simple-reference callers get a working reward head. Competitor
reproductions built on the random-init path will not match the
benchmarked +0.972 correlation — production dynamics live in the private
runtime.
"""

from __future__ import annotations

import math
from typing import Any, cast

import numpy as np

from carl_core.eml import (
    MAX_DEPTH,
    EMLNode,
    EMLOp,
    EMLTree,
    eml_scalar_reward,
)


# ---------------------------------------------------------------------------
# Module constants — public, structural (not tuned against benchmark).
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


# ---------------------------------------------------------------------------
# Pedagogical random-init fallback — coefficients un-tuned.
#
# These preserve the public tree's structural contract (depth 3, 3 trainable
# CONST leaves, input_dim = 3) and keep the reward output bounded so the
# public API delivers a working head out-of-the-box. They are INTENTIONALLY
# un-tuned — callers wanting the benchmark-accurate +0.972 correlation with
# PhaseAdaptive unlock the admin gate to route through
# ``resonance.rewards.eml_weights.initialize_weights()``.
# ---------------------------------------------------------------------------


def _random_init_tree() -> EMLTree:
    """Pedagogical reference tree — topology un-tuned.

    Builds the same depth-3 shape the benchmarked tree uses (so the Adam
    fit-step and serialization round-trip work byte-identical), but with
    CONST leaves at zero and no topology sweep. This is sufficient to
    exercise the tree geometry and let Adam learn from scratch; it is
    NOT the topology the paper's benchmark chose. For production
    dynamics install the private runtime.
    """
    ms_shift = EMLNode(
        op=EMLOp.EML,
        left=EMLNode(op=EMLOp.CONST, const=0.0),
        right=EMLNode(op=EMLOp.VAR_X, var_idx=0),
    )
    cq_shift = EMLNode(
        op=EMLOp.EML,
        left=EMLNode(op=EMLOp.CONST, const=0.0),
        right=EMLNode(op=EMLOp.VAR_X, var_idx=1),
    )
    def_shift = EMLNode(
        op=EMLOp.EML,
        left=EMLNode(op=EMLOp.CONST, const=0.0),
        right=EMLNode(op=EMLOp.VAR_X, var_idx=2),
    )
    inner = EMLNode(op=EMLOp.EML, left=ms_shift, right=cq_shift)
    root = EMLNode(op=EMLOp.EML, left=inner, right=def_shift)
    return EMLTree(root=root, input_dim=_FEATURE_DIM)


def _random_init_weights() -> dict[str, Any]:
    """Pedagogical reference init — coefficients un-tuned.

    Returns the same dict shape as
    :func:`resonance.rewards.eml_weights.initialize_weights` but with
    structurally-correct placeholders instead of benchmark-tuned numbers.
    The calibration constants are set to an identity-ish affine
    ``(scale=1.0, shift=0.0)`` so the raw EML output passes through
    unchanged before the tanh clamp — callers get a bounded output but
    lose the drop-in-CARL alignment property.

    For production dynamics install the private runtime.
    """
    return {
        "tree": _random_init_tree(),
        "raw_to_linear_scale": 1.0,
        "raw_to_linear_shift": 0.0,
        "feature_ln_floor": 0.05,
        "feature_dim": _FEATURE_DIM,
    }


def _load_init_weights() -> dict[str, Any]:
    """Resolve initialization weights via admin gate, falling back to random.

    Delegates to :func:`resonance.rewards.eml_weights.initialize_weights`
    when the admin gate unlocks (returns the benchmark-tuned coefficients).
    Falls through to :func:`_random_init_weights` when the resonance
    runtime is unavailable — the public module still returns a working
    :class:`EMLCompositeReward`, just not one that matches the
    benchmarked +0.972 correlation.
    """
    try:
        from carl_studio.admin import is_admin, load_private

        if is_admin():
            weights_mod = load_private("rewards.eml_weights")
            result = weights_mod.initialize_weights()
            if not isinstance(result, dict):
                return _random_init_weights()
            return cast(dict[str, Any], result)
    except ImportError:
        pass
    return _random_init_weights()


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

    Initialization path:
      * Caller passes an explicit ``tree`` -> use it; calibration still
        pulls from :func:`_load_init_weights` so benchmark-tuned callers
        get tuned squash constants even with a custom tree.
      * Caller leaves ``tree=None`` -> :func:`_load_init_weights` resolves
        the tree AND calibration from the private runtime when admin
        unlocks, else from :func:`_random_init_weights`.
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
        init = _load_init_weights()
        self._raw_to_linear_scale: float = float(init["raw_to_linear_scale"])
        self._raw_to_linear_shift: float = float(init["raw_to_linear_shift"])
        self._feature_ln_floor: float = float(init["feature_ln_floor"])
        self.tree: EMLTree = tree if tree is not None else init["tree"]
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
          1. Affine calibration: linearized = raw_to_linear_scale * raw + shift
             (so the init-time output is a drop-in CARL approximation
             WHEN the private runtime's tuned constants are in use).
          2. Tanh squash into the configured clamp window so the reward can
             never runaway during training even if leaf_params drift.

        The affine constants are populated from
        :func:`_load_init_weights` at construction time — they resolve to
        the benchmark-tuned values from the private runtime when admin
        unlocks, else to identity-like defaults from
        :func:`_random_init_weights`.
        """
        lo, hi = self.clamp
        half = (hi - lo) / 2.0
        mid = (hi + lo) / 2.0
        if not math.isfinite(raw):
            return float(mid)
        if half <= 0:
            return float(mid)
        linearized = self._raw_to_linear_scale * raw + self._raw_to_linear_shift
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
            feats[1] = max(feats[1], self._feature_ln_floor)
        if feats.shape[0] >= 3:
            feats[2] = max(feats[2], self._feature_ln_floor)
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
            feats[1] = max(feats[1], self._feature_ln_floor)
        if feats.shape[0] >= 3:
            feats[2] = max(feats[2], self._feature_ln_floor)
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
            "raw_to_linear_scale": self._raw_to_linear_scale,
            "raw_to_linear_shift": self._raw_to_linear_shift,
            "feature_ln_floor": self._feature_ln_floor,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "EMLCompositeReward":
        """Reconstruct from ``to_dict`` output. Preserves reward output exactly.

        Legacy dicts (pre-v0.17, without ``raw_to_linear_scale`` etc) fall
        through to the current init path — their reward output will match
        only under the runtime that produced them. Current dicts round-trip
        byte-identical regardless of whether the admin gate is unlocked on
        the loading machine.
        """
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
        # Restore calibration constants from the serialized dict when
        # present — this guarantees to_dict/from_dict round-trips to an
        # identical reward output regardless of the host's admin state.
        if "raw_to_linear_scale" in d:
            obj._raw_to_linear_scale = float(d["raw_to_linear_scale"])
        if "raw_to_linear_shift" in d:
            obj._raw_to_linear_shift = float(d["raw_to_linear_shift"])
        if "feature_ln_floor" in d:
            obj._feature_ln_floor = float(d["feature_ln_floor"])
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
