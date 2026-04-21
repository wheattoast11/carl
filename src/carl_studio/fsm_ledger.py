"""Constitutional FSM + ledger wiring for carl-studio.

Thin orchestration layer over ``carl_core.constitutional``:

- ``FSMState`` snapshots the (C, B_t, chain_head, step) tuple.
- ``ConstitutionalGatePredicate`` implements the ``carl_studio.gating``
  ``GatingPredicate`` protocol so the constitution can veto an action at
  a call-site (e.g. before a tool call, payment, or training step).
- ``evaluate_action`` applies the constitution to an action and, if the
  verdict passes, returns the advanced FSM state.

Defaults keep the module importable without pynacl — the lazy import
only fires when signing/verification is actually attempted.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from carl_core.constitutional import (
    ConstitutionalLedger,
    ConstitutionalPolicy,
    encode_action_features,
)
from carl_core.eml import EMLTree
from carl_core.hashing import content_hash

__all__ = [
    "FSMState",
    "ConstitutionalGatePredicate",
    "evaluate_action",
    "default_behavioral_tree",
    "default_ledger_root",
    "build_default_policy",
]


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def default_ledger_root() -> Path:
    """Default on-disk location for the constitutional ledger."""
    return Path.home() / ".carl" / "constitutional"


# ---------------------------------------------------------------------------
# FSM state snapshot
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FSMState:
    """Snapshot of the constitutional FSM.

    - ``constitution_hash`` — sha256 of the immutable C policy tree.
    - ``behavioral_hash``   — sha256 of the current B_t tree.
    - ``chain_head``        — block_hash of the most recent ledger block.
    - ``step``              — block_id of the head block (0 == genesis).
    """

    constitution_hash: str
    behavioral_hash: str
    chain_head: str
    step: int

    def advanced(self, *, new_behavioral: str, new_head: str, new_step: int) -> FSMState:
        return FSMState(
            constitution_hash=self.constitution_hash,
            behavioral_hash=new_behavioral,
            chain_head=new_head,
            step=new_step,
        )


# ---------------------------------------------------------------------------
# Default trees
# ---------------------------------------------------------------------------


def default_behavioral_tree() -> EMLTree:
    """A depth-1 behavioral tree: exp(x0) - ln(1) = exp(x0).

    Sized to ``FEATURE_DIM`` from the constitutional feature encoder so it
    composes cleanly with the same input vectors used by C.
    """
    # identity on x0 — placeholder; the trainer replaces this with a learned tree.
    t = EMLTree.exp_single()
    # Extend input_dim to match the constitutional feature space.
    return EMLTree(root=t.root, input_dim=25, leaf_params=t.leaf_params.copy())


def build_default_policy(threshold: float = 0.0) -> ConstitutionalPolicy:
    """Build a default constitutional policy:
    ``exp(coherence_phi) - ln(1)`` thresholded at ``threshold``.

    This is a safe starting point: the policy allows any action whose
    coherence_phi feature exceeds ``ln(1) + threshold`` (i.e., non-negative
    by default).
    """
    from carl_core.eml import EMLNode, EMLOp

    # eml(x21, 1) == exp(phi) - ln(1) = exp(phi). Index 21 is coherence_phi.
    root = EMLNode(
        op=EMLOp.EML,
        left=EMLNode(op=EMLOp.VAR_X, var_idx=21),
        right=EMLNode(op=EMLOp.CONST, const=1.0),
    )
    tree = EMLTree(root=root, input_dim=25)
    return ConstitutionalPolicy.create(
        tree=tree,
        threshold=float(threshold),
        metadata={"kind": "default", "notes": "exp(coherence_phi) > tau"},
    )


# ---------------------------------------------------------------------------
# Gating predicate — plugs into carl_studio.gating.Gate
# ---------------------------------------------------------------------------


class ConstitutionalGatePredicate:
    """Gate that consults a constitutional policy before allowing an action.

    Implements the ``GatingPredicate`` protocol. ``feature_encoder`` converts
    the captured context into a feature vector; by default we use the
    standard 25-dim encoder over a dict supplied via ``bind()``.
    """

    def __init__(
        self,
        policy: ConstitutionalPolicy,
        feature_encoder: Callable[[Any], np.ndarray] | None = None,
    ) -> None:
        self._policy = policy
        self._encoder = feature_encoder or encode_action_features
        self._pending_action: dict[str, Any] | None = None

    @property
    def name(self) -> str:
        return f"constitutional:{self._policy.policy_id[:12]}"

    def bind(self, action: dict[str, Any]) -> ConstitutionalGatePredicate:
        """Bind the next action to check. Returns self for chainable use."""
        self._pending_action = dict(action)
        return self

    def check(self) -> tuple[bool, str]:
        if self._pending_action is None:
            return (False, "constitutional predicate called without a bound action")
        features = self._encoder(self._pending_action)
        allowed, score = self._policy.evaluate(features)
        if allowed:
            return (True, f"score={score:.6f} > tau={self._policy.threshold}")
        return (
            False,
            f"constitutional veto: score={score:.6f} <= tau={self._policy.threshold}",
        )


# ---------------------------------------------------------------------------
# evaluate_action — full FSM transition
# ---------------------------------------------------------------------------


def evaluate_action(
    action: dict[str, Any],
    state: FSMState,
    ledger: ConstitutionalLedger,
) -> tuple[bool, float, FSMState | None]:
    """Evaluate ``action`` against the ledger's constitution and, on pass,
    append a block and return the advanced state.

    Returns ``(allowed, score, new_state_or_None)``. On deny, no block is
    appended and ``new_state_or_None`` is ``None``.
    """
    policy = ledger.policy()
    features = encode_action_features(action)
    allowed, score = policy.evaluate(features)
    if not allowed:
        return (False, float(score), None)

    block = ledger.append(action, policy.policy_id)
    # B_t fingerprint = hash of last-N action digests. With N=1 we just use
    # the action digest; this keeps the primitive cheap. Trainers can
    # override by carrying their own B_t tree hash.
    new_behavioral = content_hash({"head": block.action_digest})
    new_state = state.advanced(
        new_behavioral=new_behavioral,
        new_head=block.block_hash(),
        new_step=block.block_id,
    )
    return (True, float(score), new_state)
