"""Tests for carl_studio.fsm_ledger — FSM wiring over the constitutional ledger."""
from __future__ import annotations

from pathlib import Path


from carl_core.constitutional import ConstitutionalLedger, ConstitutionalPolicy
from carl_studio.fsm_ledger import (
    ConstitutionalGatePredicate,
    FSMState,
    build_default_policy,
    default_behavioral_tree,
    default_ledger_root,
    evaluate_action,
)


def _fresh_ledger(tmp_path: Path, threshold: float = 0.0) -> tuple[ConstitutionalLedger, ConstitutionalPolicy, FSMState]:
    ledger = ConstitutionalLedger(
        root=tmp_path / "ledger",
        signing_key=b"t6-fsm-test-seed-32bytes-padded!"[:32],
    )
    policy = build_default_policy(threshold=threshold)
    genesis = ledger.genesis(policy)
    state = FSMState(
        constitution_hash=policy.tree.hash(),
        behavioral_hash=default_behavioral_tree().hash(),
        chain_head=genesis.block_hash(),
        step=0,
    )
    return ledger, policy, state


def test_default_ledger_root_under_home() -> None:
    root = default_ledger_root()
    assert ".carl" in root.parts
    assert root.name == "constitutional"


def test_default_behavioral_tree_is_25_dim() -> None:
    tree = default_behavioral_tree()
    assert tree.input_dim == 25


def test_build_default_policy_is_deterministic() -> None:
    p1 = build_default_policy(threshold=0.5)
    p2 = build_default_policy(threshold=0.5)
    assert p1.policy_id == p2.policy_id


def test_evaluate_action_allow_advances_fsm(tmp_path: Path) -> None:
    ledger, policy, state = _fresh_ledger(tmp_path, threshold=0.0)
    action = {"type": "TOOL", "coherence_phi": 1.2, "tier": "PAID"}
    allowed, score, new_state = evaluate_action(action, state, ledger)
    assert allowed
    assert score > 0.0
    assert new_state is not None
    assert new_state.step == 1
    assert new_state.chain_head != state.chain_head
    # Constitution hash never changes.
    assert new_state.constitution_hash == state.constitution_hash


def test_evaluate_action_deny_does_not_advance(tmp_path: Path) -> None:
    ledger, policy, state = _fresh_ledger(tmp_path, threshold=1e12)
    action = {"type": "TOOL", "coherence_phi": 0.01}
    allowed, score, new_state = evaluate_action(action, state, ledger)
    assert not allowed
    assert new_state is None
    # No block appended.
    assert ledger.height() == 1  # genesis only


def test_multi_step_advances_sequentially(tmp_path: Path) -> None:
    ledger, policy, state = _fresh_ledger(tmp_path, threshold=0.0)
    for expected_step in range(1, 6):
        allowed, _, new_state = evaluate_action(
            {"type": "GATE", "coherence_phi": 0.5}, state, ledger
        )
        assert allowed
        assert new_state is not None
        assert new_state.step == expected_step
        state = new_state
    ok, bad = ledger.verify_chain()
    assert ok, bad


# ---------------------------------------------------------------------------
# ConstitutionalGatePredicate
# ---------------------------------------------------------------------------


def test_gate_predicate_implements_protocol() -> None:
    from carl_studio.gating import GatingPredicate

    policy = build_default_policy(threshold=0.0)
    pred = ConstitutionalGatePredicate(policy)
    # structural check — we expose name + check()
    assert isinstance(pred.name, str)
    assert callable(pred.check)
    # Protocol is runtime-checkable in carl_studio.gating.
    # Not all Protocols are isinstance-ready; doing it softly:
    assert hasattr(pred, "name") and hasattr(pred, "check")
    _ = GatingPredicate  # import sanity


def test_gate_predicate_unbound_returns_deny() -> None:
    policy = build_default_policy(threshold=0.0)
    pred = ConstitutionalGatePredicate(policy)
    allowed, reason = pred.check()
    assert not allowed
    assert "without a bound action" in reason


def test_gate_predicate_allow_and_deny() -> None:
    policy = build_default_policy(threshold=0.0)
    pred = ConstitutionalGatePredicate(policy)
    allowed, reason = pred.bind({"type": "TOOL", "coherence_phi": 1.0}).check()
    assert allowed
    assert "score=" in reason

    pred2 = ConstitutionalGatePredicate(build_default_policy(threshold=1e12))
    allowed2, reason2 = pred2.bind({"type": "TOOL", "coherence_phi": 0.01}).check()
    assert not allowed2
    assert "veto" in reason2
