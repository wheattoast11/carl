"""Tests for carl_core.constitutional — EML-gated, hash-chained, signed ledger."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from carl_core.constitutional import (
    FEATURE_DIM,
    ConstitutionalLedger,
    ConstitutionalPolicy,
    LedgerBlock,
    encode_action_features,
)
from carl_core.eml import EMLNode, EMLOp, EMLTree


# ---------------------------------------------------------------------------
# Fixtures + helpers
# ---------------------------------------------------------------------------


def _policy(threshold: float = 0.0) -> ConstitutionalPolicy:
    """Build a small deterministic policy: exp(x21) - ln(1), thresholded."""
    root = EMLNode(
        op=EMLOp.EML,
        left=EMLNode(op=EMLOp.VAR_X, var_idx=21),  # coherence_phi slot
        right=EMLNode(op=EMLOp.CONST, const=1.0),
    )
    tree = EMLTree(root=root, input_dim=FEATURE_DIM)
    return ConstitutionalPolicy.create(tree=tree, threshold=threshold)


def _fixed_key() -> bytes:
    # Deterministic 32-byte seed so signatures in tests don't vary.
    return b"t6-constitution-test-seed-32byts"[:32]


# ---------------------------------------------------------------------------
# encode_action_features
# ---------------------------------------------------------------------------


def test_feature_encoding_dim_and_onehot() -> None:
    vec = encode_action_features({"type": "PAYMENT", "amount": 12.5, "tier": "PAID"})
    assert vec.shape == (FEATURE_DIM,)
    assert vec[2] == 1.0  # PAYMENT onehot
    assert vec[16] == 12.5
    assert vec[24] == 1.0  # PAID tier

    vec2 = encode_action_features({"type": "unknown"})
    assert vec2[_ACTION_OTHER_IDX()] == 1.0


def _ACTION_OTHER_IDX() -> int:
    from carl_core.constitutional import _ACTION_TYPES

    return _ACTION_TYPES.index("OTHER")


def test_feature_encoding_consent_flags() -> None:
    vec = encode_action_features(
        {"type": "GATE", "consent_flags": {"telemetry": True, "mcp_share": True}}
    )
    assert vec[17] == 1.0  # telemetry
    assert vec[20] == 1.0  # mcp_share
    assert vec[18] == 0.0  # contract_witnessing
    assert vec[19] == 0.0  # coherence_probe


# ---------------------------------------------------------------------------
# ConstitutionalPolicy
# ---------------------------------------------------------------------------


def test_policy_id_is_stable() -> None:
    p1 = _policy(threshold=0.5)
    p2 = _policy(threshold=0.5)
    assert p1.policy_id == p2.policy_id
    p3 = _policy(threshold=0.6)
    assert p1.policy_id != p3.policy_id


def test_policy_evaluate_known_values() -> None:
    """exp(1.0) - ln(1) = e ≈ 2.718 — well above threshold 0."""
    pol = _policy(threshold=0.0)
    feats = np.zeros(FEATURE_DIM)
    feats[21] = 1.0  # coherence_phi = 1.0
    allowed, score = pol.evaluate(feats)
    assert allowed
    assert abs(score - np.e) < 1e-9


def test_policy_evaluate_deny_below_threshold() -> None:
    pol = _policy(threshold=100.0)  # impossibly high
    feats = np.zeros(FEATURE_DIM)
    feats[21] = 1.0
    allowed, score = pol.evaluate(feats)
    assert not allowed
    assert score < 100.0


def test_policy_roundtrip_save_load(tmp_path: Path) -> None:
    pol = _policy(threshold=0.25)
    path = tmp_path / "policy.json"
    pol.save(path)
    loaded = ConstitutionalPolicy.load(path)
    assert loaded.policy_id == pol.policy_id
    assert loaded.threshold == pol.threshold
    assert loaded.tree.hash() == pol.tree.hash()


# ---------------------------------------------------------------------------
# LedgerBlock signing + verification
# ---------------------------------------------------------------------------


def test_ledger_genesis_roundtrip(tmp_path: Path) -> None:
    ledger = ConstitutionalLedger(root=tmp_path / "ledger", signing_key=_fixed_key())
    pol = _policy()
    block = ledger.genesis(pol)
    assert block.block_id == 0
    assert block.prev_block_hash == "0" * 64
    assert block.verify(block.signer_pubkey)
    # Reload from disk with a fresh ledger object pointing at same root.
    reloaded = ConstitutionalLedger(root=tmp_path / "ledger")
    head = reloaded.head()
    assert head is not None
    assert head.block_hash() == block.block_hash()
    # Policy is still there.
    assert reloaded.policy().policy_id == pol.policy_id


def test_ledger_append_chain_of_100(tmp_path: Path) -> None:
    ledger = ConstitutionalLedger(root=tmp_path / "ledger", signing_key=_fixed_key())
    pol = _policy()
    ledger.genesis(pol)
    for i in range(100):
        ledger.append(
            {"type": "GATE", "coherence_phi": 0.5 + (i % 5) * 0.1},
            pol.policy_id,
        )
    ok, bad = ledger.verify_chain()
    assert ok, f"bad blocks: {bad}"
    assert ledger.height() == 101  # genesis + 100


def test_ledger_tampering_detected(tmp_path: Path) -> None:
    ledger = ConstitutionalLedger(root=tmp_path / "ledger", signing_key=_fixed_key())
    pol = _policy()
    ledger.genesis(pol)
    for i in range(99):
        ledger.append({"type": "GATE", "coherence_phi": 1.0}, pol.policy_id)

    # Flip one byte inside block 50's action_digest hex string (preserves
    # valid hex so JSON reload still works).
    chain_path = tmp_path / "ledger" / "chain.jsonl"
    lines = chain_path.read_text().splitlines()
    record = json.loads(lines[50])
    orig = record["action_digest"]
    flipped = ("f" if orig[0] != "f" else "e") + orig[1:]
    record["action_digest"] = flipped
    lines[50] = json.dumps(record, sort_keys=True)
    chain_path.write_text("\n".join(lines) + "\n")

    # Rebuild ledger to force reload of blocks.
    fresh = ConstitutionalLedger(root=tmp_path / "ledger", signing_key=_fixed_key())
    ok, bad = fresh.verify_chain()
    assert not ok
    assert 50 in bad


def test_ledger_reject_wrong_policy_id(tmp_path: Path) -> None:
    ledger = ConstitutionalLedger(root=tmp_path / "ledger", signing_key=_fixed_key())
    pol = _policy()
    ledger.genesis(pol)
    with pytest.raises(Exception) as excinfo:
        ledger.append({"type": "GATE"}, policy_id="deadbeef")
    assert "policy_id" in str(excinfo.value).lower()


def test_ledger_double_genesis_blocked(tmp_path: Path) -> None:
    ledger = ConstitutionalLedger(root=tmp_path / "ledger", signing_key=_fixed_key())
    pol = _policy()
    ledger.genesis(pol)
    with pytest.raises(Exception):
        ledger.genesis(pol)


def test_ledger_replay(tmp_path: Path) -> None:
    ledger = ConstitutionalLedger(root=tmp_path / "ledger", signing_key=_fixed_key())
    pol = _policy()
    ledger.genesis(pol)
    for _ in range(5):
        ledger.append({"type": "GATE", "coherence_phi": 1.0}, pol.policy_id)
    all_blocks = ledger.replay()
    assert len(all_blocks) == 6
    partial = ledger.replay(until_block=3)
    assert len(partial) == 4
    assert partial[-1].block_id == 3


def test_block_dict_roundtrip(tmp_path: Path) -> None:
    ledger = ConstitutionalLedger(root=tmp_path / "ledger", signing_key=_fixed_key())
    pol = _policy()
    block = ledger.genesis(pol)
    restored = LedgerBlock.from_dict(block.to_dict())
    assert restored.block_hash() == block.block_hash()
    assert restored.verify(block.signer_pubkey)


# ---------------------------------------------------------------------------
# Lazy import behavior
# ---------------------------------------------------------------------------


def test_missing_pynacl_raises_clear_import_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """If pynacl is absent, signing paths raise ImportError with a hint."""
    # Simulate absence by making the lazy loader raise.
    from carl_core import constitutional as mod

    def _broken() -> None:
        raise mod._missing_pynacl()

    monkeypatch.setattr(mod, "_lazy_nacl", _broken)
    ledger = ConstitutionalLedger(root=tmp_path / "no-nacl")
    with pytest.raises(ImportError) as excinfo:
        ledger.genesis(_policy())
    assert "pynacl" in str(excinfo.value).lower()
    assert "install" in str(excinfo.value).lower()
