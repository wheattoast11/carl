"""Tests for carl_core.constitutional — public client façade + parity surface.

After v0.17 moat extraction, carl_core.constitutional is a thin client over
``resonance.signals.constitutional`` (admin-gated). Tests split three ways:

1. **Pure-data / wire-format** (unmarked, always run). ``LedgerBlock``,
   ``ConstitutionalPolicy``, ``encode_action_features``, and the
   ``block_hash`` / ``signing_bytes`` parity surface. Shared with
   ``@terminals-tech/emlt-codec`` — must never regress.
2. **Locked-client behavior** (unmarked, always run). Verifies that every
   mutating op raises ``ConstitutionalLedgerError`` with
   ``code='carl.constitutional.private_required'`` when the admin gate
   is unreachable (the default CI state).
3. **Full-lifecycle** (``@pytest.mark.private``). Genesis + append +
   verify round-trips. Skipped unless the resonance runtime is resolvable
   (admin unlock OR the ``resonance`` package on sys.path — Team F dev
   machines). Tests use a monkey-patched admin shim to simulate unlock
   without needing a real admin key.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from carl_core.constitutional import (
    FEATURE_DIM,
    ConstitutionalLedger,
    ConstitutionalLedgerError,
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


def _try_load_resonance() -> bool:
    """Return True if the private resonance.signals.constitutional module
    is resolvable from this environment (either pip-installed or on
    ``sys.path`` — Team F dev machines).
    """
    try:
        import resonance.signals.constitutional  # type: ignore[import-not-found]  # noqa: F401
        return True
    except ImportError:
        return False


_RESONANCE_LOCAL_SRC = (
    Path(__file__).resolve().parents[3] / ".." / "resonance" / "src"
).resolve()


@pytest.fixture
def admin_unlocked(monkeypatch: pytest.MonkeyPatch) -> bool:
    """Simulate an admin-unlocked host by patching ``carl_studio.admin``.

    The fixture prepends the local ``resonance/src`` path (if it exists on
    disk) and replaces ``is_admin`` / ``load_private`` with test stubs.
    Returns ``True`` iff the private module is now importable.
    """
    if _RESONANCE_LOCAL_SRC.is_dir() and str(_RESONANCE_LOCAL_SRC) not in sys.path:
        sys.path.insert(0, str(_RESONANCE_LOCAL_SRC))

    if not _try_load_resonance():
        return False

    from carl_studio import admin as admin_mod

    monkeypatch.setattr(admin_mod, "is_admin", lambda: True)

    def _fake_load_private(name: str) -> Any:
        import importlib

        return importlib.import_module(f"resonance.{name}")

    monkeypatch.setattr(admin_mod, "load_private", _fake_load_private)
    return True


private_required = pytest.mark.private


def _maybe_skip_private(admin_unlocked: bool) -> None:
    """Helper used inside private-marked tests."""
    if not admin_unlocked:
        pytest.skip("resonance private runtime not available")


# ===========================================================================
# 1. Pure-data / wire-format surface — always runs.
# ===========================================================================


def test_feature_encoding_dim_and_onehot() -> None:
    vec = encode_action_features({"type": "PAYMENT", "amount": 12.5, "tier": "PAID"})
    assert vec.shape == (FEATURE_DIM,)
    assert vec[2] == 1.0  # PAYMENT onehot
    assert vec[16] == 12.5
    assert vec[24] == 1.0  # PAID tier

    # Unknown type folds to the OTHER slot (index 15). The 25-dim layout
    # is a published wire-format invariant; hard-coded here as the spec.
    vec2 = encode_action_features({"type": "unknown"})
    assert vec2[15] == 1.0


def test_feature_encoding_consent_flags() -> None:
    vec = encode_action_features(
        {"type": "GATE", "consent_flags": {"telemetry": True, "mcp_share": True}}
    )
    assert vec[17] == 1.0  # telemetry
    assert vec[20] == 1.0  # mcp_share
    assert vec[18] == 0.0  # contract_witnessing
    assert vec[19] == 0.0  # coherence_probe


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


def test_ledger_block_signing_bytes_layout() -> None:
    """Parity-critical: signing_bytes layout must match the TS codec."""
    blk = LedgerBlock(
        block_id=0,
        prev_block_hash="0" * 64,
        policy_id="policy.test",
        action_digest="a" * 64,
        verdict=1.0,
        timestamp_ns=1_700_000_000_000_000_000,
        signer_pubkey=bytes(32),
        signature=bytes(64),
    )
    sb = blk.signing_bytes()
    # prev (64) + '|' + policy_id (11) + '|' + action_digest (64) + '|'
    # + float64 (8) + int64 (8) + pubkey (32) = 190
    assert len(sb) == 190
    assert sb[:64] == b"0" * 64
    assert sb[65:76] == b"policy.test"


def test_ledger_block_hash_is_stable() -> None:
    blk = LedgerBlock(
        block_id=1,
        prev_block_hash="f" * 64,
        policy_id="p",
        action_digest="d" * 64,
        verdict=0.5,
        timestamp_ns=42,
        signer_pubkey=b"\x00" * 32,
        signature=b"\x01" * 64,
    )
    h1 = blk.block_hash()
    h2 = blk.block_hash()
    assert h1 == h2
    assert len(h1) == 64


def test_ledger_block_dict_roundtrip() -> None:
    blk = LedgerBlock(
        block_id=7,
        prev_block_hash="a" * 64,
        policy_id="pol",
        action_digest="b" * 64,
        verdict=2.5,
        timestamp_ns=9_999,
        signer_pubkey=b"\x22" * 32,
        signature=b"\x33" * 64,
    )
    restored = LedgerBlock.from_dict(blk.to_dict())
    assert restored.block_hash() == blk.block_hash()


# ===========================================================================
# 2. Locked-client behavior — always runs.
# ===========================================================================


def test_client_genesis_requires_private(tmp_path: Path) -> None:
    """Without admin unlock, genesis must raise ConstitutionalLedgerError."""
    ledger = ConstitutionalLedger(root=tmp_path / "ledger", signing_key=_fixed_key())
    with pytest.raises(ConstitutionalLedgerError) as excinfo:
        ledger.genesis(_policy())
    assert excinfo.value.code == "carl.constitutional.private_required"
    assert "private resonance runtime" in str(excinfo.value).lower() or "admin" in str(
        excinfo.value
    ).lower()


def test_client_append_requires_private(tmp_path: Path) -> None:
    ledger = ConstitutionalLedger(root=tmp_path / "ledger")
    with pytest.raises(ConstitutionalLedgerError) as excinfo:
        ledger.append({"type": "GATE"}, policy_id="fake")
    assert excinfo.value.code == "carl.constitutional.private_required"


def test_client_verify_chain_requires_private(tmp_path: Path) -> None:
    ledger = ConstitutionalLedger(root=tmp_path / "ledger")
    with pytest.raises(ConstitutionalLedgerError) as excinfo:
        ledger.verify_chain()
    assert excinfo.value.code == "carl.constitutional.private_required"


def test_client_read_only_ops_work_locally(tmp_path: Path) -> None:
    """head/replay/height/_load_blocks stay local; no admin gate needed."""
    root = tmp_path / "ledger"
    ledger = ConstitutionalLedger(root=root)
    assert ledger.head() is None
    assert ledger.height() == 0
    assert ledger.replay() == []


def test_client_read_only_replay_existing_chain(tmp_path: Path) -> None:
    """Verify read-only ops against a chain.jsonl written by a trusted host."""
    root = tmp_path / "ledger"
    root.mkdir()
    blk = LedgerBlock(
        block_id=0,
        prev_block_hash="0" * 64,
        policy_id="p0",
        action_digest="a" * 64,
        verdict=0.0,
        timestamp_ns=0,
        signer_pubkey=b"\x00" * 32,
        signature=b"\x00" * 64,
    )
    (root / "chain.jsonl").write_text(
        json.dumps(blk.to_dict(), sort_keys=True) + "\n"
    )
    ledger = ConstitutionalLedger(root=root)
    assert ledger.height() == 1
    head = ledger.head()
    assert head is not None
    assert head.block_id == 0
    replayed = ledger.replay()
    assert len(replayed) == 1
    assert replayed[0].block_hash() == blk.block_hash()


def test_client_policy_read_without_gate(tmp_path: Path) -> None:
    """``policy()`` is read-only; works without admin unlock."""
    root = tmp_path / "ledger"
    root.mkdir()
    pol = _policy(threshold=0.25)
    pol.save(root / "policy.json")
    ledger = ConstitutionalLedger(root=root)
    loaded = ledger.policy()
    assert loaded.policy_id == pol.policy_id


def test_client_policy_missing_raises_clear_error(tmp_path: Path) -> None:
    ledger = ConstitutionalLedger(root=tmp_path / "nope")
    with pytest.raises(Exception) as excinfo:
        ledger.policy()
    assert "no policy" in str(excinfo.value).lower() or "genesis" in str(
        excinfo.value
    ).lower()


def test_constitutional_error_code_is_stable() -> None:
    """The code is a public contract — downstream tooling branches on it."""
    assert ConstitutionalLedgerError.code == "carl.constitutional.private_required"


# ===========================================================================
# 3. Full lifecycle — requires private runtime (``@pytest.mark.private``).
# ===========================================================================


@private_required
def test_ledger_genesis_roundtrip(tmp_path: Path, admin_unlocked: bool) -> None:
    _maybe_skip_private(admin_unlocked)
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


@private_required
def test_ledger_append_chain_of_100(
    tmp_path: Path, admin_unlocked: bool
) -> None:
    _maybe_skip_private(admin_unlocked)
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


@private_required
def test_ledger_tampering_detected(
    tmp_path: Path, admin_unlocked: bool
) -> None:
    _maybe_skip_private(admin_unlocked)
    ledger = ConstitutionalLedger(root=tmp_path / "ledger", signing_key=_fixed_key())
    pol = _policy()
    ledger.genesis(pol)
    for _ in range(99):
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


@private_required
def test_ledger_reject_wrong_policy_id(
    tmp_path: Path, admin_unlocked: bool
) -> None:
    _maybe_skip_private(admin_unlocked)
    ledger = ConstitutionalLedger(root=tmp_path / "ledger", signing_key=_fixed_key())
    pol = _policy()
    ledger.genesis(pol)
    with pytest.raises(Exception) as excinfo:
        ledger.append({"type": "GATE"}, policy_id="deadbeef")
    assert "policy_id" in str(excinfo.value).lower()


@private_required
def test_ledger_double_genesis_blocked(
    tmp_path: Path, admin_unlocked: bool
) -> None:
    _maybe_skip_private(admin_unlocked)
    ledger = ConstitutionalLedger(root=tmp_path / "ledger", signing_key=_fixed_key())
    pol = _policy()
    ledger.genesis(pol)
    with pytest.raises(Exception):
        ledger.genesis(pol)


@private_required
def test_ledger_replay(tmp_path: Path, admin_unlocked: bool) -> None:
    _maybe_skip_private(admin_unlocked)
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


@private_required
def test_block_dict_roundtrip_with_signature(
    tmp_path: Path, admin_unlocked: bool
) -> None:
    _maybe_skip_private(admin_unlocked)
    ledger = ConstitutionalLedger(root=tmp_path / "ledger", signing_key=_fixed_key())
    pol = _policy()
    block = ledger.genesis(pol)
    restored = LedgerBlock.from_dict(block.to_dict())
    assert restored.block_hash() == block.block_hash()
    assert restored.verify(block.signer_pubkey)


# ---------------------------------------------------------------------------
# Lazy import behavior (pynacl absence) — still works against private impl.
# ---------------------------------------------------------------------------


@private_required
def test_missing_pynacl_raises_clear_import_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    admin_unlocked: bool,
) -> None:
    """If pynacl is absent in the private runtime, signing paths raise a
    clear ImportError with a pip install hint.
    """
    _maybe_skip_private(admin_unlocked)
    # Patch the private module's _lazy_nacl so ledger.genesis surfaces
    # the ImportError. Import via the admin-gate seam so we hit the
    # same module the client will bind to.
    from carl_studio import admin as admin_mod

    mod = admin_mod.load_private("signals.constitutional")

    def _broken() -> None:
        raise mod._missing_pynacl()

    monkeypatch.setattr(mod, "_lazy_nacl", _broken)
    ledger = ConstitutionalLedger(root=tmp_path / "no-nacl")
    with pytest.raises(ImportError) as excinfo:
        ledger.genesis(_policy())
    assert "pynacl" in str(excinfo.value).lower()
    assert "install" in str(excinfo.value).lower()
