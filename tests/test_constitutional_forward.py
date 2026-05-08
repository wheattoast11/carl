"""Tests for carl_studio.fsm_ledger_forward — Phase H-S4a.

Six core cases + a rotation test:

* consent-off short-circuits the HTTP forward (still persists locally)
* every forward attempt always appends to the local JSONL
* the local hash-chain integrity is preserved across multiple forwards
* :meth:`ConstitutionalForwarder.replay_pending` retries unacked entries
* the block signature is unchanged in transit (no re-signing)
* JSONL rotation triggers at :data:`ROTATION_BYTES`

The tests construct :class:`LedgerBlock` instances directly (no
ed25519 signing) so we don't need ``pynacl`` to exercise the
forwarder's persistence + transport layer. The block's ``signature``
field is opaque bytes from the forwarder's perspective.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest

from carl_core.constitutional import LedgerBlock
from carl_studio.fsm_ledger_forward import (
    APPEND_PATH,
    ConstitutionalForwarder,
    DEFAULT_CARL_CAMP_BASE,
    ROTATION_BYTES,
    reset_default_forwarder,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_singleton() -> Any:  # pyright: ignore[reportUnusedFunction]
    yield
    reset_default_forwarder()


def _make_block(
    *,
    block_id: int = 1,
    prev_hash: str = "0" * 64,
    policy_id: str = "p" * 64,
    action_digest: str = "a" * 64,
    verdict: float = 1.5,
    timestamp_ns: int = 1_700_000_000_000_000_000,
    signer_pubkey: bytes = b"\x01" * 32,
    signature: bytes = b"\x02" * 64,
) -> LedgerBlock:
    """Build a synthetic LedgerBlock for transport tests.

    The signature is opaque bytes; ed25519 is not exercised here —
    we only test the forwarder's persist + ship + replay behavior.
    """
    return LedgerBlock(
        block_id=block_id,
        prev_block_hash=prev_hash,
        policy_id=policy_id,
        action_digest=action_digest,
        verdict=verdict,
        timestamp_ns=timestamp_ns,
        signer_pubkey=signer_pubkey,
        signature=signature,
    )


def _make_fwd(
    tmp_path: Path,
    *,
    consent_on: bool = True,
    bearer: str | None = "test-jwt",
    http: Any = None,
) -> ConstitutionalForwarder:
    """Test-friendly forwarder rooted at *tmp_path*."""
    return ConstitutionalForwarder(
        base_url="https://carl.test",
        ledger_path=tmp_path / "constitutional_ledger.jsonl",
        consent_check=lambda: consent_on,
        bearer_token_resolver=lambda: bearer,
        http=http or _ok_client(),
    )


def _ok_client(*, status: int = 200, json_body: dict[str, Any] | None = None) -> Any:
    """An httpx.Client mock that returns 200 + JSON on POST."""
    mock = MagicMock(spec=httpx.Client)
    response = MagicMock(spec=httpx.Response)
    response.status_code = status
    response.headers = {"content-type": "application/json"}
    response.json.return_value = json_body or {"ok": True, "block_id": 1}
    response.text = ""
    mock.post.return_value = response
    return mock


# ---------------------------------------------------------------------------
# 1. Consent-off → no forward, still persists locally
# ---------------------------------------------------------------------------


def test_forward_off_when_consent_off(tmp_path: Path) -> None:
    http = _ok_client()
    fwd = _make_fwd(tmp_path, consent_on=False, http=http)
    result = fwd.forward_block(_make_block())

    assert result["persisted"] is True
    assert result["forwarded"] is False
    assert result["reason"] == "consent_off"
    assert result["response"] is None

    # No HTTP call when consent is off.
    http.post.assert_not_called()

    # JSONL still wrote a record.
    assert fwd.ledger_path.exists()
    lines = fwd.ledger_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["acked"] is False
    assert rec["signature_hex"] == (b"\x02" * 64).hex()


def test_forward_off_when_no_bearer(tmp_path: Path) -> None:
    http = _ok_client()
    fwd = _make_fwd(tmp_path, consent_on=True, bearer=None, http=http)
    result = fwd.forward_block(_make_block())

    assert result["persisted"] is True
    assert result["forwarded"] is False
    assert result["reason"] == "no_bearer"
    http.post.assert_not_called()


# ---------------------------------------------------------------------------
# 2. Local persist always happens (every forward path appends to JSONL)
# ---------------------------------------------------------------------------


def test_forward_appends_local_jsonl_always(tmp_path: Path) -> None:
    """Each of the consent-off / no-bearer / happy / network-fail paths
    must append to the JSONL exactly once per forward_block call."""

    # (a) consent off → 1 line
    fwd1 = _make_fwd(tmp_path / "a", consent_on=False)
    fwd1.forward_block(_make_block(block_id=1))
    assert _line_count(fwd1.ledger_path) == 1

    # (b) no bearer → 1 line
    fwd2 = _make_fwd(tmp_path / "b", consent_on=True, bearer=None)
    fwd2.forward_block(_make_block(block_id=1))
    assert _line_count(fwd2.ledger_path) == 1

    # (c) happy 200 → 1 line, marked acked after rewrite
    http_ok = _ok_client()
    fwd3 = _make_fwd(tmp_path / "c", http=http_ok)
    fwd3.forward_block(_make_block(block_id=1))
    assert _line_count(fwd3.ledger_path) == 1
    rec = json.loads(fwd3.ledger_path.read_text().splitlines()[0])
    assert rec["acked"] is True

    # (d) network failure → 1 line, still not acked
    http_fail = MagicMock(spec=httpx.Client)
    http_fail.post.side_effect = httpx.ConnectError("transport down")
    fwd4 = _make_fwd(tmp_path / "d", http=http_fail)
    result = fwd4.forward_block(_make_block(block_id=1))
    assert result["forwarded"] is False
    assert result["reason"].startswith("network:")
    assert _line_count(fwd4.ledger_path) == 1
    rec = json.loads(fwd4.ledger_path.read_text().splitlines()[0])
    assert rec["acked"] is False

    # (e) http 500 → 1 line, still not acked
    http_500 = _ok_client(status=500)
    fwd5 = _make_fwd(tmp_path / "e", http=http_500)
    result = fwd5.forward_block(_make_block(block_id=1))
    assert result["forwarded"] is False
    assert result["reason"].startswith("http_500")
    rec = json.loads(fwd5.ledger_path.read_text().splitlines()[0])
    assert rec["acked"] is False


# ---------------------------------------------------------------------------
# 3. Hash-chain integrity preserved (sequence of blocks land in JSONL order)
# ---------------------------------------------------------------------------


def test_hash_chain_integrity_preserved(tmp_path: Path) -> None:
    """The local JSONL must record blocks in exactly the order they
    arrive, with prev_block_hash links intact across forward calls."""
    fwd = _make_fwd(tmp_path)

    blocks = [
        _make_block(
            block_id=i,
            prev_hash=("0" * 64) if i == 0 else f"hash-{i - 1:03d}".ljust(64, "x"),
            action_digest=f"act-{i:03d}".ljust(64, "x"),
            signature=bytes([i]) + b"\x00" * 63,
        )
        for i in range(5)
    ]
    for b in blocks:
        result = fwd.forward_block(b)
        assert result["persisted"] is True

    # Read back: block_ids in order, prev_block_hashes form a chain.
    records = [
        json.loads(line)
        for line in fwd.ledger_path.read_text().splitlines()
    ]
    assert len(records) == len(blocks)
    for i, rec in enumerate(records):
        assert rec["block"]["block_id"] == i
        if i == 0:
            assert rec["block"]["prev_block_hash"] == "0" * 64
        else:
            assert rec["block"]["prev_block_hash"] == f"hash-{i - 1:03d}".ljust(64, "x")
        # signature_hex round-trips (no re-signing).
        assert rec["signature_hex"] == blocks[i].signature.hex()


# ---------------------------------------------------------------------------
# 4. replay_pending retries non-acked entries; doesn't double-ack
# ---------------------------------------------------------------------------


def test_replay_pending_retries_unacked(tmp_path: Path) -> None:
    # First pass: the network fails so nothing gets acked.
    http_fail = MagicMock(spec=httpx.Client)
    http_fail.post.side_effect = httpx.ConnectError("down")
    fwd = _make_fwd(tmp_path, http=http_fail)
    for i in range(3):
        fwd.forward_block(_make_block(block_id=i, signature=bytes([i + 1]) * 64))

    # All 3 records persisted, none acked.
    records = [json.loads(line) for line in fwd.ledger_path.read_text().splitlines()]
    assert len(records) == 3
    assert all(r["acked"] is False for r in records)

    # Second pass: swap in a healthy client, replay.
    fwd._http = _ok_client()  # pyright: ignore[reportPrivateUsage]
    n_acked = fwd.replay_pending()
    assert n_acked == 3

    # All 3 should now be acked.
    records = [json.loads(line) for line in fwd.ledger_path.read_text().splitlines()]
    assert all(r["acked"] is True for r in records)

    # Third pass: re-run replay with a counting client — should NOT
    # re-POST already-acked entries.
    counting = _ok_client()
    fwd._http = counting  # pyright: ignore[reportPrivateUsage]
    n_acked_2 = fwd.replay_pending()
    assert n_acked_2 == 0
    counting.post.assert_not_called()


def test_replay_pending_skips_when_consent_off(tmp_path: Path) -> None:
    fwd = _make_fwd(tmp_path, consent_on=False)
    fwd.forward_block(_make_block())
    # Replay path also gates on consent — it just returns 0.
    assert fwd.replay_pending() == 0


def test_replay_pending_skips_when_no_bearer(tmp_path: Path) -> None:
    http_fail = MagicMock(spec=httpx.Client)
    http_fail.post.side_effect = httpx.ConnectError("down")
    # First create with bearer to seed unacked records.
    fwd = _make_fwd(tmp_path, http=http_fail)
    fwd.forward_block(_make_block())
    # Now flip resolver to None — replay should bail at the bearer gate.
    fwd._bearer_resolver = lambda: None  # pyright: ignore[reportPrivateUsage]
    assert fwd.replay_pending() == 0


# ---------------------------------------------------------------------------
# 5. Block signature unchanged in transit (no re-signing)
# ---------------------------------------------------------------------------


def test_block_signature_unchanged_in_transit(tmp_path: Path) -> None:
    sig_bytes = b"\xde\xad\xbe\xef" * 16  # 64 bytes, distinctive
    block = _make_block(signature=sig_bytes)
    http = _ok_client()
    fwd = _make_fwd(tmp_path, http=http)
    fwd.forward_block(block)

    # Inspect the POST body: signature_hex must equal the input.
    call_args = http.post.call_args
    posted = call_args.kwargs["json"]
    assert posted["signature_hex"] == sig_bytes.hex()
    # Plus the embedded block dict carries the same signature_hex.
    assert posted["block"]["signature_hex"] == sig_bytes.hex()
    # And the local JSONL also captured it unmodified.
    rec = json.loads(fwd.ledger_path.read_text().splitlines()[0])
    assert rec["signature_hex"] == sig_bytes.hex()


def test_post_url_uses_append_path(tmp_path: Path) -> None:
    """Belt-and-suspenders: the POST URL is exactly base_url + APPEND_PATH."""
    http = _ok_client()
    fwd = _make_fwd(tmp_path, http=http)
    fwd.forward_block(_make_block())

    call_args = http.post.call_args
    url = call_args.args[0] if call_args.args else call_args.kwargs.get("url")
    assert url == f"https://carl.test{APPEND_PATH}"


# ---------------------------------------------------------------------------
# 6. JSONL rotation at 10MB
# ---------------------------------------------------------------------------


def test_jsonl_rotates_at_10mb(tmp_path: Path) -> None:
    """When the live JSONL hits ROTATION_BYTES, the next append must
    rotate it to ``.1`` and start a fresh active file."""
    ledger_path = tmp_path / "constitutional_ledger.jsonl"
    fwd = ConstitutionalForwarder(
        base_url="https://carl.test",
        ledger_path=ledger_path,
        consent_check=lambda: False,  # skip HTTP for speed
        bearer_token_resolver=lambda: None,
    )

    # Pre-seed the live file just past ROTATION_BYTES so the next
    # append will trigger rotation. We write a fake JSONL line; it
    # doesn't need to be a real LedgerBlock for the rotation logic.
    payload_size = ROTATION_BYTES + 1024
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_path.write_bytes(b"x" * payload_size)
    assert ledger_path.stat().st_size >= ROTATION_BYTES

    # Now forward — local append should rotate, then write 1 line.
    fwd.forward_block(_make_block(block_id=42))

    # .1 archive exists with the original bulk payload.
    archive = ledger_path.with_name(f"{ledger_path.name}.1")
    assert archive.exists()
    assert archive.stat().st_size >= ROTATION_BYTES

    # Live file shrunk to just the new line.
    assert ledger_path.stat().st_size < ROTATION_BYTES
    new_lines = ledger_path.read_text(encoding="utf-8").splitlines()
    assert len(new_lines) == 1
    rec = json.loads(new_lines[0])
    assert rec["block"]["block_id"] == 42


def test_rotation_caps_at_max_rotations(tmp_path: Path) -> None:
    """After MAX_ROTATIONS rotations, the oldest archive is unlinked."""
    ledger_path = tmp_path / "constitutional_ledger.jsonl"
    fwd = ConstitutionalForwarder(
        base_url="https://carl.test",
        ledger_path=ledger_path,
        consent_check=lambda: False,
        bearer_token_resolver=lambda: None,
    )

    # Pre-create .1 / .2 / .3 with distinctive content so we can
    # verify which slot survives.
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    for i in (1, 2, 3):
        ledger_path.with_name(f"{ledger_path.name}.{i}").write_text(f"slot-{i}")

    # Pre-seed live file just past the rotation ceiling.
    ledger_path.write_bytes(b"x" * (ROTATION_BYTES + 100))

    # Trigger one rotation by appending.
    fwd.forward_block(_make_block(block_id=99))

    # After rotation, the live -> .1 promotion happened, so we expect:
    #   .1 = the previous live file (large)
    #   .2 = the previous .1 ("slot-1")
    #   .3 = the previous .2 ("slot-2")
    #   .4 does NOT exist (capped at MAX_ROTATIONS=3)
    #   the previous .3 ("slot-3") was unlinked.
    archive_1 = ledger_path.with_name(f"{ledger_path.name}.1")
    archive_2 = ledger_path.with_name(f"{ledger_path.name}.2")
    archive_3 = ledger_path.with_name(f"{ledger_path.name}.3")
    archive_4 = ledger_path.with_name(f"{ledger_path.name}.4")

    assert archive_1.exists()
    assert archive_1.stat().st_size > ROTATION_BYTES  # carries the previous live payload
    assert archive_2.read_text() == "slot-1"
    assert archive_3.read_text() == "slot-2"
    assert not archive_4.exists()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _line_count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(
        1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    )


# ---------------------------------------------------------------------------
# Smoke: integration with evaluate_action — forwarder is invoked
# ---------------------------------------------------------------------------


def test_evaluate_action_invokes_forward_when_passed(tmp_path: Path) -> None:
    """evaluate_action(... forward=fwd) must call forward.forward_block
    exactly once for an allowed action — and never propagate forward errors.

    We mock both the ledger and the forwarder so the test doesn't depend
    on the private resonance runtime (the real ConstitutionalLedger.append
    path is gated). The integration point is just: 'evaluate_action
    routes the appended block through forward.forward_block on success.'
    """
    from carl_core.constitutional import ConstitutionalPolicy
    from carl_core.eml import EMLNode, EMLOp, EMLTree
    from carl_studio.fsm_ledger import FSMState, evaluate_action

    # Build a permissive constitutional policy (constant 1.0 > 0.0).
    tree = EMLTree(
        root=EMLNode(
            op=EMLOp.EML,
            left=EMLNode(op=EMLOp.CONST, const=2.0),
            right=EMLNode(op=EMLOp.CONST, const=1.0),
        ),
        input_dim=25,
    )
    policy = ConstitutionalPolicy.create(tree=tree, threshold=0.0)

    fake_block = _make_block(block_id=7, signature=b"\x07" * 64)

    # Mock ledger: policy() returns our policy, append() returns the
    # fake block, _load_blocks/_chain_path/etc unused.
    mock_ledger = MagicMock()
    mock_ledger.policy.return_value = policy
    mock_ledger.append.return_value = fake_block

    # Spy forwarder: track that forward_block was called with the block.
    spy_forwarder = MagicMock()
    spy_forwarder.forward_block.return_value = {
        "persisted": True,
        "forwarded": True,
        "response": {"ok": True},
        "reason": None,
    }

    state = FSMState(
        constitution_hash=tree.hash(),
        behavioral_hash=tree.hash(),
        chain_head="0" * 64,
        step=0,
    )
    allowed, score, new_state = evaluate_action(
        {"type": "TOOL", "coherence_phi": 1.0},
        state,
        mock_ledger,
        forward=spy_forwarder,
    )

    assert allowed
    assert score > 0.0
    assert new_state is not None
    spy_forwarder.forward_block.assert_called_once_with(fake_block)


def test_evaluate_action_swallows_forward_errors(tmp_path: Path) -> None:
    """A forwarder that raises must not abort the local FSM transition."""
    from carl_core.constitutional import ConstitutionalPolicy
    from carl_core.eml import EMLNode, EMLOp, EMLTree
    from carl_studio.fsm_ledger import FSMState, evaluate_action

    tree = EMLTree(
        root=EMLNode(
            op=EMLOp.EML,
            left=EMLNode(op=EMLOp.CONST, const=2.0),
            right=EMLNode(op=EMLOp.CONST, const=1.0),
        ),
        input_dim=25,
    )
    policy = ConstitutionalPolicy.create(tree=tree, threshold=0.0)

    mock_ledger = MagicMock()
    mock_ledger.policy.return_value = policy
    mock_ledger.append.return_value = _make_block(block_id=11)

    raising_forwarder = MagicMock()
    raising_forwarder.forward_block.side_effect = RuntimeError("forward boom")

    state = FSMState(
        constitution_hash=tree.hash(),
        behavioral_hash=tree.hash(),
        chain_head="0" * 64,
        step=0,
    )
    allowed, _score, new_state = evaluate_action(
        {"type": "TOOL", "coherence_phi": 1.0},
        state,
        mock_ledger,
        forward=raising_forwarder,
    )

    # Local transition succeeded despite the forwarder exception.
    assert allowed
    assert new_state is not None
    assert new_state.step == 11
    raising_forwarder.forward_block.assert_called_once()


def test_default_carl_camp_base_constant() -> None:
    """Sanity: the module exposes the expected canonical base URL."""
    assert DEFAULT_CARL_CAMP_BASE == "https://carl.camp"
