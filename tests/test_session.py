"""Tests for carl_studio.session.Session + TwinCheckpoint.

Covers:
- Session construction wires chain + three vaults + four toolkits
- Vaults are SHARED across toolkits (not accidentally duplicated)
- Context-manager semantics (teardown runs on __exit__)
- register_tools_to via duck-typed dispatcher
- TwinCheckpoint snapshot/restore round-trip with content-hash verification
- Tampered checkpoint detected
- Only metadata (not values) appears in the snapshot
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from carl_core.errors import CARLError
from carl_core.interaction import ActionType

from carl_studio.session import Session, TwinCheckpoint


# ---------------------------------------------------------------------------
# Session construction
# ---------------------------------------------------------------------------


def test_session_default_construction() -> None:
    s = Session()
    assert s.user is None
    assert s.chain is not None
    assert s.secret_vault is not None
    assert s.resource_vault is not None
    assert s.data_vault is not None
    # Toolkits auto-constructed
    assert s.data_toolkit is not None
    assert s.browser_toolkit is not None
    assert s.subprocess_toolkit is not None
    assert s.cu_dispatcher is not None


def test_session_vaults_shared_across_toolkits() -> None:
    s = Session()
    assert s.data_toolkit.vault is s.data_vault
    assert s.data_toolkit.chain is s.chain
    assert s.browser_toolkit.data_toolkit is s.data_toolkit
    assert s.browser_toolkit.resource_vault is s.resource_vault
    assert s.browser_toolkit.secret_vault is s.secret_vault
    assert s.subprocess_toolkit.data_toolkit is s.data_toolkit
    assert s.subprocess_toolkit.resource_vault is s.resource_vault
    assert s.cu_dispatcher.browser is s.browser_toolkit


def test_session_with_explicit_user_and_chain() -> None:
    from carl_core.interaction import InteractionChain

    chain = InteractionChain()
    s = Session(user="tej", chain=chain)
    assert s.user == "tej"
    assert s.chain is chain


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------


def test_session_as_context_manager() -> None:
    with Session() as s:
        assert s.data_toolkit is not None
    # After exit, session still usable (teardown is a best-effort release)
    # Re-entering after teardown isn't a supported workflow.


def test_session_teardown_revokes_resource_refs() -> None:
    s = Session()
    # Simulate a resource ref (browser page) sitting in the resource vault.
    fake_backend = object()
    ref = s.resource_vault.put(
        backend=fake_backend,
        kind="subprocess",
        provider="subprocess",
        uri="pid:99999",
    )
    assert s.resource_vault.exists(ref) is True
    s.teardown()
    assert s.resource_vault.exists(ref) is False


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------


class _FakeDispatcher:
    def __init__(self) -> None:
        self.tools: dict[str, Any] = {}

    def register(self, name: str, fn: Any) -> None:
        self.tools[name] = fn


def test_session_register_tools_to_populates_dispatcher() -> None:
    s = Session()
    sink = _FakeDispatcher()
    names = s.register_tools_to(sink)
    # Should register 25+ tools (data + browser + subprocess + computer)
    assert len(names) >= 20
    # Spot-check representative tools from each toolkit
    assert "data_open_file" in names
    assert "browser_open_page" in names
    assert "subprocess_spawn" in names
    assert "computer" in names
    # All registered on sink
    assert set(names) == set(sink.tools.keys())


def test_session_anthropic_tools_is_union() -> None:
    s = Session()
    tools = s.anthropic_tools()
    names = {t["name"] for t in tools}
    assert "data_open_file" in names
    assert "browser_open_page" in names
    assert "subprocess_spawn" in names
    assert "computer" in names


# ---------------------------------------------------------------------------
# TwinCheckpoint snapshot / restore
# ---------------------------------------------------------------------------


def test_snapshot_has_content_hash() -> None:
    s = Session(user="tej")
    ckpt = s.snapshot()
    assert ckpt.content_hash
    assert len(ckpt.content_hash) == 64
    assert all(c in "0123456789abcdef" for c in ckpt.content_hash)


def test_snapshot_verify_returns_true_for_fresh_snapshot() -> None:
    s = Session()
    ckpt = s.snapshot()
    assert ckpt.verify() is True


def test_snapshot_content_hash_stable_under_reserialization() -> None:
    s = Session()
    s.chain.record(ActionType.LLM_REPLY, "test", input="x", output="y")
    ckpt = s.snapshot()
    # Round-trip: dump → load → compute
    dumped = ckpt.model_dump()
    reloaded = TwinCheckpoint.model_validate(dumped)
    assert reloaded.content_hash == ckpt.content_hash
    assert reloaded.verify() is True


def test_snapshot_tampered_fails_verify() -> None:
    s = Session()
    ckpt = s.snapshot()
    # Tamper: change the chain payload after the fact. We can't mutate the
    # frozen model — build a new one with the same content_hash but different body.
    tampered = TwinCheckpoint(
        schema_version=ckpt.schema_version,
        created_at=ckpt.created_at,
        user="different-user",
        chain=ckpt.chain,
        refs=ckpt.refs,
        content_hash=ckpt.content_hash,  # stale hash
    )
    assert tampered.verify() is False


def test_snapshot_refs_are_metadata_only() -> None:
    """Refs in the snapshot carry descriptor dicts, not raw values."""
    s = Session()
    # Put a secret; the value is "top-secret"
    s.secret_vault.put("top-secret", kind="mint")
    # Put some data bytes
    s.data_vault.put_bytes(b"inline-data-payload-bytes", content_type="text/plain")

    ckpt = s.snapshot()

    # Serialized form should NOT contain the raw value
    blob = json.dumps(ckpt.model_dump(), default=str)
    assert "top-secret" not in blob
    assert "inline-data-payload-bytes" not in blob
    # But should reference the refs via describe()
    assert len(ckpt.refs["secret"]) == 1
    assert len(ckpt.refs["data"]) == 1


def test_session_restore_rehydrates_chain() -> None:
    s = Session(user="tej")
    s.chain.record(ActionType.LLM_REPLY, "greet", input="hi", output="hello")
    ckpt = s.snapshot()
    restored = Session.restore(ckpt)
    assert restored.user == "tej"
    # Chain has the one step we recorded
    assert len(restored.chain) == 1
    assert restored.chain.by_action(ActionType.LLM_REPLY)[0].name == "greet"


def test_session_restore_rejects_tampered_checkpoint() -> None:
    s = Session()
    ckpt = s.snapshot()
    bad = TwinCheckpoint(
        schema_version=ckpt.schema_version,
        created_at=ckpt.created_at,
        user="tampered",
        chain=ckpt.chain,
        refs=ckpt.refs,
        content_hash=ckpt.content_hash,
    )
    with pytest.raises(CARLError) as exc:
        Session.restore(bad)
    assert exc.value.code == "carl.session.checkpoint_tampered"


# ---------------------------------------------------------------------------
# Integration: Session is ADDITIVE — InteractionChain() still works
# ---------------------------------------------------------------------------


def test_session_does_not_break_direct_chain_usage() -> None:
    from carl_core.interaction import InteractionChain

    chain = InteractionChain()
    chain.record(ActionType.LLM_REPLY, "direct", input="x", output="y")
    assert len(chain) == 1
    # And we can still construct a Session from the same chain if we want to
    s = Session(chain=chain)
    assert s.chain is chain
    assert len(s.chain) == 1
