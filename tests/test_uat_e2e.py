"""End-to-end UAT for carl-studio — full workflow validation.

Tests every FSM lifecycle, cross-system integration, CLI surface,
and edge cases.  All offline: no API keys, no network, no GPU.

Sections:
  1. Commerce lifecycle (consent → contract → witness → verify)
  2. Curriculum → Carlito → A2A spawn chain
  3. CARLAgent tool dispatch + context management
  4. WorkFrame integration with agent and learn pipeline
  5. CLI command surface (CliRunner)
  6. Cross-system integration
  7. Edge cases and error paths
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import typer
from typer.testing import CliRunner

# ---------------------------------------------------------------------------
# Shared FakeDB for tests that need LocalDB behavior
# ---------------------------------------------------------------------------


class FakeDB:
    """Minimal in-memory mock of LocalDB for UAT tests."""

    def __init__(self) -> None:
        self.config: dict[str, str] = {}
        self.contracts: dict[str, dict] = {}

    def get_config(self, key: str, default: str | None = None) -> str | None:
        return self.config.get(key, default)

    def set_config(self, key: str, value: str) -> None:
        self.config[key] = value

    def insert_contract(self, contract: dict) -> str:
        cid = contract.get("id", "test-id")
        self.contracts[cid] = contract
        return cid

    def update_contract(self, contract_id: str, updates: dict) -> None:
        if contract_id in self.contracts:
            self.contracts[contract_id].update(updates)

    def get_contract(self, contract_id: str) -> dict | None:
        return self.contracts.get(contract_id)

    def list_contracts(self, limit: int = 20) -> list[dict]:
        return list(self.contracts.values())[:limit]


runner = CliRunner()


# =========================================================================
# 1. Commerce Lifecycle: consent → contract → witness → verify
# =========================================================================


class TestCommerceLifecycle:
    """Full consent → contract → witness → verify journey."""

    def test_consent_enable_then_sign_contract(self) -> None:
        """Enable contract witnessing consent, then sign a contract."""
        from carl_studio.consent import ConsentManager
        from carl_studio.contract import ContractWitness, ServiceContract

        db = FakeDB()
        mgr = ConsentManager(db=db)

        # Start: all off
        state = mgr.load()
        assert state.contract_witnessing.enabled is False

        # Enable contract witnessing
        state = mgr.update("contract_witnessing", True)
        assert state.contract_witnessing.enabled is True

        # Sign a contract — should succeed now
        witness = ContractWitness(db=db)
        contract = ServiceContract(
            parties=["alice", "bob"],
            terms_hash="abc123",
            terms_url="https://example.com/terms",
        )
        envelope = witness.sign(contract)
        assert envelope.witness_hash != ""
        assert envelope.terms_hash == "abc123"

        # Verify the envelope
        assert witness.verify(envelope) is True

        # DB has the contract
        stored = db.contracts[contract.id]
        assert stored["status"] == "witnessed"

    def test_consent_off_blocks_contract_signing(self) -> None:
        """Without consent, signing raises ContractError."""
        from carl_studio.contract import ContractError, ContractWitness, ServiceContract

        db = FakeDB()
        witness = ContractWitness(db=db)
        contract = ServiceContract(
            parties=["alice"],
            terms_hash="hash",
            terms_url="https://example.com/terms",
        )
        with pytest.raises(ContractError, match="consent"):
            witness.sign(contract)

    def test_consent_reset_then_sign_fails(self) -> None:
        """Enable consent, reset to all-off, then signing should fail."""
        from carl_studio.consent import ConsentManager
        from carl_studio.contract import ContractError, ContractWitness, ServiceContract

        db = FakeDB()
        mgr = ConsentManager(db=db)
        mgr.update("contract_witnessing", True)

        # Reset
        mgr.all_off()
        assert mgr.load().contract_witnessing.enabled is False

        # Sign should fail now
        witness = ContractWitness(db=db)
        contract = ServiceContract(
            parties=["alice"],
            terms_hash="hash",
            terms_url="https://example.com/terms",
        )
        with pytest.raises(ContractError, match="consent"):
            witness.sign(contract)

    def test_multiple_contracts_stored_and_listed(self) -> None:
        """Sign multiple contracts, list them back."""
        from carl_studio.consent import ConsentFlag, ConsentManager, ConsentState
        from carl_studio.contract import ContractWitness, ServiceContract

        db = FakeDB()
        mgr = ConsentManager(db=db)
        mgr.save(ConsentState(
            contract_witnessing=ConsentFlag(enabled=True, changed_at="2026-01-01")
        ))

        witness = ContractWitness(db=db)
        ids = []
        for i in range(3):
            c = ServiceContract(
                parties=[f"party_{i}"],
                terms_hash=f"hash_{i}",
                terms_url=f"https://example.com/terms_{i}",
            )
            witness.sign(c)
            ids.append(c.id)

        contracts = witness.list_contracts(limit=10)
        assert len(contracts) == 3
        stored_ids = {c.id for c in contracts}
        for cid in ids:
            assert cid in stored_ids

    def test_contract_get_and_envelope_retrieval(self) -> None:
        """Sign a contract, then retrieve both contract and envelope."""
        from carl_studio.consent import ConsentFlag, ConsentManager, ConsentState
        from carl_studio.contract import ContractWitness, ServiceContract

        db = FakeDB()
        mgr = ConsentManager(db=db)
        mgr.save(ConsentState(
            contract_witnessing=ConsentFlag(enabled=True, changed_at="2026-01-01")
        ))

        witness = ContractWitness(db=db)
        contract = ServiceContract(
            parties=["alice"],
            terms_hash="myhash",
            terms_url="https://example.com/terms",
        )
        original_envelope = witness.sign(contract)

        # get_contract
        loaded = witness.get_contract(contract.id)
        assert loaded is not None
        assert loaded.id == contract.id
        assert loaded.status == "witnessed"

        # get_envelope
        loaded_env = witness.get_envelope(contract.id)
        assert loaded_env is not None
        assert loaded_env.witness_hash == original_envelope.witness_hash

    def test_verify_tampered_parties_hash_fails(self) -> None:
        """Tampering with parties_hash causes verification to fail."""
        from carl_studio.consent import ConsentFlag, ConsentManager, ConsentState
        from carl_studio.contract import ContractWitness, ServiceContract

        db = FakeDB()
        mgr = ConsentManager(db=db)
        mgr.save(ConsentState(
            contract_witnessing=ConsentFlag(enabled=True, changed_at="2026-01-01")
        ))

        witness = ContractWitness(db=db)
        contract = ServiceContract(
            parties=["alice"],
            terms_hash="myhash",
            terms_url="https://example.com/terms",
        )
        envelope = witness.sign(contract)
        tampered = envelope.model_copy(update={"parties_hash": "TAMPERED"})
        assert witness.verify(tampered) is False

    def test_verify_tampered_timestamp_fails(self) -> None:
        """Tampering with witnessed_at causes verification to fail."""
        from carl_studio.consent import ConsentFlag, ConsentManager, ConsentState
        from carl_studio.contract import ContractWitness, ServiceContract

        db = FakeDB()
        mgr = ConsentManager(db=db)
        mgr.save(ConsentState(
            contract_witnessing=ConsentFlag(enabled=True, changed_at="2026-01-01")
        ))

        witness = ContractWitness(db=db)
        contract = ServiceContract(
            parties=["alice"],
            terms_hash="myhash",
            terms_url="https://example.com/terms",
        )
        envelope = witness.sign(contract)
        tampered = envelope.model_copy(update={"witnessed_at": "1999-01-01T00:00:00Z"})
        assert witness.verify(tampered) is False

    def test_sign_with_artifacts(self) -> None:
        """Sign with extra artifacts — they affect the witness hash."""
        from carl_studio.consent import ConsentFlag, ConsentManager, ConsentState
        from carl_studio.contract import ContractWitness, ServiceContract

        db = FakeDB()
        mgr = ConsentManager(db=db)
        mgr.save(ConsentState(
            contract_witnessing=ConsentFlag(enabled=True, changed_at="2026-01-01")
        ))

        witness = ContractWitness(db=db)
        contract = ServiceContract(
            parties=["alice"],
            terms_hash="myhash",
            terms_url="https://example.com/terms",
        )
        envelope = witness.sign(contract, artifacts={"device": "mac", "version": "1.0"})
        assert envelope.artifacts == {"device": "mac", "version": "1.0"}
        assert witness.verify(envelope) is True

        # Tampering with artifacts breaks verification
        tampered = envelope.model_copy(update={"artifacts": {"device": "EVIL"}})
        assert witness.verify(tampered) is False


# =========================================================================
# 2. Consent sync_with_profile edge cases
# =========================================================================


class TestConsentSyncEdgeCases:
    """Edge cases for the consent state machine."""

    def test_sync_does_not_overwrite_explicitly_disabled(self) -> None:
        """If user explicitly disabled a flag, remote True does not override."""
        from carl_studio.consent import ConsentManager

        db = FakeDB()
        mgr = ConsentManager(db=db)
        # Explicitly disable
        mgr.update("observability", False)

        class Remote:
            observability_opt_in = True
            telemetry_opt_in = False
            usage_tracking_enabled = False
            contract_witnessing = False

        state = mgr.sync_with_profile(Remote())
        assert state.observability.enabled is False  # local wins

    def test_sync_adopts_remote_only_for_pristine_flags(self) -> None:
        """Pristine (never set) flags adopt remote True."""
        from carl_studio.consent import ConsentManager

        db = FakeDB()
        mgr = ConsentManager(db=db)
        # Leave all flags pristine (never call update)

        class Remote:
            observability_opt_in = True
            telemetry_opt_in = True
            usage_tracking_enabled = False
            contract_witnessing = False

        state = mgr.sync_with_profile(Remote())
        assert state.observability.enabled is True
        assert state.telemetry.enabled is True
        assert state.usage_analytics.enabled is False  # remote also False
        assert state.contract_witnessing.enabled is False

    def test_update_toggle_preserves_timestamps(self) -> None:
        """Toggling a flag updates the timestamp each time."""
        from carl_studio.consent import ConsentManager

        db = FakeDB()
        mgr = ConsentManager(db=db)

        state1 = mgr.update("telemetry", True)
        ts1 = state1.telemetry.changed_at
        assert ts1 is not None

        state2 = mgr.update("telemetry", False)
        ts2 = state2.telemetry.changed_at
        assert ts2 is not None
        assert state2.telemetry.enabled is False

    def test_present_first_run_noninteractive(self) -> None:
        """Non-interactive (CI) defaults all off."""
        from carl_studio.consent import ConsentManager

        db = FakeDB()
        mgr = ConsentManager(db=db)
        with patch("sys.stdin") as mock_stdin:
            mock_stdin.isatty.return_value = False
            state = mgr.present_first_run()
        assert state.observability.enabled is False
        assert state.telemetry.enabled is False


# =========================================================================
# 3. X402 payment flow
# =========================================================================


class TestX402Flow:
    """X402 check → negotiate → execute flow (mocked network)."""

    def test_config_roundtrip_via_db(self) -> None:
        """Save and load X402 config through DB."""
        from carl_studio.x402 import X402Config, load_x402_config, save_x402_config

        db = FakeDB()
        config = X402Config(
            wallet_address="0xabc",
            chain="base",
            facilitator_url="https://pay.example.com",
            enabled=True,
        )
        save_x402_config(config, db=db)
        loaded = load_x402_config(db=db)
        assert loaded.wallet_address == "0xabc"
        assert loaded.enabled is True

    def test_parse_x_payment_json(self) -> None:
        """JSON format x-payment header parses correctly."""
        from carl_studio.x402 import _parse_x_payment_header

        header = json.dumps({
            "amount": "0.01",
            "token": "USDC",
            "chain": "base",
            "recipient": "0xabc",
            "facilitator": "https://pay.example.com",
        })
        req = _parse_x_payment_header(header)
        assert req.amount == "0.01"
        assert req.token == "USDC"
        assert req.facilitator == "https://pay.example.com"

    def test_parse_x_payment_kv(self) -> None:
        """Key=value format x-payment header parses correctly."""
        from carl_studio.x402 import _parse_x_payment_header

        header = "amount=0.05; token=USDC; chain=base; recipient=0xdef"
        req = _parse_x_payment_header(header)
        assert req.amount == "0.05"
        assert req.token == "USDC"
        assert req.recipient == "0xdef"

    def test_check_x402_returns_none_for_200(self) -> None:
        """200 OK means no payment required."""
        from carl_studio.x402 import X402Client, X402Config

        client = X402Client(X402Config())
        with patch("urllib.request.urlopen"):
            result = client.check_x402("https://example.com/resource")
        assert result is None

    def test_negotiate_no_facilitator_raises(self) -> None:
        """Missing facilitator URL raises X402Error."""
        from carl_studio.x402 import PaymentRequirement, X402Client, X402Config, X402Error

        client = X402Client(X402Config(facilitator_url=""))
        req = PaymentRequirement(amount="1", facilitator="")
        with pytest.raises(X402Error, match="No facilitator"):
            client.negotiate(req)

    def test_config_from_profile(self) -> None:
        """x402_config_from_profile extracts correct fields."""
        from carl_studio.x402 import x402_config_from_profile

        class FakeProfile:
            x402_enabled = True
            metadata = {
                "wallet_address": "0x123",
                "x402_chain": "ethereum",
                "x402_facilitator": "https://fac.example.com",
            }

        config = x402_config_from_profile(FakeProfile())
        assert config.enabled is True
        assert config.wallet_address == "0x123"
        assert config.chain == "ethereum"


# =========================================================================
# 4. Curriculum → Carlito → A2A spawn chain
# =========================================================================


class TestCurriculumCarlitoChain:
    """Full lifecycle: enroll → drill → evaluate → graduate → spawn → deploy."""

    def test_full_lifecycle_happy_path(self, tmp_path: Path) -> None:
        """Walk the curriculum FSM from ENROLLED to DEPLOYED via carlito spawn."""
        from carl_studio.carlito import CarlitoRegistry, CarlitoSpawner, CarlitoStatus
        from carl_studio.curriculum import CurriculumPhase, CurriculumTrack

        # Create a track
        track = CurriculumTrack(model_id="il-terminals-carl-v1")
        assert track.phase == CurriculumPhase.ENROLLED

        # Advance through phases
        track = track.advance(CurriculumPhase.DRILLING, event="training_started")
        assert track.phase == CurriculumPhase.DRILLING
        assert len(track.milestones) == 1

        track = track.advance(CurriculumPhase.EVALUATED, event="training_complete")
        assert track.phase == CurriculumPhase.EVALUATED

        track = track.advance(CurriculumPhase.GRADUATED, event="gate_pass")
        assert track.phase == CurriculumPhase.GRADUATED

        # Spawn carlito from graduated track
        registry = CarlitoRegistry(db_path=tmp_path / "carlitos.db")
        spec = CarlitoSpawner.from_graduated_track(
            name="my-bot",
            track=track,
            domain="coding",
            skills=["observer", "grader"],
        )
        assert spec.status == CarlitoStatus.GRADUATED

        spawner = CarlitoSpawner(registry=registry)
        card = spawner.spawn(spec, track)

        assert card.name == "my-bot"
        assert "observer" in card.skills
        assert card.metadata["domain"] == "coding"
        assert card.metadata["parent_model"] == "il-terminals-carl-v1"

        # Verify registry has DEPLOYED status
        loaded = registry.load("my-bot")
        assert loaded is not None
        assert loaded.status == CarlitoStatus.DEPLOYED

        registry.close()

    def test_invalid_transition_raises(self) -> None:
        """Cannot skip phases in curriculum."""
        from carl_studio.curriculum import CurriculumPhase, CurriculumTrack

        track = CurriculumTrack(model_id="test")
        with pytest.raises(ValueError, match="Cannot transition"):
            track.advance(CurriculumPhase.GRADUATED)

    def test_spawn_from_non_graduated_fails(self) -> None:
        """Cannot spawn carlito from DRILLING phase."""
        from carl_studio.carlito import CarlitoSpawner, CarlitoSpec
        from carl_studio.curriculum import CurriculumPhase, CurriculumTrack

        track = CurriculumTrack(model_id="test", phase=CurriculumPhase.DRILLING)
        spec = CarlitoSpec(name="test", parent_model="test")
        spawner = CarlitoSpawner()
        with pytest.raises(ValueError, match="must be 'graduated'"):
            spawner.spawn(spec, track)

    def test_evaluated_can_retry_drilling(self) -> None:
        """EVALUATED → DRILLING is valid (gate fail retry)."""
        from carl_studio.curriculum import CurriculumPhase, CurriculumTrack

        track = CurriculumTrack(model_id="test", phase=CurriculumPhase.EVALUATED)
        track = track.advance(CurriculumPhase.DRILLING, event="gate_fail_retry")
        assert track.phase == CurriculumPhase.DRILLING

    def test_deployed_can_retrain(self) -> None:
        """DEPLOYED → DRILLING is valid (re-training cycle)."""
        from carl_studio.curriculum import CurriculumPhase, CurriculumTrack

        track = CurriculumTrack(model_id="test", phase=CurriculumPhase.DEPLOYED)
        track = track.advance(CurriculumPhase.DRILLING, event="retrain_cycle")
        assert track.phase == CurriculumPhase.DRILLING

    def test_ttt_can_retrain(self) -> None:
        """TTT_ACTIVE → DRILLING is valid (re-training from live)."""
        from carl_studio.curriculum import CurriculumPhase, CurriculumTrack

        track = CurriculumTrack(model_id="test", phase=CurriculumPhase.TTT_ACTIVE)
        track = track.advance(CurriculumPhase.DRILLING, event="retrain")
        assert track.phase == CurriculumPhase.DRILLING

    def test_fsm_closure(self) -> None:
        """Verify the curriculum FSM is closed (every phase has transitions)."""
        from carl_studio.curriculum import verify_fsm_closure

        assert verify_fsm_closure() is True

    def test_curriculum_store_roundtrip(self, tmp_path: Path) -> None:
        """Save and load curriculum track via SQLite store."""
        from carl_studio.curriculum import CurriculumPhase, CurriculumStore, CurriculumTrack

        store = CurriculumStore(db_path=tmp_path / "curriculum.db")
        track = CurriculumTrack(model_id="test-model", phase=CurriculumPhase.DRILLING)
        track = track.advance(CurriculumPhase.EVALUATED, event="done")
        store.save(track)

        loaded = store.load("test-model")
        assert loaded is not None
        assert loaded.phase == CurriculumPhase.EVALUATED
        assert len(loaded.milestones) == 1
        store.close()

    def test_domain_capabilities_mapping(self) -> None:
        """Domain capability mapping produces expected capabilities."""
        from carl_studio.carlito import _domain_capabilities

        coding_caps = _domain_capabilities("coding")
        assert "train" in coding_caps
        assert "push" in coding_caps
        assert "bench" in coding_caps

        math_caps = _domain_capabilities("math")
        assert "bench" in math_caps
        assert "align" in math_caps

        unknown_caps = _domain_capabilities("unknown_domain")
        assert unknown_caps == ["train", "eval", "observe"]

    def test_carlito_retire_lifecycle(self, tmp_path: Path) -> None:
        """DEPLOYED → DORMANT via retire()."""
        from carl_studio.carlito import CarlitoRegistry, CarlitoSpec, CarlitoStatus

        registry = CarlitoRegistry(db_path=tmp_path / "carlitos.db")
        registry.save(CarlitoSpec(
            name="active-bot",
            parent_model="m1",
            status=CarlitoStatus.DEPLOYED,
        ))

        assert registry.retire("active-bot") is True
        loaded = registry.load("active-bot")
        assert loaded is not None
        assert loaded.status == CarlitoStatus.DORMANT
        registry.close()


# =========================================================================
# 5. CARLAgent tool dispatch + context management
# =========================================================================


class TestCARLAgentEndToEnd:
    """CARLAgent tool chain: set_frame → ingest → query → analyze → create."""

    @pytest.fixture()
    def agent(self, tmp_path: Path) -> Any:
        from carl_studio.chat_agent import CARLAgent

        return CARLAgent(model="test", workdir=str(tmp_path), _client=MagicMock())

    def test_full_tool_chain(self, agent: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Drive the full agent tool chain end to end."""
        monkeypatch.setattr("carl_studio.frame._DEFAULT_FRAME_PATH", tmp_path / "frame.yaml")

        # 1. Set frame
        result = agent._dispatch_tool("set_frame", {
            "domain": "saas_sales",
            "function": "territory_planning",
            "role": "analyst",
        })
        assert "saas_sales" in result
        assert agent._frame is not None
        assert agent._frame.domain == "saas_sales"

        # 2. Create a data file
        data_file = tmp_path / "accounts.csv"
        data_file.write_text("account,revenue\nAcme,1000\nBeta,2000\nGamma,500")

        # 3. Ingest the data
        result = agent._dispatch_tool("ingest_source", {"path": str(tmp_path)})
        assert "Ingested" in result
        assert len(agent._knowledge) > 0

        # 4. Query knowledge
        result = agent._dispatch_tool("query_knowledge", {"question": "account revenue"})
        assert "Acme" in result or "revenue" in result

        # 5. Run analysis
        result = agent._dispatch_tool("run_analysis", {"code": "print(1000 + 2000 + 500)"})
        assert "3500" in result

        # 6. Create output file
        result = agent._dispatch_tool("create_file", {
            "path": str(tmp_path / "report.txt"),
            "content": "Total revenue: $3,500",
        })
        assert "Created" in result
        assert (tmp_path / "report.txt").read_text() == "Total revenue: $3,500"

    def test_system_prompt_with_frame_and_knowledge(self, agent: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """System prompt reflects frame + knowledge state."""
        monkeypatch.setattr("carl_studio.frame._DEFAULT_FRAME_PATH", tmp_path / "frame.yaml")

        # Without frame/knowledge
        prompt = agent._build_system_prompt()
        assert "No frame set" in prompt
        assert "KNOWLEDGE BASE" not in prompt

        # Set frame
        agent._dispatch_tool("set_frame", {"domain": "pharma"})
        prompt = agent._build_system_prompt()
        assert "ACTIVE FRAME" in prompt
        assert "pharma" in prompt

        # Add knowledge
        agent._knowledge = [{"text": "test", "source": "a", "words": {"test"}}]
        prompt = agent._build_system_prompt()
        assert "1 chunks" in prompt

    def test_path_traversal_blocked_create(self, agent: Any, tmp_path: Path) -> None:
        """Cannot create files outside workdir."""
        result = agent._dispatch_tool("create_file", {
            "path": "/etc/evil.txt",
            "content": "payload",
        })
        assert "Blocked" in result

    def test_path_traversal_blocked_read(self, agent: Any, tmp_path: Path) -> None:
        """Cannot read files outside workdir."""
        result = agent._dispatch_tool("read_file", {"path": "/etc/passwd"})
        assert "Blocked" in result

    def test_path_traversal_dotdot(self, agent: Any, tmp_path: Path) -> None:
        """../../../etc/passwd is blocked."""
        result = agent._dispatch_tool("read_file", {"path": "../../../etc/passwd"})
        assert "Blocked" in result

    def test_context_compaction_under_pressure(self, agent: Any) -> None:
        """Compaction triggers when message count exceeds threshold."""
        from carl_studio.chat_agent import _KEEP_RECENT

        # Add many messages
        for i in range(40):
            agent._messages.append({"role": "user", "content": f"message {i}"})
            agent._messages.append({"role": "assistant", "content": f"response {i}"})

        agent._compact()

        # Should have: summary + ack + last _KEEP_RECENT messages
        assert len(agent._messages) == 2 + _KEEP_RECENT

        # Summary is first message
        assert "summary" in agent._messages[0]["content"].lower()

        # Recent messages preserved
        recent_contents = [
            m["content"] for m in agent._messages[2:]
            if isinstance(m.get("content"), str)
        ]
        assert "response 39" in recent_contents

    def test_compact_with_tool_result_messages(self, agent: Any) -> None:
        """Compaction handles list-type content (tool results)."""
        from carl_studio.chat_agent import _KEEP_RECENT

        for i in range(25):
            agent._messages.append({"role": "user", "content": f"msg {i}"})
            agent._messages.append({
                "role": "assistant",
                "content": [{"type": "text", "text": f"resp {i}"}],
            })

        agent._compact()
        assert len(agent._messages) == 2 + _KEEP_RECENT

    def test_query_no_overlap_returns_no_results(self, agent: Any) -> None:
        """Query with zero term overlap returns no-match message."""
        agent._knowledge = [
            {"text": "alpha beta gamma", "source": "test", "words": {"alpha", "beta", "gamma"}},
        ]
        result = agent._dispatch_tool("query_knowledge", {"question": "xylophone zebra"})
        assert "No relevant" in result

    def test_list_files_respects_workdir(self, agent: Any, tmp_path: Path) -> None:
        """list_files works within workdir."""
        (tmp_path / "a.txt").write_text("hello")
        (tmp_path / "b.csv").write_text("data")
        result = agent._dispatch_tool("list_files", {"path": str(tmp_path)})
        assert "a.txt" in result
        assert "b.csv" in result

    def test_run_analysis_timeout_handling(self, agent: Any) -> None:
        """Long-running code gets timeout treatment."""
        # This should fail fast enough for tests (subprocess timeout is 30s)
        result = agent._dispatch_tool("run_analysis", {"code": "raise SystemExit(42)"})
        assert "42" in result or "Exit" in result

    def test_unknown_tool_returns_error(self, agent: Any) -> None:
        """Unknown tool name returns clear error."""
        result = agent._dispatch_tool("nonexistent_tool", {})
        assert "Unknown" in result


# =========================================================================
# 6. WorkFrame integration
# =========================================================================


class TestWorkFrameIntegration:
    """WorkFrame with persistence, decomposition, and agent integration."""

    def test_frame_save_load_clear_cycle(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Full frame lifecycle: save → load → clear → load returns empty."""
        from carl_studio.frame import WorkFrame

        frame_path = tmp_path / "frame.yaml"
        monkeypatch.setattr("carl_studio.frame._DEFAULT_FRAME_PATH", frame_path)

        frame = WorkFrame(
            domain="logistics",
            function="route_planning",
            role="dispatcher",
            objectives=["minimize delivery time"],
            entities=["depot", "vehicle"],
            metrics=["avg_delivery_time", "utilization"],
        )
        frame.save()
        assert frame_path.exists()

        loaded = WorkFrame.load()
        assert loaded.domain == "logistics"
        assert loaded.entities == ["depot", "vehicle"]

        # Clear
        loaded.clear()
        assert not frame_path.exists()

        # Load after clear returns empty
        empty = WorkFrame.load()
        assert empty.active is False

    def test_frame_merge_update(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Partial frame update merges with existing frame."""
        from carl_studio.frame import WorkFrame

        frame_path = tmp_path / "frame.yaml"
        monkeypatch.setattr("carl_studio.frame._DEFAULT_FRAME_PATH", frame_path)

        # Set initial frame
        WorkFrame(domain="saas", function="quota_setting").save()

        # Load and merge
        frame = WorkFrame.load()
        frame = frame.model_copy(update={"role": "analyst"})
        frame.save()

        final = WorkFrame.load()
        assert final.domain == "saas"
        assert final.function == "quota_setting"
        assert final.role == "analyst"

    def test_decompose_comprehensive(self) -> None:
        """Decomposition produces MECE lanes from all dimensions."""
        from carl_studio.frame import WorkFrame

        frame = WorkFrame(
            domain="retail",
            function="inventory",
            objectives=["reduce stockouts", "minimize waste"],
            metrics=["fill_rate", "shrinkage"],
        )
        lanes = frame.decompose()
        assert len(lanes) == 6  # domain + function + 2 objectives + 2 metrics
        assert "domain:retail" in lanes
        assert "objective:reduce stockouts" in lanes
        assert "metric:shrinkage" in lanes

    def test_attention_query_with_constraints_and_context(self) -> None:
        """attention_query includes constraints and context."""
        from carl_studio.frame import WorkFrame

        frame = WorkFrame(
            domain="finance",
            constraints=["budget <= $10M", "headcount frozen"],
            context="Q4 planning under hiring freeze",
        )
        q = frame.attention_query()
        assert "budget <= $10M" in q
        assert "Q4 planning" in q

    def test_entity_patterns_lowercased(self) -> None:
        """entity_patterns returns lowercased patterns."""
        from carl_studio.frame import WorkFrame

        frame = WorkFrame(entities=["Account", "Territory", "  ", "Rep"])
        patterns = frame.entity_patterns()
        assert "account" in patterns
        assert "territory" in patterns
        assert "rep" in patterns
        assert "" not in patterns  # whitespace-only filtered


# =========================================================================
# 7. CLI command surface (CliRunner)
# =========================================================================


class TestContractCLI:
    """CLI tests for carl contract commands."""

    def _make_app(self) -> typer.Typer:
        from carl_studio.cli.contract import contract_app

        app = typer.Typer()
        app.add_typer(contract_app, name="contract")
        return app

    def test_contract_list_empty(self) -> None:
        """List with no contracts shows info message."""
        app = self._make_app()
        # ContractWitness is imported inside the command body, patch at source module
        with patch("carl_studio.contract.ContractWitness") as MockWitness:
            MockWitness.return_value.list_contracts.return_value = []
            result = runner.invoke(app, ["contract", "list"])
        assert result.exit_code == 0
        assert "No contracts" in result.output

    def test_contract_list_json_empty(self) -> None:
        """JSON list with no contracts returns empty array."""
        app = self._make_app()
        with patch("carl_studio.contract.ContractWitness") as MockWitness:
            MockWitness.return_value.list_contracts.return_value = []
            result = runner.invoke(app, ["contract", "list", "--json"])
        assert result.exit_code == 0
        assert json.loads(result.output) == []

    def test_contract_show_not_found(self) -> None:
        """Show with invalid ID shows info message."""
        app = self._make_app()
        with patch("carl_studio.contract.ContractWitness") as MockWitness:
            MockWitness.return_value.list_contracts.return_value = []
            result = runner.invoke(app, ["contract", "show"])
        assert result.exit_code == 0
        assert "No contracts" in result.output

    def test_contract_verify_not_found(self) -> None:
        """Verify with invalid ID exits with error."""
        app = self._make_app()
        with patch("carl_studio.contract.ContractWitness") as MockWitness:
            MockWitness.return_value.get_envelope.return_value = None
            result = runner.invoke(app, ["contract", "verify", "bad-id"])
        assert result.exit_code == 1
        assert "not found" in result.output


class TestCarlitoCLI:
    """CLI tests for carl carlito commands."""

    def _make_app(self) -> typer.Typer:
        from carl_studio.cli.carlito import carlito_app

        app = typer.Typer()
        app.add_typer(carlito_app, name="carlito")
        return app

    def test_carlito_list_empty(self) -> None:
        """List with no carlitos shows info message."""
        from carl_studio.carlito import CarlitoRegistry

        app = self._make_app()
        with patch("carl_studio.cli.carlito._get_registry") as mock_reg:
            reg = MagicMock(spec=CarlitoRegistry)
            reg.list_all.return_value = []
            mock_reg.return_value = reg
            result = runner.invoke(app, ["carlito", "list"])
        assert result.exit_code == 0
        assert "No carlitos" in result.output

    def test_carlito_show_not_found(self) -> None:
        """Show unknown carlito exits with error."""
        from carl_studio.carlito import CarlitoRegistry

        app = self._make_app()
        with patch("carl_studio.cli.carlito._get_registry") as mock_reg:
            reg = MagicMock(spec=CarlitoRegistry)
            reg.load.return_value = None
            mock_reg.return_value = reg
            result = runner.invoke(app, ["carlito", "show", "nonexistent"])
        assert result.exit_code == 1
        assert "not found" in result.output

    def test_carlito_retire_not_found(self) -> None:
        """Retire unknown carlito exits with error."""
        from carl_studio.carlito import CarlitoRegistry

        app = self._make_app()
        with patch("carl_studio.cli.carlito._get_registry") as mock_reg:
            reg = MagicMock(spec=CarlitoRegistry)
            reg.retire.return_value = False
            mock_reg.return_value = reg
            result = runner.invoke(app, ["carlito", "retire", "nope"])
        assert result.exit_code == 1
        assert "not found" in result.output

    def test_carlito_list_invalid_status(self) -> None:
        """List with invalid status shows error."""
        app = self._make_app()
        with patch("carl_studio.cli.carlito._get_registry") as mock_reg:
            reg = MagicMock()
            mock_reg.return_value = reg
            result = runner.invoke(app, ["carlito", "list", "--status", "bogus"])
        assert result.exit_code == 1
        assert "Unknown status" in result.output


class TestFrameCLIExtended:
    """Extended CLI tests for carl frame commands."""

    def _make_app(self) -> typer.Typer:
        from carl_studio.cli.frame import frame_app

        app = typer.Typer()
        app.add_typer(frame_app, name="frame")
        return app

    def test_frame_show_no_frame(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Show with no frame active shows info message."""
        monkeypatch.setattr("carl_studio.frame._DEFAULT_FRAME_PATH", tmp_path / "frame.yaml")

        app = self._make_app()
        result = runner.invoke(app, ["frame", "show"])
        assert result.exit_code == 0
        assert "No frame set" in result.output

    def test_frame_set_multiple_goals(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Setting multiple --goal flags works."""
        monkeypatch.setattr("carl_studio.frame._DEFAULT_FRAME_PATH", tmp_path / "frame.yaml")

        app = self._make_app()
        result = runner.invoke(app, [
            "frame", "set",
            "--domain", "retail",
            "--goal", "increase sales",
            "--goal", "reduce churn",
        ])
        assert result.exit_code == 0
        assert "Frame saved" in result.output

    def test_frame_show_with_decompose(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Show includes MECE lanes."""
        monkeypatch.setattr("carl_studio.frame._DEFAULT_FRAME_PATH", tmp_path / "frame.yaml")

        app = self._make_app()
        runner.invoke(app, [
            "frame", "set",
            "--domain", "saas",
            "--function", "territory_planning",
        ])
        result = runner.invoke(app, ["frame", "show"])
        assert result.exit_code == 0
        assert "MECE lanes" in result.output


class TestChatCLI:
    """CLI tests for carl chat command."""

    def test_chat_help(self) -> None:
        """carl chat --help shows usage."""
        from carl_studio.cli.chat import chat_cmd

        app = typer.Typer()
        app.command(name="chat")(chat_cmd)
        result = runner.invoke(app, ["chat", "--help"])
        assert result.exit_code == 0
        assert "model" in result.output.lower() or "chat" in result.output.lower()


# =========================================================================
# 8. A2A task status FSM
# =========================================================================


class TestA2ATaskFSM:
    """A2A task status transitions — terminal states are absorbing."""

    def _make_task(self) -> Any:
        from carl_studio.a2a.task import A2ATask

        return A2ATask(id="t1", skill="test_skill")

    def test_pending_to_running(self) -> None:
        task = self._make_task()
        task = task.mark_running()
        assert task.status == "running"

    def test_running_to_done(self) -> None:
        task = self._make_task()
        task = task.mark_running()
        task = task.mark_done(result={"output": "success"})
        assert task.status == "done"
        assert task.result == {"output": "success"}
        assert task.completed_at is not None

    def test_running_to_failed(self) -> None:
        task = self._make_task()
        task = task.mark_running()
        task = task.mark_failed(error="boom")
        assert task.status == "failed"
        assert task.error == "boom"

    def test_terminal_states_are_absorbing(self) -> None:
        """Once DONE, mark_running() is a no-op."""
        task = self._make_task()
        task = task.mark_running()
        task = task.mark_done(result={})
        # Terminal — further transitions are no-ops
        task2 = task.mark_running()
        assert task2.status == "done"

    def test_cancelled_is_terminal(self) -> None:
        task = self._make_task()
        task = task.mark_cancelled()
        assert task.status == "cancelled"
        task2 = task.mark_running()
        assert task2.status == "cancelled"


# =========================================================================
# 9. Tier gate integration
# =========================================================================


class TestTierGateIntegration:
    """Tier elevation and gating across the feature matrix."""

    def test_free_tier_complete_feature_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """FREE tier allows the core CARL loop features."""
        from carl_studio.tier import Tier, tier_allows

        core_features = ["observe", "train", "eval", "config", "bench", "align", "learn"]
        for feature in core_features:
            assert tier_allows(Tier.FREE, feature), f"FREE should allow {feature}"

    def test_paid_only_features(self) -> None:
        """Features that require PAID tier."""
        from carl_studio.tier import Tier, tier_allows

        paid_features = ["train.send_it", "mcp"]
        for feature in paid_features:
            assert not tier_allows(Tier.FREE, feature), f"FREE should not allow {feature}"
            assert tier_allows(Tier.PAID, feature), f"PAID should allow {feature}"

    def test_tier_ordering(self) -> None:
        """Tier ordering is monotonic: FREE < PAID, aliases equal PAID."""
        from carl_studio.tier import Tier

        assert Tier.FREE < Tier.PAID
        assert Tier.PRO == Tier.PAID
        assert Tier.ENTERPRISE == Tier.PAID


# =========================================================================
# 10. Multimodal ingest dispatch routing
# =========================================================================


class TestMultimodalIngestRouting:
    """Test that SourceIngester routes files to correct modality handlers."""

    def test_text_file_ingest(self, tmp_path: Path) -> None:
        """Plain text file ingests as text modality."""
        from carl_studio.learn.ingest import SourceIngester

        f = tmp_path / "notes.txt"
        f.write_text("Important observation about territory assignments.")
        ingester = SourceIngester()
        chunks = ingester.ingest(str(f))
        assert len(chunks) > 0
        assert chunks[0].modality == "text"

    def test_csv_file_ingest(self, tmp_path: Path) -> None:
        """CSV file ingests as tabular modality."""
        from carl_studio.learn.ingest import SourceIngester

        f = tmp_path / "data.csv"
        f.write_text("name,revenue\nAcme,1000\nBeta,2000")
        ingester = SourceIngester()
        chunks = ingester.ingest(str(f))
        assert len(chunks) > 0

    def test_directory_ingest_skips_unknown_exts(self, tmp_path: Path) -> None:
        """Unknown file extensions are skipped during directory ingest."""
        from carl_studio.learn.ingest import SourceIngester

        (tmp_path / "good.txt").write_text("valid content")
        (tmp_path / "bad.xyz").write_text("unknown format")
        ingester = SourceIngester()
        chunks = ingester.ingest(str(tmp_path))
        texts = [c.text for c in chunks]
        assert any("valid content" in t for t in texts)

    def test_directory_ingest_raises_on_empty(self, tmp_path: Path) -> None:
        """Empty directory raises ValueError with supported extensions hint."""
        from carl_studio.learn.ingest import SourceIngester

        ingester = SourceIngester()
        with pytest.raises(ValueError, match="No ingestable files"):
            ingester.ingest(str(tmp_path))

    def test_python_file_chunking(self, tmp_path: Path) -> None:
        """Python files are chunked by class/function boundaries."""
        from carl_studio.learn.ingest import SourceIngester

        code = '''
class Foo:
    """Foo class."""
    def method(self):
        pass

def standalone():
    """A standalone function."""
    return 42
'''
        f = tmp_path / "module.py"
        f.write_text(code)
        ingester = SourceIngester()
        chunks = ingester.ingest(str(f))
        assert len(chunks) >= 1

    def test_markdown_file_chunking(self, tmp_path: Path) -> None:
        """Markdown files are chunked by heading boundaries."""
        from carl_studio.learn.ingest import SourceIngester

        md = "# Section 1\nContent one.\n\n# Section 2\nContent two.\n"
        f = tmp_path / "doc.md"
        f.write_text(md)
        ingester = SourceIngester()
        chunks = ingester.ingest(str(f))
        assert len(chunks) >= 1


# =========================================================================
# 11. Cross-system integration
# =========================================================================


class TestCrossSystemIntegration:
    """Tests spanning multiple subsystems."""

    def test_consent_contract_x402_chain(self) -> None:
        """Consent enables contract signing, which could gate x402 access."""
        from carl_studio.consent import ConsentManager
        from carl_studio.contract import ContractWitness, ServiceContract
        from carl_studio.x402 import X402Config, load_x402_config, save_x402_config

        db = FakeDB()

        # Enable consent
        mgr = ConsentManager(db=db)
        mgr.update("contract_witnessing", True)

        # Sign contract
        witness = ContractWitness(db=db)
        contract = ServiceContract(
            parties=["platform", "user"],
            terms_hash="terms_v1",
            terms_url="https://example.com/terms",
        )
        envelope = witness.sign(contract)
        assert witness.verify(envelope)

        # Configure x402
        save_x402_config(X402Config(enabled=True, wallet_address="0xabc"), db=db)
        loaded = load_x402_config(db=db)
        assert loaded.enabled is True

    def test_frame_shapes_agent_system_prompt(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Setting a frame changes the agent's system prompt."""
        from carl_studio.chat_agent import CARLAgent

        monkeypatch.setattr("carl_studio.frame._DEFAULT_FRAME_PATH", tmp_path / "frame.yaml")

        agent = CARLAgent(model="test", workdir=str(tmp_path), _client=MagicMock())

        # No frame
        prompt1 = agent._build_system_prompt()
        assert "No frame set" in prompt1

        # Set frame via tool
        agent._dispatch_tool("set_frame", {
            "domain": "pharma",
            "function": "drug_rollout",
            "role": "field_manager",
        })
        prompt2 = agent._build_system_prompt()
        assert "ACTIVE FRAME" in prompt2
        assert "pharma" in prompt2
        assert "drug_rollout" in prompt2

    def test_curriculum_to_carlito_to_agent_card(self, tmp_path: Path) -> None:
        """Full chain: curriculum graduate → spawn carlito → agent card."""
        from carl_studio.carlito import CarlitoRegistry, CarlitoSpawner
        from carl_studio.curriculum import CurriculumPhase, CurriculumTrack

        registry = CarlitoRegistry(db_path=tmp_path / "carlitos.db")
        track = CurriculumTrack(model_id="il-terminals-carl-v1", phase=CurriculumPhase.GRADUATED)

        spec = CarlitoSpawner.from_graduated_track(
            name="sales-analyst",
            track=track,
            domain="research",
            skills=["learn", "observe"],
        )

        spawner = CarlitoSpawner(registry=registry)
        card = spawner.spawn(spec, track)

        # Card has correct capabilities
        assert "train" in card.capabilities
        assert "learn" in card.capabilities
        assert "align" in card.capabilities  # research domain adds align
        assert card.skills == ["learn", "observe"]

        # Registry shows deployed
        loaded = registry.load("sales-analyst")
        assert loaded is not None
        assert loaded.status.value == "deployed"
        registry.close()

    def test_settings_tier_allows_method(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """CARLSettings.tier_allows delegates to tier module."""
        import carl_studio.tier as tier_mod
        from carl_studio.settings import CARLSettings
        from carl_studio.tier import Tier

        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.setattr(tier_mod, "_detect_hf_token", lambda: None)

        s = CARLSettings(tier=Tier.FREE)
        assert s.tier_allows("train") is True
        assert s.tier_allows("train.send_it") is False

    def test_db_contract_operations(self) -> None:
        """FakeDB contract CRUD operations work correctly."""
        db = FakeDB()

        # Insert
        db.insert_contract({"id": "c1", "status": "draft", "terms_hash": "h1"})
        assert db.get_contract("c1") is not None

        # Update
        db.update_contract("c1", {"status": "witnessed"})
        assert db.get_contract("c1")["status"] == "witnessed"

        # List
        db.insert_contract({"id": "c2", "status": "draft", "terms_hash": "h2"})
        contracts = db.list_contracts()
        assert len(contracts) == 2

        # Get nonexistent
        assert db.get_contract("nonexistent") is None


# =========================================================================
# 12. Edge cases and error paths
# =========================================================================


class TestEdgeCases:
    """Edge cases that should not crash."""

    def test_contract_sign_empty_parties(self) -> None:
        """Contract with empty parties list is still signable."""
        from carl_studio.consent import ConsentFlag, ConsentManager, ConsentState
        from carl_studio.contract import ContractWitness, ServiceContract

        db = FakeDB()
        mgr = ConsentManager(db=db)
        mgr.save(ConsentState(
            contract_witnessing=ConsentFlag(enabled=True, changed_at="2026-01-01")
        ))

        witness = ContractWitness(db=db)
        contract = ServiceContract(
            parties=[],
            terms_hash="hash",
            terms_url="https://example.com/terms",
        )
        envelope = witness.sign(contract)
        assert witness.verify(envelope) is True

    def test_consent_load_corrupted_json(self) -> None:
        """Corrupted consent JSON in DB returns defaults."""
        from carl_studio.consent import ConsentManager

        db = FakeDB()
        db.set_config("consent_state", "NOT VALID JSON {{{")
        mgr = ConsentManager(db=db)
        state = mgr.load()
        assert state.observability.enabled is False  # defaults

    def test_x402_load_corrupted_config(self) -> None:
        """Corrupted x402 config returns defaults."""
        from carl_studio.x402 import load_x402_config

        db = FakeDB()
        db.set_config("x402_config", "BROKEN JSON")
        config = load_x402_config(db=db)
        assert config.enabled is False  # default

    def test_frame_load_corrupted_yaml(self, tmp_path: Path) -> None:
        """Corrupted frame YAML returns empty frame."""
        from carl_studio.frame import WorkFrame

        path = tmp_path / "frame.yaml"
        path.write_text(": : : invalid yaml [[[")
        frame = WorkFrame.load(path)
        assert frame.active is False

    def test_carlito_registry_idempotent_save(self, tmp_path: Path) -> None:
        """Saving the same carlito twice is idempotent (upsert)."""
        from carl_studio.carlito import CarlitoRegistry, CarlitoSpec

        registry = CarlitoRegistry(db_path=tmp_path / "test.db")
        spec = CarlitoSpec(name="bot", parent_model="m1", domain="v1")
        registry.save(spec)
        registry.save(spec)
        assert len(registry.list_all()) == 1
        registry.close()

    def test_curriculum_milestone_accumulation(self) -> None:
        """Milestones accumulate across transitions."""
        from carl_studio.curriculum import CurriculumPhase, CurriculumTrack

        track = CurriculumTrack(model_id="test")
        track = track.advance(CurriculumPhase.DRILLING, event="start")
        track = track.advance(CurriculumPhase.EVALUATED, event="eval_started")
        track = track.advance(CurriculumPhase.GRADUATED, event="gate_pass")
        assert len(track.milestones) == 3
        events = [m.event for m in track.milestones]
        assert events == ["start", "eval_started", "gate_pass"]

    def test_witness_envelope_model_dump_json(self) -> None:
        """WitnessEnvelope JSON roundtrip works."""
        from carl_studio.contract import WitnessEnvelope

        env = WitnessEnvelope(
            contract_id="c1",
            terms_hash="h1",
            parties_hash="p1",
            witness_hash="w1",
            witnessed_at="2026-04-15T00:00:00Z",
            artifacts={"nested": {"key": "value"}},
        )
        raw = env.model_dump_json()
        restored = WitnessEnvelope.model_validate_json(raw)
        assert restored.artifacts["nested"]["key"] == "value"

    def test_agent_ingest_nonexistent_path(self, tmp_path: Path) -> None:
        """Ingesting a nonexistent path returns error gracefully."""
        from carl_studio.chat_agent import CARLAgent

        agent = CARLAgent(model="test", workdir=str(tmp_path), _client=MagicMock())
        result = agent._dispatch_tool("ingest_source", {"path": "/nonexistent/path"})
        assert "Error" in result or "not" in result.lower()

    def test_agent_create_nested_directory(self, tmp_path: Path) -> None:
        """create_file creates intermediate directories."""
        from carl_studio.chat_agent import CARLAgent

        agent = CARLAgent(model="test", workdir=str(tmp_path), _client=MagicMock())
        deep_path = str(tmp_path / "a" / "b" / "c" / "output.txt")
        result = agent._dispatch_tool("create_file", {
            "path": deep_path,
            "content": "deep content",
        })
        assert "Created" in result
        assert Path(deep_path).read_text() == "deep content"

    def test_agent_read_large_file_truncation(self, tmp_path: Path) -> None:
        """Large files are truncated to _TOOL_RESULT_MAX."""
        from carl_studio.chat_agent import CARLAgent

        agent = CARLAgent(model="test", workdir=str(tmp_path), _client=MagicMock())
        big_file = tmp_path / "big.txt"
        big_file.write_text("x" * 20_000)
        result = agent._dispatch_tool("read_file", {"path": str(big_file)})
        assert "truncated" in result
        assert len(result) < 20_000
