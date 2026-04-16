"""Tests for carl_studio.contract -- ServiceContract, ContractWitness, WitnessEnvelope."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from carl_studio.contract import (
    ContractError,
    ContractWitness,
    ServiceContract,
    WitnessEnvelope,
)


class FakeDB:
    """In-memory mock of LocalDB for contract tests."""

    def __init__(self) -> None:
        self.contracts: dict[str, dict] = {}
        self.config: dict[str, str] = {}

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

    def get_config(self, key: str, default: str | None = None) -> str | None:
        return self.config.get(key, default)

    def set_config(self, key: str, value: str) -> None:
        self.config[key] = value


class TestServiceContract:
    def test_defaults(self) -> None:
        ct = ServiceContract(parties=["alice"], terms_hash="abc", terms_url="https://t.co/terms")
        assert ct.status == "draft"
        assert ct.id != ""
        assert ct.signed_at is None
        assert ct.witness_hash is None

    def test_parties_list(self) -> None:
        ct = ServiceContract(parties=["alice", "bob"], terms_hash="h", terms_url="u")
        assert len(ct.parties) == 2


class TestContractWitness:
    def test_hash_terms_deterministic(self) -> None:
        h1 = ContractWitness.hash_terms("hello world")
        h2 = ContractWitness.hash_terms("hello world")
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex digest

    def test_hash_terms_different_inputs(self) -> None:
        h1 = ContractWitness.hash_terms("hello")
        h2 = ContractWitness.hash_terms("world")
        assert h1 != h2

    def test_sign_creates_envelope(self) -> None:
        db = FakeDB()
        # Enable consent for contract witnessing
        from carl_studio.consent import ConsentFlag, ConsentManager, ConsentState

        mgr = ConsentManager(db=db)
        state = ConsentState(
            contract_witnessing=ConsentFlag(enabled=True, changed_at="2026-01-01T00:00:00Z")
        )
        mgr.save(state)

        witness = ContractWitness(db=db)
        contract = ServiceContract(
            parties=["alice", "bob"],
            terms_hash="abc123",
            terms_url="https://carl.camp/terms",
        )
        envelope = witness.sign(contract)

        assert envelope.contract_id == contract.id
        assert envelope.terms_hash == "abc123"
        assert envelope.witness_hash != ""
        assert envelope.witnessed_at != ""
        # sign() no longer mutates — check DB state instead
        stored = db.contracts.get(contract.id)
        assert stored is not None
        assert stored["status"] == "witnessed"

    def test_sign_without_consent_raises(self) -> None:
        db = FakeDB()
        # Default consent is all off
        witness = ContractWitness(db=db)
        contract = ServiceContract(
            parties=["alice"],
            terms_hash="abc",
            terms_url="https://carl.camp/terms",
        )
        with pytest.raises(ContractError, match="consent"):
            witness.sign(contract)

    def test_verify_valid_envelope(self) -> None:
        db = FakeDB()
        from carl_studio.consent import ConsentFlag, ConsentManager, ConsentState

        mgr = ConsentManager(db=db)
        mgr.save(ConsentState(
            contract_witnessing=ConsentFlag(enabled=True, changed_at="2026-01-01")
        ))

        witness = ContractWitness(db=db)
        contract = ServiceContract(
            parties=["alice"],
            terms_hash="myhash",
            terms_url="https://carl.camp/terms",
        )
        envelope = witness.sign(contract)
        assert witness.verify(envelope) is True

    def test_verify_tampered_envelope(self) -> None:
        db = FakeDB()
        from carl_studio.consent import ConsentFlag, ConsentManager, ConsentState

        mgr = ConsentManager(db=db)
        mgr.save(ConsentState(
            contract_witnessing=ConsentFlag(enabled=True, changed_at="2026-01-01")
        ))

        witness = ContractWitness(db=db)
        contract = ServiceContract(
            parties=["alice"],
            terms_hash="myhash",
            terms_url="https://carl.camp/terms",
        )
        envelope = witness.sign(contract)
        # Tamper with the terms hash
        tampered = envelope.model_copy(update={"terms_hash": "TAMPERED"})
        assert witness.verify(tampered) is False

    def test_fetch_terms_success(self) -> None:
        resp = MagicMock()
        resp.read.return_value = b"Terms of Service v1.0"
        resp.__enter__ = lambda s: s
        resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=resp):
            content = ContractWitness.fetch_terms("https://carl.camp/terms")
        assert content == "Terms of Service v1.0"

    def test_fetch_terms_network_error(self) -> None:
        import urllib.error

        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError("connection refused"),
        ):
            with pytest.raises(ContractError, match="Network"):
                ContractWitness.fetch_terms("https://carl.camp/terms")

    def test_sign_stores_in_db(self) -> None:
        db = FakeDB()
        from carl_studio.consent import ConsentFlag, ConsentManager, ConsentState

        mgr = ConsentManager(db=db)
        mgr.save(ConsentState(
            contract_witnessing=ConsentFlag(enabled=True, changed_at="2026-01-01")
        ))

        witness = ContractWitness(db=db)
        contract = ServiceContract(
            parties=["alice"],
            terms_hash="abc",
            terms_url="https://carl.camp/terms",
        )
        envelope = witness.sign(contract)
        assert contract.id in db.contracts
        stored = db.contracts[contract.id]
        assert stored["status"] == "witnessed"
        assert stored["witness_hash"] == envelope.witness_hash


class TestWitnessEnvelope:
    def test_model_dump(self) -> None:
        env = WitnessEnvelope(
            contract_id="c1",
            terms_hash="h1",
            parties_hash="p1",
            witness_hash="w1",
            witnessed_at="2026-04-15T00:00:00Z",
        )
        d = env.model_dump()
        assert d["contract_id"] == "c1"

    def test_round_trip_json(self) -> None:
        env = WitnessEnvelope(
            contract_id="c1",
            terms_hash="h1",
            parties_hash="p1",
            witness_hash="w1",
            witnessed_at="2026-04-15T00:00:00Z",
            artifacts={"key": "value"},
        )
        raw = env.model_dump_json()
        restored = WitnessEnvelope.model_validate_json(raw)
        assert restored.artifacts["key"] == "value"
