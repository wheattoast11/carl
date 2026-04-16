"""Chain-agnostic service agreement primitive with local witness hashing.

The witness is a deterministic hash (not a cryptographic signature initially)
that proves "this contract was acknowledged at this time with these parties."

On-chain inscription is deferred — the ``chain`` field exists for future use.
"""

from __future__ import annotations

import hashlib
import json
import urllib.error
import urllib.request
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class ContractError(Exception):
    """Raised when contract operations fail."""


class ContractStatus(str, Enum):
    """Lifecycle status of a service contract."""

    DRAFT = "draft"
    SIGNED = "signed"
    WITNESSED = "witnessed"


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class ServiceContract(BaseModel):
    """A service agreement between parties."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    parties: list[str] = Field(default_factory=list)
    terms_hash: str = ""
    terms_url: str = ""
    created_at: str = Field(default_factory=_now_iso)
    signed_at: str | None = None
    witness_hash: str | None = None
    chain: str | None = None
    status: ContractStatus = ContractStatus.DRAFT


class WitnessEnvelope(BaseModel):
    """Wraps a contract with its witness signature and timestamp."""

    contract_id: str
    terms_hash: str
    parties_hash: str
    witness_hash: str
    witnessed_at: str
    artifacts: dict[str, Any] = Field(default_factory=dict)


class ContractWitness:
    """Signs a hash of the contract + runtime artifacts, stores locally."""

    def __init__(self, db: Any | None = None) -> None:
        self._db = db

    def _get_db(self) -> Any:
        if self._db is not None:
            return self._db
        from carl_studio.db import LocalDB

        self._db = LocalDB()
        return self._db

    @staticmethod
    def hash_terms(content: str) -> str:
        """SHA-256 hash of terms document content."""
        return hashlib.sha256(content.encode()).hexdigest()

    @staticmethod
    def fetch_terms(terms_url: str, timeout: int = 10) -> str:
        """Fetch terms document from URL using urllib."""
        req = urllib.request.Request(terms_url, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read().decode()
        except urllib.error.HTTPError as exc:
            raise ContractError(
                f"Failed to fetch terms ({exc.code})"
            ) from exc
        except urllib.error.URLError as exc:
            raise ContractError(f"Network error: {exc.reason}") from exc

    def sign(
        self,
        contract: ServiceContract,
        artifacts: dict[str, Any] | None = None,
    ) -> WitnessEnvelope:
        """Create a deterministic witness hash and store the envelope locally.

        Checks contract witnessing consent before proceeding.
        """
        try:
            from carl_studio.consent import ConsentManager

            consent = ConsentManager(db=self._db).load()
            if not consent.contract_witnessing.enabled:
                raise ContractError(
                    "Contract witnessing consent is not enabled. "
                    "Enable with: carl camp consent update contract_witnessing --enable"
                )
        except ImportError:
            pass  # consent module not available — proceed

        artifacts = artifacts or {}
        ts = _now_iso()
        parties_hash = hashlib.sha256(
            json.dumps(sorted(contract.parties)).encode()
        ).hexdigest()
        artifact_json = json.dumps(artifacts, sort_keys=True)

        witness_input = (
            f"{contract.terms_hash}|{parties_hash}|{artifact_json}|{ts}"
        )
        witness_hash = hashlib.sha256(witness_input.encode()).hexdigest()

        envelope = WitnessEnvelope(
            contract_id=contract.id,
            terms_hash=contract.terms_hash,
            parties_hash=parties_hash,
            witness_hash=witness_hash,
            witnessed_at=ts,
            artifacts=artifacts,
        )

        # Persist witnessed contract (immutable — no in-place mutation)
        db = self._get_db()
        db.insert_contract({
            "id": contract.id,
            "parties": json.dumps(contract.parties),
            "terms_hash": contract.terms_hash,
            "terms_url": contract.terms_url,
            "status": ContractStatus.WITNESSED.value,
            "signed_at": ts,
            "witness_hash": witness_hash,
            "chain": contract.chain,
            "envelope": envelope.model_dump_json(),
        })

        return envelope

    def verify(self, envelope: WitnessEnvelope) -> bool:
        """Verify the witness hash matches the contract data.

        Re-computes the expected hash from the envelope's fields.
        """
        artifact_json = json.dumps(envelope.artifacts, sort_keys=True)
        expected_input = (
            f"{envelope.terms_hash}|{envelope.parties_hash}"
            f"|{artifact_json}|{envelope.witnessed_at}"
        )
        expected = hashlib.sha256(expected_input.encode()).hexdigest()
        return expected == envelope.witness_hash

    def get_contract(self, contract_id: str) -> ServiceContract | None:
        """Load a contract from local storage."""
        db = self._get_db()
        row = db.get_contract(contract_id)
        if row is None:
            return None
        return ServiceContract(
            id=row["id"],
            parties=json.loads(row["parties"]) if isinstance(row["parties"], str) else row["parties"],
            terms_hash=row["terms_hash"],
            terms_url=row["terms_url"],
            status=row["status"],
            signed_at=row.get("signed_at"),
            witness_hash=row.get("witness_hash"),
            chain=row.get("chain"),
        )

    def get_envelope(self, contract_id: str) -> WitnessEnvelope | None:
        """Load a witness envelope from local storage."""
        db = self._get_db()
        row = db.get_contract(contract_id)
        if row is None:
            return None
        raw = row.get("envelope")
        if not raw:
            return None
        return WitnessEnvelope.model_validate_json(raw)

    def list_contracts(self, limit: int = 20) -> list[ServiceContract]:
        """List locally stored contracts."""
        db = self._get_db()
        rows = db.list_contracts(limit=limit)
        result = []
        for row in rows:
            result.append(ServiceContract(
                id=row["id"],
                parties=json.loads(row["parties"]) if isinstance(row["parties"], str) else row["parties"],
                terms_hash=row["terms_hash"],
                terms_url=row["terms_url"],
                status=row["status"],
                signed_at=row.get("signed_at"),
                witness_hash=row.get("witness_hash"),
                chain=row.get("chain"),
            ))
        return result
