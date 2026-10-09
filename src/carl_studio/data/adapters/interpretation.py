"""Offline dataset projection of consented, idempotently committed interpretations."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from carl_core.data_handles import DataRef, DataVault
from carl_core.hashing import content_hash
from carl_core.memory import MemoryLayer, MemoryStore

from carl_studio.data.types import Domain, Modality, UnifiedSample, Verification
from carl_studio.semantic.learning import EncoderExample
from carl_studio.semantic.types import Interpretation, SemanticInput, SemanticPart

from .base import DataAdapter


class InterpretationAdapter(DataAdapter):
    """Read captured local records; never download or execute training."""

    def load_and_adapt(self, split: str | None = None) -> list[UnifiedSample]:
        if split is not None and split != "unsplit":
            raise ValueError("Group captured episodes before requesting a dataset split")
        return list(self.adapt(self.records()))

    def records(self) -> list[dict[str, Any]]:
        root = Path(self.source.repo_id).expanduser()
        return [
            item.to_dict()
            for item in MemoryStore(root).list_active(layers={MemoryLayer.LONG})
            if item.metadata.get("capture_kind") == "accepted_interpretation"
        ]

    def _sources(self, raw: dict[str, Any]) -> tuple[Interpretation, DataVault, dict[str, DataRef]]:
        recorded = raw["metadata"].get("interpretation")
        record = (
            Interpretation.model_validate(recorded)
            if recorded is not None
            else Interpretation.model_validate_json(raw["content"])
        )
        artifacts: dict[str, DataRef] = {}
        vault = DataVault()
        for descriptor in raw["metadata"]["source_artifacts"]:
            ref = DataRef.model_validate(
                {key: value for key, value in descriptor.items() if key != "expires_at"}
            )
            vault.restore_file(ref)
            artifacts[str(ref.ref_id)] = ref
        return record, vault, artifacts

    def adapt(self, raw: list[dict[str, Any]]) -> Iterator[UnifiedSample]:
        for row in raw:
            record, vault, artifacts = self._sources(row)
            utterance = vault.read(artifacts[record.utterance_ref]).decode("utf-8")
            target = vault.read(artifacts[record.proposal_ref]).decode("utf-8")
            yield UnifiedSample(
                problem_statement=utterance,
                domain=Domain.INSTRUCTION,
                modality=Modality.TEXT,
                id=record.commit_key,
                prompt=[{"role": "user", "content": utterance}],
                source="local-capture:" + record.event_id,
                golden_solution=target,
                verification=Verification(expected_output=target),
                metadata={
                    "interpretation": record.model_dump(mode="json"),
                    "episode_id": record.event_id,
                    "source_artifacts": [ref.describe() for ref in artifacts.values()],
                },
            )

    def encoder_examples(self) -> list[EncoderExample]:
        """A rejected predecessor supplies an explicit negative for a correction."""
        result: list[EncoderExample] = []
        for raw in self.records():
            negative = raw["metadata"].get("rejected_proposal_ref")
            if not negative:
                continue
            record, vault, artifacts = self._sources(raw)

            def input_for(
                ref_id: str,
                recipe: str,
                vault: DataVault = vault,
                artifacts: dict[str, DataRef] = artifacts,
            ) -> SemanticInput:
                return SemanticInput(
                    event_id=ref_id,
                    recipe=recipe,
                    parts=(
                        SemanticPart(
                            modality="text", text=vault.read(artifacts[ref_id]).decode("utf-8")
                        ),
                    ),
                )

            result.append(
                EncoderExample(
                    id=record.commit_key,
                    episode_id=record.event_id,
                    original_sources=(
                        artifacts[record.utterance_ref].sha256
                        or content_hash(record.utterance_ref),
                    ),
                    observed_at=raw["created_at"],
                    feedback_ref=record.feedback_ref or "",
                    query=SemanticInput(
                        event_id=record.event_id + ":query",
                        recipe="SearchQuery",
                        parts=tuple(
                            SemanticPart(
                                modality="text", text=vault.read(artifacts[ref_id]).decode("utf-8")
                            )
                            for ref_id in (
                                record.utterance_ref,
                                record.context_ref,
                                record.goal_ref,
                            )
                        ),
                    ),
                    positive=input_for(record.proposal_ref, "Document"),
                    negatives=(input_for(negative, "Document"),),
                    relation="corrects",
                    action_label=vault.read(artifacts[record.proposal_ref]).decode("utf-8"),
                    source_artifacts=artifacts,
                )
            )
        return result
