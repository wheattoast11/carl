"""Local projection of explicitly supervised decisions into encoder examples."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal

from carl_core.encoder import SemanticInput, SemanticPart
from carl_core.hashing import canonical_json, content_hash

if TYPE_CHECKING:
    from carl_studio.semantic.learning import EncoderExample


def decision_example(
    row: Mapping[str, Any],
    positive: Mapping[str, Any],
    negatives: Sequence[Mapping[str, Any]],
    *,
    observed_at: str,
    source_schema: str,
    original_sources: Sequence[str],
    feedback_ref: str,
    relation: Literal["supports", "contradicts", "corresponds", "corrects"] = "corresponds",
) -> EncoderExample:
    """Preserve source inputs; the caller must supply oracle-confirmed alternatives."""
    from carl_studio.semantic.learning import EncoderExample

    if row.get("evidence_class") != "executed_finite_oracle":
        raise ValueError("Executed finite-oracle provenance required")
    identity, prompt = row.get("id"), row.get("prompt")
    if not isinstance(identity, str) or not identity.strip():
        raise ValueError("Decision occurrence identity required")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Complete original prompt required")
    if not source_schema.strip() or not feedback_ref.strip():
        raise ValueError("Source schema and explicit oracle feedback reference required")
    if not original_sources or any(not source.strip() for source in original_sources):
        raise ValueError("Original source lineage required")
    if datetime.fromisoformat(observed_at).tzinfo is None:
        raise ValueError("Observation time requires a timezone")
    task = row.get("task_input")
    if isinstance(task, str):
        task = json.loads(task)
    if not isinstance(task, dict):
        raise TypeError("Structured original task required")
    if not negatives:
        raise ValueError("Explicit oracle-rejected alternatives required")
    target = canonical_json(dict(positive))
    alternatives = [canonical_json(dict(value)) for value in negatives]
    if target in alternatives or len(set(alternatives)) != len(alternatives):
        raise ValueError("Alternatives must be distinct from the positive and each other")

    def candidate(value: Mapping[str, Any], occurrence: str) -> SemanticInput:
        return SemanticInput(
            event_id=identity + ":" + occurrence,
            recipe="Document",
            parts=(
                SemanticPart(
                    modality="structured",
                    text=canonical_json({"task": task, "action": dict(value)}),
                    source_schema=source_schema,
                ),
            ),
        )

    family = row.get("task_kind", "general")
    if not isinstance(family, str) or not family.strip():
        raise ValueError("Task family must be a nonempty string")
    family_fields: dict[str, Any] = (
        {"task_family": family} if "task_family" in EncoderExample.model_fields else {}
    )
    return EncoderExample(
        **family_fields,
        id=identity,
        episode_id=row.get("episode_id", identity),
        original_sources=tuple(dict.fromkeys((*original_sources, content_hash(task)))),
        observed_at=observed_at,
        feedback_ref=feedback_ref,
        query=SemanticInput(
            event_id=identity + ":query",
            recipe="SearchQuery",
            parts=(SemanticPart(modality="text", text=prompt),),
        ),
        positive=candidate(positive, "positive"),
        negatives=tuple(candidate(value, "negative" + str(i)) for i, value in enumerate(negatives)),
        relation=relation,
        action_label=target,
    )
