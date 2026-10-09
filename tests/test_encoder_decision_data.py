"""Offline decision input fidelity and lineage checks."""

from __future__ import annotations

import json

import pytest
from carl_core.hashing import content_hash

from carl_studio.data.adapters.decision import decision_example


def example(**changes):
    row = {
        "id": "episode-a",
        "prompt": "Tinny mode: single.\nChoose the latest required fact.\nKeep its source.",
        "task_input": {"required": ["fact"], "revision": 2},
        "evidence_class": "executed_finite_oracle",
        "verification": {"expected_output": "DO_NOT_INCLUDE_THIS_LABEL"},
    }
    row.update(changes.pop("row", {}))
    kwargs = {
        "observed_at": "2026-10-10T00:00:00+00:00",
        "source_schema": "decision/v1",
        "original_sources": ("original-task-source",),
        "feedback_ref": "finite-oracle:receipt-a",
    }
    kwargs.update(changes)
    return decision_example(row, {"revision": 2}, ({"revision": 1},), **kwargs)


def test_complete_instruction_and_candidate_task_are_preserved():
    value = example()
    assert value.query.parts[0].text == (
        "Tinny mode: single.\nChoose the latest required fact.\nKeep its source."
    )
    assert len(value.query.parts) == 1
    assert value.positive.parts[0].source_schema == "decision/v1"
    assert "DO_NOT_INCLUDE_THIS_LABEL" not in value.query.model_dump_json()
    assert json.loads(value.positive.parts[0].text) == {
        "task": {"required": ["fact"], "revision": 2},
        "action": {"revision": 2},
    }


def test_original_lineage_and_oracle_feedback_are_retained():
    value = example(relation="corrects")
    assert value.original_sources == (
        "original-task-source",
        content_hash({"required": ["fact"], "revision": 2}),
    )
    assert value.feedback_ref == "finite-oracle:receipt-a"
    assert value.relation == "corrects"
    assert example().relation == "corresponds"


def test_occurrences_preserve_shared_content_identity():
    first, second = example(), example(row={"id": "episode-b"})
    assert first.query.event_id != second.query.event_id
    assert first.query.content_id == second.query.content_id
    assert first.positive.content_id == second.positive.content_id
    assert first.original_sources == second.original_sources


def test_changed_task_changes_source_and_candidate_content():
    first = example()
    second = example(row={"task_input": '{"required":["other"],"revision":2}'})
    assert first.original_sources != second.original_sources
    assert first.query.content_id == second.query.content_id
    assert first.positive.content_id != second.positive.content_id


def test_task_family_is_projected_when_supported(monkeypatch):
    from carl_studio.semantic import learning

    class FamilyExample(learning.EncoderExample):
        task_family: str = "general"

    monkeypatch.setattr(learning, "EncoderExample", FamilyExample)
    assert example(row={"task_kind": "context-repair"}).task_family == "context-repair"


@pytest.mark.parametrize(
    "changes",
    [
        {"row": {"evidence_class": "declared"}},
        {"row": {"prompt": ""}},
        {"original_sources": ()},
        {"feedback_ref": ""},
        {"source_schema": ""},
        {"observed_at": "2026-10-10T00:00:00"},
        {"row": {"task_input": "[]"}},
    ],
)
def test_missing_source_and_supervision_boundaries_are_rejected(changes):
    with pytest.raises((ValueError, TypeError)):
        example(**changes)


def test_positive_cannot_be_an_explicit_negative():
    with pytest.raises(ValueError, match="distinct"):
        decision_example(
            {
                "id": "a",
                "prompt": "Choose",
                "task_input": {},
                "evidence_class": "executed_finite_oracle",
            },
            {"revision": 2},
            ({"revision": 2},),
            observed_at="2026-10-10T00:00:00+00:00",
            source_schema="decision/v1",
            original_sources=("source",),
            feedback_ref="finite-oracle:receipt",
        )
