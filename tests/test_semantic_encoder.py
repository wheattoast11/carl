"""Offline counterexamples for semantic contracts and owner integration."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from carl_core.hashing import content_hash
from carl_core.interaction import ActionType, InteractionChain, Step
from carl_core.memory import MemoryLayer, MemoryStore

from carl_studio.knowledge_store import KnowledgeStore
from carl_studio.semantic.learning import RepresentationMeasurement, representation_acceptance
from carl_studio.semantic.types import Carrier, EncoderBinding, ExecutionBinding, Interpretation
from carl_studio.session import Session


def binding():
    return EncoderBinding(artifact_sha256="a" * 64, processor_sha256="b" * 64)


def runtime():
    return ExecutionBinding(
        interpreter="/offline/python",
        interpreter_sha256="c" * 64,
        dependencies={},
        processor_sha256="b" * 64,
    )


def carrier(axis=0):
    values = [0.01] * 768
    values[axis] = 1
    return Carrier(values=tuple(values))


def test_raw_restriction_tail_and_scoring():
    vector = Carrier(values=tuple(float(i + 1) for i in range(768)))
    for dimension in (128, 256, 512, 768):
        assert vector.prefix(dimension) + vector.values[dimension:] == vector.values
        assert vector.prefix(768)[:dimension] == vector.prefix(dimension)
        assert vector.score(vector, dimension) == pytest.approx(1)
    with pytest.raises(ValueError):
        Carrier(values=(0.0,) * 768)
    with pytest.raises(ValueError):
        Carrier(values=(float("nan"),) * 768)


def test_shared_encoding_zero_overlap_and_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("CARL_HOME", str(tmp_path))
    session = Session()
    calls = []

    def encode(request):
        calls.append(request.content_id)
        return carrier(1 if request.parts[0].text == "unrelated" else 0)

    session.semantic.bind(binding(), runtime(), encode)
    store = KnowledgeStore(
        chunks=[
            {"source": "one", "text": "automobile", "words": {"automobile"}},
            {"source": "two", "text": "unrelated", "words": {"unrelated"}},
        ]
    )
    result = store.semantic_recall("car", session.semantic)
    assert result[0]["source_ref"] == "one"
    assert result[0]["lexical_score"] == 0
    assert result[0]["mode"] == "semantic"
    store.semantic_recall("car", session.semantic)
    assert len(calls) == 3
    restored = Session.restore(session.snapshot())
    assert not restored.restore_issues
    assert len(restored.data_vault.list_refs()) == 3
    session.semantic.encoder = None
    assert store.semantic_recall("automobile", session.semantic)[0]["mode"] == "lexical"


def test_legacy_and_semantic_serialization():
    step = Step(ActionType.DATA_TRANSFORM, "encode")
    assert "semantic_refs" not in step.to_dict()
    step.semantic_refs = {"artifact_ref": "opaque"}
    chain = InteractionChain(steps=[step])
    assert InteractionChain.from_dict(chain.to_dict()).steps[0].semantic_refs == step.semantic_refs


def test_concurrent_consolidation_restart_and_key(tmp_path):
    key = content_hash("accepted event")

    def commit(_):
        return (
            MemoryStore(tmp_path).commit_once(key, "accepted", metadata={"feedback_ref": "yes"}).id
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        assert len(set(pool.map(commit, range(32)))) == 1
    assert len(MemoryStore(tmp_path).list_layer(MemoryLayer.LONG)) == 1
    with pytest.raises(ValueError):
        MemoryStore(tmp_path).commit_once(key, "changed")


def test_silence_and_goal_identity():
    record = Interpretation(
        id="one",
        event_id="event",
        utterance_ref="u",
        context_ref="c",
        goal_ref="g",
        proposal_ref="p",
    )
    with pytest.raises(ValueError):
        _ = record.commit_key
    confirmed = Interpretation.model_validate(
        {
            **record.model_dump(),
            "feedback": "confirmed",
            "feedback_ref": "f",
            "confirmed_artifact": "a",
        }
    )
    changed = confirmed.model_copy(update={"goal_ref": "other"})
    assert confirmed.commit_key != changed.commit_key


def measurement(accuracy, **updates):
    return RepresentationMeasurement(
        **{
            "population": "p",
            "processing": "r",
            "tasks": "t",
            "budget": "b",
            "action_accuracy": accuracy,
            "dimensions": 768,
            "finite": True,
            "positive_negative_margin": 0.3,
            "variance": 0.1,
            "slices": {"text:128": 0.8},
            "policies": {"local": True},
            "resources_passed": True,
            "updated_parameters": ("head.weight",),
            **updates,
        }
    )


@pytest.mark.parametrize(
    "updates, status",
    [
        ({}, "accepted"),
        ({"variance": 0}, "rejected"),
        ({"updated_parameters": ()}, "rejected"),
        ({"positive_negative_margin": -1}, "rejected"),
        ({"slices": {"text:128": 0.7}}, "rejected"),
        ({"slices": {}}, "inconclusive"),
        ({"population": "different"}, "inconclusive"),
        ({"resources_passed": False}, "rejected"),
    ],
)
def test_acceptance_falsifiers(updates, status):
    assert (
        representation_acceptance(
            measurement(0.7), measurement(0.9, **updates), {"text:128"}, {"local"}
        )["status"]
        == status
    )


def test_feedback_atomicity_scope_and_restart(monkeypatch, tmp_path):
    import importlib

    from carl_studio.consent import ConsentManager

    monkeypatch.setattr(
        importlib.import_module("carl_studio.settings"), "carl_home", lambda: tmp_path
    )
    monkeypatch.setattr(ConsentManager, "capture_allowed", lambda *args, **kwargs: True)
    session = Session(workspace=str(tmp_path))
    refs = [
        str(session.data_vault.put_bytes(value.encode()).ref_id)
        for value in (
            "utterance",
            "context",
            "goal",
            "proposal",
            "correction",
            "feedback",
            "confirmed",
        )
    ]
    record = {
        "id": "first",
        "event_id": "episode",
        "utterance_ref": refs[0],
        "context_ref": refs[1],
        "goal_ref": refs[2],
        "proposal_ref": refs[3],
    }
    session.semantic.interpret(record)
    with pytest.raises(ValueError):
        session.semantic.interpretation_feedback(
            "first", refs[5], True, correction={**record, "id": "next", "proposal_ref": refs[4]}
        )
    assert session.semantic.interpretations["first"].feedback == "unconfirmed"
    assert session.chain.steps[-1].success is False
    with pytest.raises(Exception, match="unavailable"):
        session.semantic.interpretation_feedback("first", refs[5], True, "not-a-ref")
    correction = {**record, "id": "next", "proposal_ref": refs[4]}
    result = session.semantic.interpretation_feedback("first", refs[5], True, refs[6], correction)
    assert result["supersedes"] == "first"
    restored = Session.restore(session.snapshot())
    assert not restored.restore_issues
    repeated = restored.semantic.interpretation_feedback(
        "first", refs[5], True, refs[6], correction
    )
    assert repeated["memory_id"] == result["memory_id"]
    assert len(MemoryStore(tmp_path / "memory").list_layer(MemoryLayer.LONG)) == 1


def test_mcp_event_budget_not_session_budget():
    session = Session()
    refs = [str(session.data_vault.put_bytes(b"source").ref_id) for _ in range(4)]
    record = {
        "event_id": "turn",
        "utterance_ref": refs[0],
        "context_ref": refs[1],
        "goal_ref": refs[2],
        "proposal_ref": refs[3],
    }
    session.semantic.interpret({**record, "id": "one"})
    session.semantic.interpret({**record, "id": "two"})
    with pytest.raises(Exception, match="budget"):
        session.semantic.interpret({**record, "id": "three"})
    session.semantic.interpret({**record, "id": "three", "event_id": "next-turn"})


def test_stale_and_missing_restored_artifacts(monkeypatch, tmp_path):
    import importlib

    monkeypatch.setattr(
        importlib.import_module("carl_studio.settings"), "carl_home", lambda: tmp_path
    )
    session = Session()
    session.semantic.bind(binding(), runtime(), lambda _: carrier())
    descriptor = session.semantic.encode_data(
        {"event_id": "one", "parts": [{"modality": "text", "text": "hello"}]}
    )["artifact"]
    checkpoint = session.snapshot()
    from urllib.parse import urlparse

    Path(urlparse(descriptor["uri"]).path).write_bytes(b"changed")
    restored = Session.restore(checkpoint)
    assert restored.restore_issues
    assert not restored.data_vault.list_refs()


def test_rebinding_drops_heads_and_cache_checks_bindings(monkeypatch, tmp_path):
    import importlib

    monkeypatch.setattr(
        importlib.import_module("carl_studio.settings"), "carl_home", lambda: tmp_path
    )
    session = Session()
    session.semantic.bind(binding(), runtime(), lambda _: carrier())
    data = {"event_id": "source", "parts": [{"modality": "text", "text": "source"}]}
    session.semantic.encode_data(data)

    def stale():
        raise ValueError("stale binding")

    session.semantic.validate_binding = stale
    with pytest.raises(ValueError, match="stale"):
        session.semantic.encode_data(data)
    session.semantic.head_weights = (2.0,) * 768
    session.semantic.bind(binding(), runtime(), lambda _: carrier())
    assert session.semantic.head_weights is None
    assert session.semantic.validate_binding is None


def test_capture_after_confirmation_keeps_rejected_predecessor(monkeypatch, tmp_path):
    import importlib

    from carl_studio.consent import ConsentManager
    from carl_studio.data.adapters.interpretation import InterpretationAdapter
    from carl_studio.data.types import DataSource

    monkeypatch.setattr(
        importlib.import_module("carl_studio.settings"), "carl_home", lambda: tmp_path
    )
    grant = {"enabled": False}
    monkeypatch.setattr(ConsentManager, "capture_allowed", lambda *args, **kwargs: grant["enabled"])
    session = Session()
    refs = [
        str(session.data_vault.put_bytes(text.encode()).ref_id)
        for text in (
            "utterance",
            "context",
            "goal",
            "wrong proposal",
            "correct proposal",
            "explicit feedback",
            "artifact",
        )
    ]
    record = {
        "id": "old",
        "event_id": "episode",
        "utterance_ref": refs[0],
        "context_ref": refs[1],
        "goal_ref": refs[2],
        "proposal_ref": refs[3],
    }
    session.semantic.interpret(record)
    session.semantic.interpretation_feedback(
        "old", refs[5], True, refs[6], {**record, "id": "new", "proposal_ref": refs[4]}
    )
    grant["enabled"] = True
    session.semantic.interpretation_feedback("new", refs[5], True, refs[6])
    source = DataSource(
        name="capture",
        repo_id=str(tmp_path / "memory"),
        adapter="local",
        domain="instruction",
        modality="text",
        description="test",
        license="local-capture",
    )
    adapter = InterpretationAdapter(source)
    rows = adapter.encoder_examples()
    assert len(rows) == 1
    assert rows[0].positive.parts[0].text == "correct proposal"
    assert rows[0].negatives[0].parts[0].text == "wrong proposal"
    assert len(adapter.load_and_adapt()) == 1


def test_commit_after_torn_tail(tmp_path):
    path = tmp_path / "long.jsonl"
    path.write_text('{"partial":')
    store = MemoryStore(tmp_path)
    result = store.commit_once(content_hash("event"), "accepted")
    assert [row.id for row in store.list_layer(MemoryLayer.LONG)] == [result.id]
    assert store.commit_once(content_hash("event"), "accepted").id == result.id


def test_space_identity_excludes_view_capabilities():
    original = binding()
    changed = original.model_copy(update={"dimensions": (256, 768), "recipes": ("OtherRecipe",)})
    assert original.space_id == changed.space_id
    custom = EncoderBinding(
        model="custom",
        revision="local",
        native_dimension=384,
        dimensions=(128, 256, 384),
        artifact_sha256="a" * 64,
        processor_sha256="b" * 64,
    )
    assert custom.native_dimension == 384
    vector = Carrier(values=(1.0,) * 384, dimensions=custom.dimensions)
    assert vector.score(vector, 384) == pytest.approx(1.0)


def test_capture_all_off_revokes_scopes():
    from carl_studio.consent import ConsentManager

    class DB:
        def __init__(self):
            self.values = {}

        def get_config(self, key):
            return self.values.get(key)

        def set_config(self, key, value):
            self.values[key] = value

    manager = ConsentManager(DB())
    manager.capture_grant(workspace="/workspace", enabled=True)
    assert manager.capture_allowed(session_id="session", workspace="/workspace")
    manager.all_off()
    assert not manager.capture_allowed(session_id="session", workspace="/workspace")
    manager.capture_grant(session_id="session", enabled=True)
    assert manager.capture_allowed(session_id="session", workspace="/workspace")


def test_a2a_preserves_structured_and_media_order():
    from carl_studio.a2a.spec import message_send_to_task

    parts = [
        {"kind": "text", "text": "before"},
        {"kind": "file", "file": {"uri": "carl-data://ref/image"}},
        {"kind": "data", "data": {"schema": "action/v1", "name": "copy"}},
        {"kind": "text", "text": "after"},
    ]
    result = message_send_to_task({"message": {"parts": parts}})
    assert result["inputs"]["parts"] == parts
    assert result["inputs"]["text"] == "beforeafter"


@pytest.mark.asyncio
async def test_a2a_uses_shared_session_and_dispatcher_denial(tmp_path):
    from carl_studio.a2a.bus import LocalBus
    from carl_studio.a2a.connection import A2AServerConnection
    from carl_studio.tool_dispatcher import ToolPermission

    session = Session()
    refs = [str(session.data_vault.put_bytes(b"source").ref_id) for _ in range(4)]
    record = {
        "id": "candidate",
        "event_id": "episode",
        "utterance_ref": refs[0],
        "context_ref": refs[1],
        "goal_ref": refs[2],
        "proposal_ref": refs[3],
    }
    bus = LocalBus(tmp_path / "a2a.db")
    connection = A2AServerConnection(bus=bus, session=session)
    await connection.open()
    response = await connection.handle_request(
        {"id": 1, "method": "interpret", "params": {"record": record}}
    )
    assert response["result"]["feedback"] == "unconfirmed"
    assert connection.semantic_session is session
    await connection.close()
    denied = A2AServerConnection(
        bus=bus, session=session, semantic_permission=lambda *args: ToolPermission.DENY
    )
    await denied.open()
    response = await denied.handle_request(
        {"id": 2, "method": "interpret", "params": {"record": {**record, "id": "denied"}}}
    )
    assert "error" in response
    assert "denied" not in session.semantic.interpretations
    await denied.close()
    bus.close()


def test_same_file_semantic_chunks_keep_occurrence_and_provenance(monkeypatch):
    from types import SimpleNamespace

    from carl_studio.chat_agent import CARLAgent

    chunks = [
        {"source": "notes.txt", "text": "chosen first chunk", "words": {"chosen"}},
        {"source": "notes.txt", "text": "wrong second chunk", "words": {"wrong"}},
    ]
    store = KnowledgeStore(chunks=chunks)

    def recall(query, sources, *, limit):
        assert [row[0] for row in sources] == ["knowledge:0", "knowledge:1"]
        return [{"source_ref": sources[0][0], "semantic_score": 0.9, "lexical_score": 1.0}]

    semantic = SimpleNamespace(configured=True, recall=recall)
    agent = CARLAgent.__new__(CARLAgent)
    agent._knowledge_store = store
    monkeypatch.setattr(agent, "_runtime_session", lambda: SimpleNamespace(semantic=semantic))
    matches = store.semantic_recall("chosen", semantic)
    assert matches[0]["source_ref"] == "notes.txt"
    assert matches[0]["chunk_ref"] == "knowledge:0"
    answer = agent._tool_query("chosen")
    assert "chosen first chunk" in answer
    assert "wrong second chunk" not in answer
    semantic.configured = False
    assert "chosen first chunk" in agent._tool_query("chosen")
