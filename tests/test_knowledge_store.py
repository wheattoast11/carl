"""Unit tests for :class:`carl_studio.knowledge_store.KnowledgeStore`.

Locks in the BC contract the v0.7 Phase-2 extraction promised:

* ``chunks`` is a mutable ``list[dict[str, Any]]`` (tests + downstream
  callers reach in and mutate directly),
* ``enforce_cap`` evicts oldest-first and logs at most one warning per
  instance,
* ``recall`` returns overlap-scored ``(score, chunk)`` tuples,
* ``to_list`` / ``from_list`` round-trip the ``set``-vs-``list`` shape
  the chat_agent session save/load path used to do inline.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

from carl_studio.knowledge_store import (
    KnowledgeChunk,
    KnowledgeStore,
    _KNOWLEDGE_MAX_CHUNKS,
)


class TestKnowledgeChunk:
    """Typed constructor helper — not the main API, but used at ingest sites."""

    def test_to_dict_materializes_set_words(self) -> None:
        chunk = KnowledgeChunk(
            text="hello world",
            source="s.txt",
            words={"hello", "world"},
        )
        out = chunk.to_dict()
        assert out["text"] == "hello world"
        assert out["source"] == "s.txt"
        assert isinstance(out["words"], set)
        assert out["words"] == {"hello", "world"}

    def test_to_dict_drops_empty_meta(self) -> None:
        chunk = KnowledgeChunk(text="t", source="s", words={"t"})
        assert "meta" not in chunk.to_dict()

    def test_to_dict_keeps_non_empty_meta(self) -> None:
        chunk = KnowledgeChunk(text="t", source="s", meta={"modality": "text"})
        out = chunk.to_dict()
        assert out["meta"] == {"modality": "text"}
        # Defensive copy — mutating the chunk's meta after the fact must
        # not leak into a previously returned dict.
        chunk.meta["modality"] = "image"
        assert out["meta"] == {"modality": "text"}

    def test_from_dict_list_words_become_set(self) -> None:
        chunk = KnowledgeChunk.from_dict(
            {"text": "t", "source": "s", "words": ["a", "b"]},
        )
        assert chunk.words == {"a", "b"}
        assert isinstance(chunk.words, set)

    def test_from_dict_garbled_words_degrade_to_empty_set(self) -> None:
        chunk = KnowledgeChunk.from_dict(
            {"text": "t", "source": "s", "words": "not-a-list"},  # type: ignore[dict-item]
        )
        assert chunk.words == set()

    def test_from_dict_missing_fields_default(self) -> None:
        chunk = KnowledgeChunk.from_dict({})
        assert chunk.text == ""
        assert chunk.source == ""
        assert chunk.words == set()
        assert chunk.meta == {}


class TestKnowledgeStoreAppendAndEvict:
    def test_default_cap_constant(self) -> None:
        assert _KNOWLEDGE_MAX_CHUNKS == 2000
        assert KnowledgeStore().max_chunks == 2000

    def test_append_dict_preserves_shape(self) -> None:
        store = KnowledgeStore(max_chunks=10)
        entry = {"text": "hello", "source": "s", "words": {"hello"}}
        store.append_dict(entry)
        assert len(store) == 1
        # Identity check — the store must not copy on append (legacy
        # code mutates the same dict instance it passed in).
        assert store.chunks[0] is entry

    def test_append_chunk_serializes_via_to_dict(self) -> None:
        store = KnowledgeStore(max_chunks=10)
        store.append_chunk(KnowledgeChunk(text="t", source="s", words={"t"}))
        assert len(store) == 1
        assert store.chunks[0]["text"] == "t"
        assert isinstance(store.chunks[0]["words"], set)

    def test_extend_dicts(self) -> None:
        store = KnowledgeStore(max_chunks=10)
        store.extend_dicts(
            [{"text": f"t{i}", "source": "s", "words": set()} for i in range(3)],
        )
        assert len(store) == 3

    def test_enforce_cap_evicts_oldest(self) -> None:
        store = KnowledgeStore(max_chunks=3)
        for i in range(7):
            store.append_dict(
                {"text": f"t{i}", "source": "s", "words": set()},
            )
        evicted = store.enforce_cap()
        assert evicted == 4
        assert len(store) == 3
        texts = [c["text"] for c in store.chunks]
        assert texts == ["t4", "t5", "t6"]

    def test_enforce_cap_noop_under_cap(self) -> None:
        store = KnowledgeStore(max_chunks=5)
        store.append_dict({"text": "t", "source": "s", "words": set()})
        assert store.enforce_cap() == 0
        assert store.evicted_warned is False

    def test_enforce_cap_logs_once(self, caplog: pytest.LogCaptureFixture) -> None:
        store = KnowledgeStore(max_chunks=2)
        with caplog.at_level(logging.WARNING, logger="carl_studio.knowledge_store"):
            for _ in range(3):
                # Saturate past the cap three times in a row.
                store.chunks.extend(
                    [
                        {"text": f"t{i}", "source": "s", "words": set()}
                        for i in range(5)
                    ],
                )
                store.enforce_cap()
        messages = [
            r.getMessage() for r in caplog.records
            if "knowledge cap reached" in r.getMessage()
        ]
        assert len(messages) == 1
        assert store.evicted_warned is True

    def test_evicted_warned_setter_reset(self) -> None:
        store = KnowledgeStore(max_chunks=1)
        store.append_dict({"text": "a", "source": "s", "words": set()})
        store.append_dict({"text": "b", "source": "s", "words": set()})
        store.enforce_cap()
        assert store.evicted_warned is True
        store.evicted_warned = False
        assert store.evicted_warned is False

    def test_replace_enforces_cap(self) -> None:
        store = KnowledgeStore(max_chunks=2)
        store.replace([
            {"text": "a", "source": "s", "words": set()},
            {"text": "b", "source": "s", "words": set()},
            {"text": "c", "source": "s", "words": set()},
        ])
        texts = [c["text"] for c in store.chunks]
        assert texts == ["b", "c"]

    def test_clear_resets_warning_flag(self) -> None:
        store = KnowledgeStore(max_chunks=1)
        store.chunks.extend([
            {"text": "a", "source": "s", "words": set()},
            {"text": "b", "source": "s", "words": set()},
        ])
        store.enforce_cap()
        assert store.evicted_warned is True
        store.clear()
        assert len(store) == 0
        assert store.evicted_warned is False


class TestKnowledgeStoreRecall:
    def _store_with(self, entries: list[dict[str, Any]]) -> KnowledgeStore:
        s = KnowledgeStore(max_chunks=100)
        for e in entries:
            s.append_dict(e)
        return s

    def test_recall_empty_query(self) -> None:
        store = self._store_with(
            [{"text": "t", "source": "s", "words": {"hello"}}],
        )
        assert store.recall("") == []
        assert store.recall("   ") == []

    def test_recall_scoring_and_order(self) -> None:
        store = self._store_with([
            {"text": "t1", "source": "s", "words": {"apple"}},
            {"text": "t2", "source": "s", "words": {"apple", "pie"}},
            {"text": "t3", "source": "s", "words": {"banana"}},
        ])
        results = store.recall("apple pie", limit=5)
        # Highest overlap first — t2 (2) > t1 (1) > t3 (0 skipped).
        assert len(results) == 2
        assert results[0][0] == 2 and results[0][1]["text"] == "t2"
        assert results[1][0] == 1 and results[1][1]["text"] == "t1"

    def test_recall_limit(self) -> None:
        store = self._store_with([
            {"text": f"t{i}", "source": "s", "words": {"w"}}
            for i in range(10)
        ])
        results = store.recall("w", limit=3)
        assert len(results) == 3

    def test_recall_list_words_also_matched(self) -> None:
        # Legacy persisted entries store words as sorted lists; recall
        # must still match those without blowing up.
        store = self._store_with([
            {"text": "t", "source": "s", "words": ["apple", "pie"]},  # type: ignore[list-item]
        ])
        results = store.recall("apple", limit=5)
        assert len(results) == 1
        assert results[0][0] == 1

    def test_recall_missing_words_returns_no_matches(self) -> None:
        store = self._store_with([
            {"text": "t", "source": "s"},  # no 'words' key at all
        ])
        assert store.recall("anything") == []


class TestKnowledgeStoreRoundtrip:
    def test_to_list_materializes_sorted_word_list(self) -> None:
        store = KnowledgeStore(max_chunks=10)
        store.append_dict(
            {"text": "t", "source": "s", "words": {"banana", "apple"}},
        )
        out = store.to_list()
        assert isinstance(out[0]["words"], list)
        assert out[0]["words"] == ["apple", "banana"]  # sorted

    def test_to_list_is_a_copy(self) -> None:
        store = KnowledgeStore(max_chunks=10)
        store.append_dict({"text": "t", "source": "s", "words": {"w"}})
        out = store.to_list()
        out[0]["text"] = "mutated"
        assert store.chunks[0]["text"] == "t"

    def test_from_list_restores_sets(self) -> None:
        raw = [
            {"text": "t", "source": "s", "words": ["a", "b"]},
        ]
        store = KnowledgeStore.from_list(raw, max_chunks=5)
        assert isinstance(store.chunks[0]["words"], set)
        assert store.chunks[0]["words"] == {"a", "b"}

    def test_from_list_skips_non_dict_entries(self) -> None:
        raw = [
            {"text": "t", "source": "s", "words": ["a"]},
            "not-a-dict",  # type: ignore[list-item]
            None,  # type: ignore[list-item]
        ]
        store = KnowledgeStore.from_list(raw, max_chunks=5)
        assert len(store) == 1

    def test_from_list_enforces_cap(self) -> None:
        raw = [
            {"text": f"t{i}", "source": "s", "words": []}
            for i in range(10)
        ]
        store = KnowledgeStore.from_list(raw, max_chunks=3)
        assert len(store) == 3
        assert [c["text"] for c in store.chunks] == ["t7", "t8", "t9"]

    def test_from_list_none_or_empty_ok(self) -> None:
        assert len(KnowledgeStore.from_list(None, max_chunks=5)) == 0
        assert len(KnowledgeStore.from_list([], max_chunks=5)) == 0

    def test_roundtrip_preserves_content(self) -> None:
        original = KnowledgeStore(max_chunks=5)
        original.append_dict(
            {"text": "hello", "source": "a.txt", "words": {"hello"}},
        )
        original.append_dict(
            {"text": "world", "source": "b.txt", "words": {"world"}},
        )
        restored = KnowledgeStore.from_list(
            original.to_list(), max_chunks=5,
        )
        assert len(restored) == 2
        assert restored.chunks[0]["text"] == "hello"
        assert restored.chunks[0]["words"] == {"hello"}
        assert restored.chunks[1]["text"] == "world"


class TestKnowledgeStoreHelpers:
    def test_is_empty(self) -> None:
        store = KnowledgeStore()
        assert store.is_empty() is True
        store.append_dict({"text": "t", "source": "s", "words": set()})
        assert store.is_empty() is False

    def test_sources_returns_distinct_set(self) -> None:
        store = KnowledgeStore()
        store.append_dict({"text": "t1", "source": "a", "words": set()})
        store.append_dict({"text": "t2", "source": "a", "words": set()})
        store.append_dict({"text": "t3", "source": "b", "words": set()})
        assert store.sources() == {"a", "b"}

    def test_sources_ignores_missing_or_nonstring(self) -> None:
        store = KnowledgeStore()
        store.append_dict({"text": "t1", "source": "a", "words": set()})
        store.append_dict({"text": "t2", "words": set()})  # no source key
        store.append_dict(
            {"text": "t3", "source": 42, "words": set()},  # type: ignore[dict-item]
        )
        assert store.sources() == {"a"}
