"""Tests for carl_core.memory — layered memory with resonance retrieval."""
from __future__ import annotations

import json
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from carl_core.errors import ValidationError
from carl_core.memory import (
    MemoryItem,
    MemoryLayer,
    MemoryStore,
)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(tmp_path / "mem")


# ---------------------------------------------------------------------------
# round-trip
# ---------------------------------------------------------------------------


def test_write_and_list_layer_roundtrip(store: MemoryStore) -> None:
    a = store.write("alpha beta gamma", layer=MemoryLayer.SHORT, tags=["math"])
    b = store.write("delta epsilon", layer=MemoryLayer.SHORT)
    items = store.list_layer(MemoryLayer.SHORT)
    ids = {i.id for i in items}
    assert a.id in ids
    assert b.id in ids
    assert len(items) == 2
    # tags preserved through roundtrip
    recovered = next(i for i in items if i.id == a.id)
    assert recovered.tags == {"math"}
    assert recovered.layer == MemoryLayer.SHORT


def test_memory_item_to_dict_from_dict_roundtrip() -> None:
    now = datetime.now(timezone.utc)
    item = MemoryItem(
        id="abc123",
        content="hello world",
        layer=MemoryLayer.LONG,
        created_at=now,
        tags={"k", "v"},
        access_count=3,
        last_accessed=now,
        metadata={"source": "test"},
    )
    data = item.to_dict()
    rehydrated = MemoryItem.from_dict(data)
    assert rehydrated.id == item.id
    assert rehydrated.content == item.content
    assert rehydrated.layer == item.layer
    assert rehydrated.tags == item.tags
    assert rehydrated.access_count == 3
    assert rehydrated.metadata == {"source": "test"}


# ---------------------------------------------------------------------------
# recall
# ---------------------------------------------------------------------------


def test_recall_returns_items_with_token_overlap(store: MemoryStore) -> None:
    store.write("grpo reward shaping experiment", layer=MemoryLayer.SHORT)
    store.write("something completely unrelated", layer=MemoryLayer.SHORT)
    results = store.recall("grpo reward", top_k=5)
    assert len(results) == 1
    assert "grpo" in results[0].content


def test_recall_filters_by_layer(store: MemoryStore) -> None:
    store.write("grpo training loop", layer=MemoryLayer.SHORT)
    store.write("grpo training loop", layer=MemoryLayer.LONG)
    # only SHORT requested -> only one result
    results = store.recall("grpo training", layers={MemoryLayer.SHORT}, top_k=10)
    assert len(results) == 1
    assert results[0].layer == MemoryLayer.SHORT
    # only LONG requested -> only one result
    results = store.recall("grpo training", layers={MemoryLayer.LONG}, top_k=10)
    assert len(results) == 1
    assert results[0].layer == MemoryLayer.LONG


def test_recall_respects_min_score(store: MemoryStore) -> None:
    store.write("alpha beta gamma", layer=MemoryLayer.SHORT)
    store.write("completely different words", layer=MemoryLayer.SHORT)
    # min_score clamped very high -> nothing passes
    results = store.recall("alpha", top_k=5, min_score=0.99)
    assert results == []
    # with low threshold we should get the alpha-bearing entry
    results = store.recall("alpha", top_k=5, min_score=0.0)
    assert any("alpha" in i.content for i in results)


def test_recall_sorts_by_resonance(store: MemoryStore) -> None:
    # older, less relevant
    old = datetime.now(timezone.utc) - timedelta(hours=2)
    store.write(
        "grpo experiment log",
        layer=MemoryLayer.SHORT,
        created_at=old,
    )
    # newer, more relevant (full query overlap)
    store.write(
        "grpo reward shaping reinforcement learning",
        layer=MemoryLayer.SHORT,
    )
    results = store.recall("grpo reward shaping", top_k=5, min_score=0.0)
    assert len(results) >= 2
    # the newer, more-similar item should come first
    assert "reward" in results[0].content


def test_recall_top_k_bounds(store: MemoryStore) -> None:
    for i in range(10):
        store.write(f"alpha entry {i}", layer=MemoryLayer.SHORT)
    results = store.recall("alpha entry", top_k=3, min_score=0.0)
    assert len(results) == 3


def test_recall_empty_query_returns_nothing(store: MemoryStore) -> None:
    store.write("alpha beta", layer=MemoryLayer.SHORT)
    assert store.recall("", top_k=5) == []


def test_recall_top_k_zero(store: MemoryStore) -> None:
    store.write("alpha", layer=MemoryLayer.SHORT)
    assert store.recall("alpha", top_k=0) == []


def test_recall_negative_top_k_raises(store: MemoryStore) -> None:
    with pytest.raises(ValidationError) as exc:
        store.recall("alpha", top_k=-1)
    assert exc.value.code == "carl.memory.bad_top_k"


def test_recall_updates_access_stats(store: MemoryStore) -> None:
    store.write("grpo reward shaping", layer=MemoryLayer.SHORT)
    results = store.recall("grpo reward", top_k=5, min_score=0.0)
    assert len(results) == 1
    assert results[0].access_count == 1
    assert results[0].last_accessed is not None


# ---------------------------------------------------------------------------
# promote
# ---------------------------------------------------------------------------


def test_promote_to_higher_layer_writes_new_entry(store: MemoryStore) -> None:
    source = store.write("seed belief", layer=MemoryLayer.SHORT, tags=["core"])
    promoted = store.promote(source, MemoryLayer.LONG)
    assert promoted.layer == MemoryLayer.LONG
    assert promoted.content == "seed belief"
    assert promoted.tags == {"core"}
    assert promoted.metadata.get("origin_id") == source.id
    assert promoted.metadata.get("promoted_from") == int(MemoryLayer.SHORT)
    # source is untouched -> still present in SHORT
    short_items = store.list_layer(MemoryLayer.SHORT)
    assert any(i.id == source.id for i in short_items)
    # new entry exists in LONG
    long_items = store.list_layer(MemoryLayer.LONG)
    assert any(i.id == promoted.id for i in long_items)


def test_promote_to_equal_layer_raises(store: MemoryStore) -> None:
    source = store.write("x", layer=MemoryLayer.SHORT)
    with pytest.raises(ValidationError) as exc:
        store.promote(source, MemoryLayer.SHORT)
    assert exc.value.code == "carl.memory.bad_promotion"


def test_promote_to_lower_layer_raises(store: MemoryStore) -> None:
    source = store.write("x", layer=MemoryLayer.LONG)
    with pytest.raises(ValidationError) as exc:
        store.promote(source, MemoryLayer.SHORT)
    assert exc.value.code == "carl.memory.bad_promotion"


# ---------------------------------------------------------------------------
# decay
# ---------------------------------------------------------------------------


def test_decay_pass_removes_stale_echoic_items(store: MemoryStore) -> None:
    # ECHOIC half-life is 60s; write with a far-past timestamp so recency is ~0
    stale_ts = datetime.now(timezone.utc) - timedelta(hours=6)
    fresh_ts = datetime.now(timezone.utc)
    store.write("old echo", layer=MemoryLayer.ECHOIC, created_at=stale_ts)
    store.write("fresh echo", layer=MemoryLayer.ECHOIC, created_at=fresh_ts)
    removed = store.decay_pass()
    assert removed == 1
    survivors = store.list_layer(MemoryLayer.ECHOIC)
    assert len(survivors) == 1
    assert survivors[0].content == "fresh echo"


def test_decay_pass_leaves_long_and_crystal_untouched(store: MemoryStore) -> None:
    old = datetime.now(timezone.utc) - timedelta(days=365 * 5)
    store.write("ancient long", layer=MemoryLayer.LONG, created_at=old)
    store.write("ancient crystal", layer=MemoryLayer.CRYSTAL, created_at=old)
    removed = store.decay_pass()
    assert removed == 0
    assert len(store.list_layer(MemoryLayer.LONG)) == 1
    assert len(store.list_layer(MemoryLayer.CRYSTAL)) == 1


# ---------------------------------------------------------------------------
# resonance scoring edge cases
# ---------------------------------------------------------------------------


def test_resonance_score_zero_for_empty_query(store: MemoryStore) -> None:
    item = store.write("alpha beta", layer=MemoryLayer.SHORT)
    assert store.resonance_score(item, "") == 0.0


def test_resonance_score_zero_for_empty_content(store: MemoryStore) -> None:
    item = store.write("", layer=MemoryLayer.SHORT)
    assert store.resonance_score(item, "hello world") == 0.0


def test_resonance_score_is_nonnegative_and_bounded_ish(store: MemoryStore) -> None:
    item = store.write("grpo reward shaping", layer=MemoryLayer.SHORT, tags=["grpo"])
    s = store.resonance_score(item, "grpo reward shaping")
    assert s > 0.0
    # upper bound: sim <= 1, recency <= 1, (1 + 0.25 * freq) <= 1.25
    assert s <= 1.25 + 1e-9


# ---------------------------------------------------------------------------
# tags
# ---------------------------------------------------------------------------


def test_tag_based_retrieval(store: MemoryStore) -> None:
    # Content has no token overlap with query 'grpo'; tag does.
    store.write("notes about policy gradients", layer=MemoryLayer.LONG, tags=["grpo"])
    store.write("totally unrelated entry", layer=MemoryLayer.LONG)
    results = store.recall("grpo", top_k=5, min_score=0.0)
    assert len(results) >= 1
    assert any("grpo" in item.tags for item in results)


# ---------------------------------------------------------------------------
# resilience
# ---------------------------------------------------------------------------


def test_corrupt_jsonl_line_is_skipped(store: MemoryStore) -> None:
    store.write("valid entry one", layer=MemoryLayer.SHORT)
    # inject corruption
    path = store.root / "short.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.write("NOT JSON\n")
        f.write(json.dumps({"missing": "fields"}) + "\n")
        f.write("\n")
    store.write("valid entry two", layer=MemoryLayer.SHORT)
    items = store.list_layer(MemoryLayer.SHORT)
    contents = {i.content for i in items}
    assert "valid entry one" in contents
    assert "valid entry two" in contents
    # corrupt lines did not crash iteration
    assert len(items) == 2


def test_concurrent_writes_to_same_layer_preserve_all(store: MemoryStore) -> None:
    threads: list[threading.Thread] = []
    errors: list[BaseException] = []

    def writer(idx: int) -> None:
        try:
            for j in range(10):
                store.write(f"thread-{idx} entry-{j}", layer=MemoryLayer.SHORT)
        except BaseException as e:  # noqa: BLE001 - surface to parent
            errors.append(e)

    for i in range(4):
        t = threading.Thread(target=writer, args=(i,))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()
    assert not errors
    items = store.list_layer(MemoryLayer.SHORT)
    assert len(items) == 40
    # every content string is still valid (no partial/truncated lines)
    assert all(item.content.startswith("thread-") for item in items)


def test_store_creates_root_directory(tmp_path: Path) -> None:
    root = tmp_path / "nested" / "deep" / "mem"
    assert not root.exists()
    MemoryStore(root)
    assert root.exists()
    assert root.is_dir()


def test_write_rejects_non_string_content(store: MemoryStore) -> None:
    with pytest.raises(ValidationError) as exc:
        store.write(12345, layer=MemoryLayer.SHORT)  # type: ignore[arg-type]
    assert exc.value.code == "carl.memory.bad_content"
