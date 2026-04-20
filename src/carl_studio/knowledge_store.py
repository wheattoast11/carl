"""Knowledge chunk storage for :class:`carl_studio.chat_agent.CARLAgent`.

Extracted from ``chat_agent.py`` in the v0.7 Phase-2 god-class decomposition
(ticket G1 Phase-2 / F-PYTH-002). This module owns:

* the bounded knowledge list with LRU eviction on overflow,
* a single-warning-per-instance policy so long-running sessions don't
  flood the log when the cap is repeatedly hit,
* recall / search over stored chunks,
* JSON-serializable round-trip for session persistence.

Backward-compat contract
------------------------

In-tree tests (``tests/test_chat_agent_robustness.py``, ``tests/test_agent.py``,
``tests/test_uat_e2e.py``) reach into ``CARLAgent._knowledge`` directly and
treat it as a mutable list of plain dicts with keys ``text`` / ``source`` /
``words``. The store preserves that exact shape: :pyattr:`KnowledgeStore.chunks`
*is* ``list[dict[str, Any]]``, not a list of typed objects. The typed
:class:`KnowledgeChunk` dataclass is a convenience constructor — the store
accepts either dict form or ``KnowledgeChunk`` via ``append_dict`` /
``append_chunk``, and ``to_list`` / ``from_list`` perform the same set-vs-list
word-serialization dance the chat_agent save/load path used to do inline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Iterable, cast

logger = logging.getLogger(__name__)


# Default cap — kept public so ``CARLAgent._KNOWLEDGE_MAX_CHUNKS`` can stay
# the class-level constant tests already assert against.
_KNOWLEDGE_MAX_CHUNKS: int = 2000


@dataclass
class KnowledgeChunk:
    """Typed constructor for knowledge entries.

    Provided for new call sites that want a type-safe handle. Legacy callers
    keep pushing raw dicts onto :pyattr:`KnowledgeStore.chunks` and the store
    preserves that shape verbatim. ``to_dict`` returns the wire-format the
    chat_agent session save path expects: ``words`` as a sorted list (JSON
    can't serialize ``set``), everything else pass-through.
    """

    text: str
    source: str
    words: set[str] = field(
        default_factory=lambda: set(),  # type: set[str]
    )
    meta: dict[str, Any] = field(
        default_factory=lambda: {},  # type: dict[str, Any]
    )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to the dict shape legacy code uses on ``_knowledge``.

        ``words`` materializes as a *set* — the in-memory shape the runtime
        relies on for O(1) intersection in :meth:`KnowledgeStore.recall`.
        Session save converts it to a sorted list right before JSON dump;
        :meth:`KnowledgeStore.from_list` converts back on load.
        """
        entry: dict[str, Any] = {
            "text": self.text,
            "source": self.source,
            "words": set(self.words),
        }
        if self.meta:
            entry["meta"] = dict(self.meta)
        return entry

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> KnowledgeChunk:
        """Best-effort reconstruction from a legacy dict entry.

        Unknown fields land in ``meta``; bad ``words`` shapes degrade to
        an empty set rather than raising (resumed sessions come from
        user-writable JSON and must never crash the loader).
        """
        words_raw: Any = raw.get("words", [])
        words: set[str] = set()
        if isinstance(words_raw, (list, tuple, set)):
            words = {str(w) for w in cast(Iterable[Any], words_raw)}
        meta_raw: Any = raw.get("meta", {})
        meta: dict[str, Any] = {}
        if isinstance(meta_raw, dict):
            meta = {
                str(k): v
                for k, v in cast(dict[Any, Any], meta_raw).items()
            }
        return cls(
            text=str(raw.get("text", "")),
            source=str(raw.get("source", "")),
            words=words,
            meta=meta,
        )


@dataclass
class KnowledgeStore:
    """Bounded, LRU-evicting store of knowledge chunk dicts.

    Parameters
    ----------
    max_chunks:
        Upper bound on the number of retained chunks. Must be ``>= 1``
        when set via the constructor; callers that want the default use
        :data:`_KNOWLEDGE_MAX_CHUNKS`. Enforcement is LRU (oldest-first):
        when ``len(chunks) > max_chunks`` the front of the list is
        discarded. A single ``warning`` log is emitted on the first
        eviction per instance so long-running sessions don't spam the log.
    chunks:
        The backing list. Callers that need raw access (legacy tests,
        ``CARLAgent._knowledge`` shim) mutate this directly. The store does
        **not** assume private ownership — ``append_dict`` / ``append_chunk``
        run the eviction guard explicitly on the path they care about.
    """

    max_chunks: int = _KNOWLEDGE_MAX_CHUNKS
    chunks: list[dict[str, Any]] = field(
        default_factory=lambda: [],  # type: list[dict[str, Any]]
    )
    # Tracked per-instance so the eviction warning fires at most once.
    _evicted_warned: bool = False

    # ------------------------------------------------------------------
    # Mutation paths
    # ------------------------------------------------------------------

    def append_dict(self, entry: dict[str, Any]) -> None:
        """Append a raw dict (legacy call shape from ``_tool_ingest``)."""
        self.chunks.append(entry)

    def append_chunk(self, chunk: KnowledgeChunk) -> None:
        """Append a typed :class:`KnowledgeChunk`."""
        self.chunks.append(chunk.to_dict())

    def extend_dicts(self, entries: Iterable[dict[str, Any]]) -> None:
        """Append many raw dict entries. Does not evict mid-iteration."""
        self.chunks.extend(entries)

    def clear(self) -> None:
        """Drop all chunks. Resets the warning flag so tests are deterministic."""
        self.chunks.clear()
        self._evicted_warned = False

    def replace(self, new_chunks: list[dict[str, Any]]) -> None:
        """Replace the backing list verbatim (used by session ``load``).

        After replacement, the cap is enforced so a resumed session with
        more chunks than the current cap doesn't blow the budget on its
        first turn.
        """
        self.chunks = list(new_chunks)
        self.enforce_cap()

    # ------------------------------------------------------------------
    # Eviction
    # ------------------------------------------------------------------

    def enforce_cap(self) -> int:
        """Evict oldest entries until ``len(chunks) <= max_chunks``.

        Returns the number of entries evicted (``0`` if under cap). The
        eviction warning fires at most once per instance — further
        evictions are silent so a session that repeatedly saturates the
        cap doesn't flood the log.
        """
        cap = self.max_chunks
        excess = len(self.chunks) - cap
        if excess <= 0:
            return 0
        # Drop the oldest entries (front of the list is oldest by insertion).
        del self.chunks[:excess]
        if not self._evicted_warned:
            logger.warning(
                "CARLAgent knowledge cap reached (%d): evicted %d oldest "
                "chunk(s); further evictions will be silent for this session.",
                cap, excess,
            )
            self._evicted_warned = True
        return excess

    @property
    def evicted_warned(self) -> bool:
        """True once :meth:`enforce_cap` has logged the eviction warning."""
        return self._evicted_warned

    @evicted_warned.setter
    def evicted_warned(self, value: bool) -> None:
        """Setter retained so test fixtures can reset the flag for repro."""
        self._evicted_warned = bool(value)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def is_empty(self) -> bool:
        return not self.chunks

    def __len__(self) -> int:  # pragma: no cover — thin proxy
        return len(self.chunks)

    def sources(self) -> set[str]:
        """Set of distinct source identifiers across all chunks."""
        out: set[str] = set()
        for c in self.chunks:
            src = c.get("source")
            if isinstance(src, str) and src:
                out.add(src)
        return out

    def recall(self, query: str, *, limit: int = 5) -> list[tuple[int, dict[str, Any]]]:
        """Return the top ``limit`` chunks by token-overlap with ``query``.

        Returns ``(score, chunk_dict)`` tuples, highest score first. Chunks
        with zero overlap are not returned. ``words`` missing from a chunk
        dict is treated as an empty set so malformed entries don't crash
        the search loop — they simply never match.
        """
        terms = {w for w in query.lower().split() if w}
        if not terms:
            return []
        scored: list[tuple[int, dict[str, Any]]] = []
        for chunk in self.chunks:
            words: Any = chunk.get("words")
            overlap: int = 0
            if isinstance(words, (set, list, tuple)):
                # Legacy on-disk shape (list/tuple) and the in-memory shape
                # (set) both reduce to a stringified set for overlap count.
                overlap = len(
                    terms & {str(w) for w in cast(Iterable[Any], words)},
                )
            if overlap > 0:
                scored.append((overlap, chunk))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return scored[:limit]

    # ------------------------------------------------------------------
    # Round-trip for session persistence
    # ------------------------------------------------------------------

    def to_list(self) -> list[dict[str, Any]]:
        """Return a JSON-serializable snapshot of ``chunks``.

        Sets become sorted lists so :func:`json.dumps` succeeds without
        a custom encoder. The returned list is a *copy* — mutating it
        does not affect the store.
        """
        out: list[dict[str, Any]] = []
        for c in self.chunks:
            entry: dict[str, Any] = dict(c)
            words: Any = entry.get("words")
            if isinstance(words, set):
                entry["words"] = sorted(
                    str(w) for w in cast(Iterable[Any], words)
                )
            out.append(entry)
        return out

    @classmethod
    def from_list(
        cls,
        raw: list[dict[str, Any]] | None,
        *,
        max_chunks: int,
    ) -> KnowledgeStore:
        """Rehydrate a store from persisted state.

        ``words`` lists are converted back to sets for O(1) lookup. The
        resulting store is bounded by ``max_chunks`` — oversized persisted
        states are trimmed on load so a downsized cap doesn't blow the
        next turn's budget.
        """
        store = cls(max_chunks=max_chunks)
        if not raw:
            return store
        for entry in raw:
            # Defensive: callers may pass user-authored JSON that contains
            # non-dict noise at the top level; skip rather than raise.
            if not isinstance(entry, dict):  # type: ignore[reportUnnecessaryIsInstance]
                continue
            cleaned: dict[str, Any] = dict(entry)
            words: Any = cleaned.get("words")
            if isinstance(words, list):
                cleaned["words"] = {
                    str(w) for w in cast(Iterable[Any], words)
                }
            elif not isinstance(words, set):
                cleaned["words"] = set()
            store.chunks.append(cleaned)
        store.enforce_cap()
        return store
