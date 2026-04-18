"""Layered memory with decay-aware resonance retrieval.

The memory primitive is the substrate of CARL's fractal intelligence loop:
hypothesize -> eval -> infer -> commit -> recall.

Items live in layers ordered by decay speed. ICONIC is replaced every turn;
CRYSTAL is immutable within a deploy. Retrieval is resonance-driven:
similarity * recency_decay * (1 + frequency_bonus).

Persistence is JSONL-per-layer under a root directory. The store is
dependency-light (stdlib + carl_core only) and uses overlap-based similarity
instead of embeddings so it can run in any environment.
"""
from __future__ import annotations

import json
import math
import re
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import IntEnum
from pathlib import Path
from typing import Any, cast

from carl_core.errors import ValidationError
from carl_core.hashing import content_hash


class MemoryLayer(IntEnum):
    """Decay-ordered layers. Lower = more ephemeral.

    ICONIC:  one exchange (~immediate, replaced every turn)
    ECHOIC:  recent turn window (~seconds-to-minutes, last ~5-10 turns)
    SHORT:   current session (~this conversation, InteractionChain-backed)
    WORKING: recent sessions (~last 7d, retrieved on trigger)
    LONG:    committed learnings (~persistent, compiled into prompt)
    CRYSTAL: constitution + code truths (~immutable within a deploy)
    """

    ICONIC = 0
    ECHOIC = 1
    SHORT = 2
    WORKING = 3
    LONG = 4
    CRYSTAL = 5


# Per-layer decay constant (half-life in seconds). Crystal has infinite half-life.
_LAYER_HALFLIFE_S: dict[MemoryLayer, float] = {
    MemoryLayer.ICONIC: 5.0,
    MemoryLayer.ECHOIC: 60.0,
    MemoryLayer.SHORT: 3600.0,
    MemoryLayer.WORKING: 7 * 24 * 3600.0,
    MemoryLayer.LONG: 365 * 24 * 3600.0,
    MemoryLayer.CRYSTAL: math.inf,
}

# Layers that are pruned by decay_pass(). LONG and CRYSTAL are persistent.
_DECAY_PRUNED_LAYERS: tuple[MemoryLayer, ...] = (
    MemoryLayer.ICONIC,
    MemoryLayer.ECHOIC,
    MemoryLayer.SHORT,
    MemoryLayer.WORKING,
)


@dataclass
class MemoryItem:
    """A single memory entry scoped to one layer."""

    id: str
    content: str
    layer: MemoryLayer
    created_at: datetime
    tags: set[str] = field(default_factory=lambda: set[str]())
    access_count: int = 0
    last_accessed: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=lambda: dict[str, Any]())

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict with deterministic tag ordering."""
        return {
            "id": self.id,
            "content": self.content,
            "layer": int(self.layer),
            "created_at": self.created_at.isoformat(),
            "tags": sorted(self.tags),
            "access_count": self.access_count,
            "last_accessed": (
                self.last_accessed.isoformat() if self.last_accessed is not None else None
            ),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MemoryItem:
        """Rehydrate from a dict produced by ``to_dict``.

        Missing optional fields default gracefully; malformed required fields
        raise ``KeyError``/``ValueError`` which callers can catch to skip lines.
        """
        last_raw = data.get("last_accessed")
        last_accessed: datetime | None
        if last_raw is None or last_raw == "":
            last_accessed = None
        else:
            last_accessed = datetime.fromisoformat(str(last_raw))

        tags_raw: Any = data.get("tags", [])
        tags: set[str] = set()
        if isinstance(tags_raw, (list, tuple, set, frozenset)):
            iterable = cast("Iterable[Any]", tags_raw)
            for t in iterable:
                tags.add(str(t))

        metadata_raw: Any = data.get("metadata") or {}
        metadata: dict[str, Any] = {}
        if isinstance(metadata_raw, dict):
            meta_map = cast("dict[Any, Any]", metadata_raw)
            for k, v in meta_map.items():
                metadata[str(k)] = v

        return cls(
            id=str(data["id"]),
            content=str(data["content"]),
            layer=MemoryLayer(int(data["layer"])),
            created_at=datetime.fromisoformat(str(data["created_at"])),
            tags=tags,
            access_count=int(data.get("access_count", 0)),
            last_accessed=last_accessed,
            metadata=metadata,
        )


# ------------------------- similarity primitives ----------------------------

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> set[str]:
    """Rough token set for overlap-based similarity.

    Lowercases, keeps alphanumerics, drops tokens shorter than 2 chars.
    """
    return {t for t in _TOKEN_RE.findall(text.lower()) if len(t) >= 2}


def _jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard index. Zero for either empty set."""
    if not a or not b:
        return 0.0
    intersection = len(a & b)
    if intersection == 0:
        return 0.0
    return intersection / len(a | b)


def _recency_weight(item: MemoryItem, *, now: datetime | None = None) -> float:
    """Exponential-decay weight in [0, 1] based on the item's layer half-life.

    ``dt == 0`` yields 1.0; ``dt == half_life`` yields 0.5. Future timestamps
    are clamped to 0 seconds to avoid weights > 1.
    """
    ref = now if now is not None else datetime.now(timezone.utc)
    created = item.created_at
    # If either datetime is naive, treat both as UTC for a safe subtraction.
    if ref.tzinfo is None:
        ref = ref.replace(tzinfo=timezone.utc)
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    dt = (ref - created).total_seconds()
    half_life = _LAYER_HALFLIFE_S[item.layer]
    if math.isinf(half_life):
        return 1.0
    if dt <= 0.0:
        return 1.0
    return math.exp(-math.log(2.0) * dt / half_life)


def _frequency_weight(item: MemoryItem) -> float:
    """Log-compressed frequency bonus in ~[0, 1].

    ``log1p(access_count) / log1p(100)`` so frequent items aren't swamped
    by outliers and a fresh item isn't penalized for zero accesses.
    """
    return math.log1p(max(item.access_count, 0)) / math.log1p(100.0)


# ------------------------------- store --------------------------------------


class MemoryStore:
    """JSONL-per-layer memory store with resonance-based recall.

    Layout::

        <root>/iconic.jsonl
        <root>/echoic.jsonl
        <root>/short.jsonl
        <root>/working.jsonl
        <root>/long.jsonl
        <root>/crystal.jsonl

    Writes are atomic appends; ``decay_pass`` rewrites survivors to a
    ``.tmp`` sibling and uses ``Path.replace`` for a crash-safe swap.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    # ---- paths / ids -------------------------------------------------------

    def _path(self, layer: MemoryLayer) -> Path:
        return self.root / f"{layer.name.lower()}.jsonl"

    def _next_id(self, content: str) -> str:
        """Deterministic-ish id: first 16 hex chars of a content+time hash."""
        return content_hash({"c": content, "t": time.time_ns()})[:16]

    # ---- write paths -------------------------------------------------------

    def write(
        self,
        content: str,
        *,
        layer: MemoryLayer = MemoryLayer.SHORT,
        tags: Iterable[str] | None = None,
        metadata: dict[str, Any] | None = None,
        created_at: datetime | None = None,
    ) -> MemoryItem:
        """Append a new item to ``layer``.

        ``created_at`` is accepted for tests and replay; defaults to now(UTC).
        Append is a single ``write()`` of a JSON line + ``\\n`` which POSIX
        guarantees to be atomic up to pipe-buffer size for our payloads.
        """
        # Runtime guards handle callers that bypass the type checker.
        if not isinstance(layer, MemoryLayer):  # type: ignore[unreachable]
            raise ValidationError(
                "layer must be a MemoryLayer",
                code="carl.memory.bad_layer",
                context={"layer": repr(layer)},
            )
        if not isinstance(content, str):  # type: ignore[unreachable]
            raise ValidationError(
                "content must be a string",
                code="carl.memory.bad_content",
                context={"type": type(content).__name__},
            )

        created = created_at if created_at is not None else datetime.now(timezone.utc)
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)

        item = MemoryItem(
            id=self._next_id(content),
            content=content,
            layer=layer,
            created_at=created,
            tags=set(tags or ()),
            metadata=dict(metadata or {}),
        )
        path = self._path(layer)
        line = json.dumps(item.to_dict(), sort_keys=True, ensure_ascii=False) + "\n"
        # Append-only is safe under concurrent single-line writes on POSIX
        # when each call opens/closes its own file handle.
        with path.open("a", encoding="utf-8") as f:
            f.write(line)
        return item

    # ---- iteration / scoring ----------------------------------------------

    def _iter_layer(self, layer: MemoryLayer) -> Iterator[MemoryItem]:
        path = self._path(layer)
        if not path.exists():
            return
        with path.open("r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(data, dict):
                    continue
                raw_map = cast("dict[Any, Any]", data)
                typed: dict[str, Any] = {}
                for k, v in raw_map.items():
                    typed[str(k)] = v
                try:
                    yield MemoryItem.from_dict(typed)
                except (KeyError, ValueError, TypeError):
                    # Skip corrupt/legacy lines without tearing down the stream.
                    continue

    def resonance_score(
        self,
        item: MemoryItem,
        query: str,
        *,
        now: datetime | None = None,
    ) -> float:
        """Score the resonance between ``item`` and ``query``.

        Returns 0.0 when either side has no usable tokens. Otherwise::

            similarity * recency * (1 + 0.25 * frequency)

        where ``similarity`` blends token Jaccard with a tag-overlap boost.
        """
        q_tokens = _tokenize(query)
        c_tokens = _tokenize(item.content)
        if not q_tokens or not c_tokens:
            # Still allow pure-tag hits even when content has no tokens — but
            # only if the query itself has tokens to match against tags.
            if not q_tokens:
                return 0.0

        sim = _jaccard(c_tokens, q_tokens)
        if item.tags:
            tag_overlap = len(q_tokens & item.tags) / max(len(q_tokens), 1)
            # Tag boost contributes at half weight so it can't crowd out
            # a strong content match but still surfaces tag-only hits.
            sim = max(sim, tag_overlap * 0.5)
        if sim <= 0.0:
            return 0.0
        rec = _recency_weight(item, now=now)
        freq = _frequency_weight(item)
        return sim * rec * (1.0 + 0.25 * freq)

    def recall(
        self,
        query: str,
        *,
        layers: set[MemoryLayer] | None = None,
        top_k: int = 5,
        min_score: float = 0.05,
    ) -> list[MemoryItem]:
        """Retrieve top-k items across requested layers, sorted by resonance.

        Increments ``access_count`` and updates ``last_accessed`` on the
        returned objects in-memory. Per-read persistence is intentionally
        skipped — callers who want to commit frequency stats should pass the
        item through ``write`` in a higher layer via ``promote``.
        """
        if top_k < 0:
            raise ValidationError(
                "top_k must be non-negative",
                code="carl.memory.bad_top_k",
                context={"top_k": top_k},
            )
        if top_k == 0:
            return []

        requested: set[MemoryLayer] = layers if layers is not None else set(MemoryLayer)
        now = datetime.now(timezone.utc)
        scored: list[tuple[float, int, MemoryItem]] = []
        tie_breaker = 0
        for layer in requested:
            for item in self._iter_layer(layer):
                s = self.resonance_score(item, query, now=now)
                if s >= min_score:
                    # tie_breaker keeps sort stable without relying on
                    # MemoryItem being orderable.
                    scored.append((s, tie_breaker, item))
                    tie_breaker += 1
        scored.sort(key=lambda t: (t[0], -t[1]), reverse=True)
        results = [item for _, _, item in scored[:top_k]]
        for item in results:
            item.access_count += 1
            item.last_accessed = now
        return results

    # ---- layer transitions -------------------------------------------------

    def promote(self, item: MemoryItem, to: MemoryLayer) -> MemoryItem:
        """Copy ``item`` into a higher layer. Source is untouched.

        Raises ``ValidationError`` (``carl.memory.bad_promotion``) if ``to``
        is not strictly higher than the source's layer.
        """
        if not isinstance(to, MemoryLayer):  # type: ignore[unreachable]
            raise ValidationError(
                "promote target must be a MemoryLayer",
                code="carl.memory.bad_layer",
                context={"to": repr(to)},
            )
        if int(to) <= int(item.layer):
            raise ValidationError(
                f"promote target {to.name} must be higher than current {item.layer.name}",
                code="carl.memory.bad_promotion",
                context={"from": item.layer.name, "to": to.name},
            )
        merged_meta: dict[str, Any] = {
            **item.metadata,
            "promoted_from": int(item.layer),
            "origin_id": item.id,
        }
        return self.write(
            item.content,
            layer=to,
            tags=item.tags,
            metadata=merged_meta,
        )

    def decay_pass(self, *, min_score: float = 0.01) -> int:
        """Prune items from ephemeral layers whose recency weight has decayed.

        Only ICONIC, ECHOIC, SHORT, and WORKING are scanned. LONG and CRYSTAL
        are persistent by design — use ``list_layer`` + manual rewrite if you
        need to curate them.

        Returns the number of items removed across all pruned layers.
        """
        removed = 0
        now = datetime.now(timezone.utc)
        for layer in _DECAY_PRUNED_LAYERS:
            path = self._path(layer)
            if not path.exists():
                continue
            survivors: list[dict[str, Any]] = []
            layer_removed = 0
            for item in self._iter_layer(layer):
                if _recency_weight(item, now=now) >= min_score:
                    survivors.append(item.to_dict())
                else:
                    layer_removed += 1
            if layer_removed == 0:
                # Nothing to do; skip the rewrite so we don't churn mtime.
                continue
            tmp = path.with_suffix(".tmp")
            with tmp.open("w", encoding="utf-8") as f:
                for rec in survivors:
                    f.write(json.dumps(rec, sort_keys=True, ensure_ascii=False) + "\n")
            tmp.replace(path)
            removed += layer_removed
        return removed

    # ---- admin / export ----------------------------------------------------

    def list_layer(self, layer: MemoryLayer) -> list[MemoryItem]:
        """Return every item currently persisted in ``layer``."""
        return list(self._iter_layer(layer))


__all__ = [
    "MemoryItem",
    "MemoryLayer",
    "MemoryStore",
]
