"""Deterministic, canonical content hashing."""
from __future__ import annotations

import decimal
import hashlib
import json
import math
from datetime import date, datetime
from pathlib import Path
from typing import Any

from carl_core.errors import ValidationError


def canonical_json(obj: Any) -> str:
    """Return the canonical JSON representation used for hashing.

    Deterministic for: dict, list, tuple, str, int, bool, None, float, bytes,
    Decimal, datetime, date, Path. NaN/inf floats raise ValidationError.
    """
    return json.dumps(
        _to_canonical(obj),
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    )


def content_hash(data: Any, *, algorithm: str = "sha256") -> str:
    """Hex digest of canonical JSON encoding of data. Stable across runs."""
    h = hashlib.new(algorithm)
    h.update(canonical_json(data).encode("utf-8"))
    return h.hexdigest()


def content_hash_bytes(data: bytes, *, algorithm: str = "sha256") -> str:
    """Hex digest of raw bytes. For file content where canonicalization is wrong."""
    h = hashlib.new(algorithm)
    h.update(data)
    return h.hexdigest()


def _to_canonical(obj: Any) -> Any:
    # Order matters: bool is a subclass of int, so handle bool via the
    # (bool, int, str) branch. None/str/int/bool are JSON-native leaves.
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            raise ValidationError(
                "cannot hash non-finite float",
                code="carl.hash_non_finite",
                context={"value": str(obj)},
            )
        return obj
    if isinstance(obj, (list, tuple)):
        seq: list[Any] = list(obj)  # type: ignore[arg-type]
        return [_to_canonical(x) for x in seq]
    if isinstance(obj, dict):
        # sort keys by str-cast; verify stringable
        items: list[tuple[str, Any]] = []
        mapping: dict[Any, Any] = obj  # type: ignore[assignment]
        for k, v in mapping.items():
            if not isinstance(k, (str, int, float, bool)) and k is not None:
                raise ValidationError(
                    "dict key not JSON-serializable",
                    code="carl.hash_bad_key",
                    context={"key_type": type(k).__name__},
                )
            items.append((str(k), _to_canonical(v)))
        return dict(items)
    if isinstance(obj, bytes):
        return obj.hex()
    if isinstance(obj, decimal.Decimal):
        return str(obj)
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, Path):
        return obj.as_posix()
    if isinstance(obj, (set, frozenset)):
        # sorted list for determinism; requires orderable items
        collection: list[Any] = list(obj)  # type: ignore[arg-type]
        try:
            ordered: list[Any] = sorted(collection)
        except TypeError as exc:
            raise ValidationError(
                "unorderable set cannot be hashed deterministically",
                code="carl.hash_unorderable",
            ) from exc
        return [_to_canonical(x) for x in ordered]
    # Pydantic v2 model
    dump = getattr(obj, "model_dump", None)
    if callable(dump):
        try:
            return _to_canonical(dump(mode="json"))
        except TypeError:
            # Older signatures or non-standard implementations.
            return _to_canonical(dump())
    raise ValidationError(
        f"type {type(obj).__name__} is not hashable",
        code="carl.hash_unsupported",
        context={"type": type(obj).__name__},
    )


__all__ = [
    "canonical_json",
    "content_hash",
    "content_hash_bytes",
]
