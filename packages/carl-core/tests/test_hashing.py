"""Tests for carl_core.hashing — deterministic content hashing."""
from __future__ import annotations

import decimal
import hashlib
from datetime import date, datetime, timezone
from pathlib import Path

import pytest
from pydantic import BaseModel

from carl_core.errors import ValidationError
from carl_core.hashing import canonical_json, content_hash, content_hash_bytes


def test_none_is_deterministic() -> None:
    a = content_hash(None)
    b = content_hash(None)
    assert a == b
    assert len(a) == 64  # sha256 hex


def test_key_order_invariance() -> None:
    assert content_hash({"a": 1, "b": 2}) == content_hash({"b": 2, "a": 1})
    assert content_hash({"x": {"a": 1, "b": 2}}) == content_hash(
        {"x": {"b": 2, "a": 1}}
    )


def test_same_content_same_hash_different_content_different_hash() -> None:
    assert content_hash({"a": 1}) == content_hash({"a": 1})
    assert content_hash({"a": 1}) != content_hash({"a": 2})
    assert content_hash([1, 2, 3]) != content_hash([1, 2, 4])
    assert content_hash("foo") != content_hash("bar")


def test_nested_dict_and_list_stable() -> None:
    data = {
        "outer": {
            "list": [1, {"z": 1, "a": 2}, [3, 4]],
            "nested": {"b": True, "a": None},
        }
    }
    # Reorder keys at every level — hash must be identical.
    reordered = {
        "outer": {
            "nested": {"a": None, "b": True},
            "list": [1, {"a": 2, "z": 1}, [3, 4]],
        }
    }
    assert content_hash(data) == content_hash(reordered)


def test_float_nan_raises_validation_error() -> None:
    with pytest.raises(ValidationError) as excinfo:
        content_hash(float("nan"))
    assert excinfo.value.code == "carl.hash_non_finite"
    assert "nan" in excinfo.value.context["value"].lower()


def test_float_inf_raises_validation_error() -> None:
    with pytest.raises(ValidationError) as excinfo:
        content_hash(float("inf"))
    assert excinfo.value.code == "carl.hash_non_finite"

    with pytest.raises(ValidationError) as excinfo2:
        content_hash(float("-inf"))
    assert excinfo2.value.code == "carl.hash_non_finite"


def test_float_nan_nested_raises() -> None:
    with pytest.raises(ValidationError) as excinfo:
        content_hash({"metric": [1.0, float("nan"), 2.0]})
    assert excinfo.value.code == "carl.hash_non_finite"


def test_bytes_encoded_as_hex() -> None:
    h1 = content_hash(b"\x00\x01\x02")
    h2 = content_hash(b"\x00\x01\x02")
    h3 = content_hash(b"\x00\x01\x03")
    assert h1 == h2
    assert h1 != h3
    # Bytes are encoded via .hex(), so canonical json representation includes
    # the hex string literal for the bytes value.
    assert "000102" in canonical_json({"b": b"\x00\x01\x02"})


def test_datetime_via_isoformat() -> None:
    dt = datetime(2026, 4, 17, 12, 0, 0, tzinfo=timezone.utc)
    canon = canonical_json({"t": dt})
    assert "2026-04-17T12:00:00+00:00" in canon
    assert content_hash({"t": dt}) == content_hash({"t": dt})


def test_date_via_isoformat() -> None:
    d = date(2026, 4, 17)
    canon = canonical_json({"d": d})
    assert "2026-04-17" in canon
    assert content_hash({"d": d}) == content_hash({"d": d})


def test_decimal_is_stringified_preserving_trailing_zero() -> None:
    # Decimal("1.10") must hash as "1.10" not "1.1" — preserving precision.
    assert canonical_json(decimal.Decimal("1.10")) == '"1.10"'
    assert content_hash(decimal.Decimal("1.10")) != content_hash(
        decimal.Decimal("1.1")
    )
    assert content_hash(decimal.Decimal("1.10")) == content_hash(
        decimal.Decimal("1.10")
    )


def test_path_normalized_via_as_posix() -> None:
    p = Path("foo") / "bar" / "baz.txt"
    canon = canonical_json(p)
    assert canon == '"foo/bar/baz.txt"'
    assert content_hash(p) == content_hash(Path("foo/bar/baz.txt"))


def test_pydantic_v2_basemodel_serialized_via_model_dump() -> None:
    class Item(BaseModel):
        name: str
        qty: int
        when: datetime

    item = Item(name="x", qty=3, when=datetime(2026, 4, 17, tzinfo=timezone.utc))
    # model_dump(mode="json") converts datetime → iso string; canonical json
    # then sorts keys.
    canon = canonical_json(item)
    assert '"name":"x"' in canon
    assert '"qty":3' in canon
    assert "2026-04-17" in canon

    # Same semantic content produces same hash.
    item2 = Item(name="x", qty=3, when=datetime(2026, 4, 17, tzinfo=timezone.utc))
    assert content_hash(item) == content_hash(item2)


def test_unsupported_type_raises_validation_error() -> None:
    class Weird:
        pass

    with pytest.raises(ValidationError) as excinfo:
        content_hash(Weird())
    assert excinfo.value.code == "carl.hash_unsupported"
    assert excinfo.value.context["type"] == "Weird"


def test_set_hashed_after_sort() -> None:
    assert content_hash({1, 2, 3}) == content_hash({3, 2, 1})
    assert content_hash({"b", "a", "c"}) == content_hash({"c", "a", "b"})
    assert canonical_json({1, 2, 3}) == "[1,2,3]"


def test_frozenset_hashed_like_set() -> None:
    assert content_hash(frozenset({1, 2, 3})) == content_hash({3, 2, 1})


def test_unorderable_set_raises() -> None:
    with pytest.raises(ValidationError) as excinfo:
        content_hash({1, "a", 2})  # mixed types — not orderable in py3
    assert excinfo.value.code == "carl.hash_unorderable"


def test_canonical_json_stable_across_runs() -> None:
    payload = {
        "b": 2,
        "a": 1,
        "nested": {"z": [1, 2, 3], "y": "val"},
        "list": [{"k2": "v2", "k1": "v1"}],
    }
    results = {canonical_json(payload) for _ in range(10)}
    assert len(results) == 1
    # Keys sorted deterministically.
    single = next(iter(results))
    assert single.index('"a"') < single.index('"b"')


def test_algorithm_sha512_produces_longer_digest() -> None:
    h256 = content_hash({"x": 1}, algorithm="sha256")
    h512 = content_hash({"x": 1}, algorithm="sha512")
    assert len(h256) == 64
    assert len(h512) == 128
    # And matches direct hashlib on the canonical bytes.
    expected = hashlib.sha512(canonical_json({"x": 1}).encode("utf-8")).hexdigest()
    assert h512 == expected


def test_tuple_and_list_are_equivalent() -> None:
    # Tuples canonicalize to JSON arrays — same as lists.
    assert content_hash((1, 2, 3)) == content_hash([1, 2, 3])


def test_bool_not_coerced_to_int() -> None:
    # JSON preserves bool vs int distinction.
    assert canonical_json(True) == "true"
    assert canonical_json(1) == "1"
    assert content_hash(True) != content_hash(1)


def test_content_hash_bytes_uses_raw_bytes() -> None:
    b = b"file contents\nrow 2\n"
    h = content_hash_bytes(b)
    assert h == hashlib.sha256(b).hexdigest()
    # content_hash_bytes must NOT canonicalize — it's raw-bytes mode.
    assert content_hash_bytes(b) != content_hash(b)


def test_content_hash_bytes_algorithm_arg() -> None:
    b = b"abc"
    assert content_hash_bytes(b, algorithm="sha512") == hashlib.sha512(b).hexdigest()


def test_dict_bad_key_type_raises() -> None:
    # A non-scalar dict key (e.g., a tuple) can't serialize canonically.
    with pytest.raises(ValidationError) as excinfo:
        content_hash({("a", "b"): 1})
    assert excinfo.value.code == "carl.hash_bad_key"
    assert excinfo.value.context["key_type"] == "tuple"


def test_empty_containers_are_hashable() -> None:
    assert content_hash({}) == content_hash({})
    assert content_hash([]) == content_hash([])
    assert content_hash(set()) == content_hash(frozenset())
    assert content_hash({}) != content_hash([])
