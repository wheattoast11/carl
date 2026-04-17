"""Property tests for carl_core.hashing.

Invariants:
- content_hash is idempotent (same input -> same digest across calls).
- Distinct primitive values hash to distinct digests (no spurious collision).
- Dict key-order does not matter.
- NaN / +inf / -inf floats are rejected as ValidationError.
"""
from __future__ import annotations

import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from carl_core.errors import ValidationError
from carl_core.hashing import content_hash

from hypothesis_strategies import st_jsonable, st_jsonable_primitive


@given(value=st_jsonable)
@settings(max_examples=100, deadline=None)
def test_content_hash_idempotent(value: object) -> None:
    """content_hash(x) == content_hash(x) for any hashable value."""
    assert content_hash(value) == content_hash(value)


@given(a=st_jsonable_primitive, b=st_jsonable_primitive)
@settings(max_examples=100, deadline=None)
def test_distinct_primitives_distinct_hashes(a: object, b: object) -> None:
    """Two distinct primitives should not collide. Float NaN skipped."""
    # Filter out the special case where both are equal-but-not-==  (NaN/-0.0).
    # We only require: a != b -> hash(a) != hash(b).
    if isinstance(a, float) and math.isnan(a):
        return
    if isinstance(b, float) and math.isnan(b):
        return
    # bool is subclass of int: True == 1 and False == 0 at the Python level,
    # but canonical_json preserves the boolean literal so their hashes differ.
    # That's a correctness property we cover in the unit test already; here we
    # simply skip the Python-equality cases.
    if a == b and type(a) is type(b):
        return
    if isinstance(a, bool) != isinstance(b, bool):
        # different types -> we want different hashes.
        assert content_hash(a) != content_hash(b)
        return
    assert content_hash(a) != content_hash(b)


@given(
    keys=st.lists(
        st.text(min_size=1, max_size=8),
        min_size=1,
        max_size=6,
        unique=True,
    ),
    values=st.lists(
        st.integers(min_value=-1000, max_value=1000),
        min_size=1,
        max_size=6,
    ),
)
@settings(max_examples=100, deadline=None)
def test_key_order_invariance(keys: list[str], values: list[int]) -> None:
    """Dict insertion order must not affect the hash."""
    n = min(len(keys), len(values))
    items = list(zip(keys[:n], values[:n], strict=False))
    forward = dict(items)
    reversed_ = dict(reversed(items))
    assert content_hash(forward) == content_hash(reversed_)


@given(
    nan_like=st.sampled_from([float("nan"), float("inf"), float("-inf")]),
)
@settings(max_examples=25, deadline=None)
def test_nonfinite_float_rejected(nan_like: float) -> None:
    """NaN/inf at top level or nested must raise ValidationError."""
    with pytest.raises(ValidationError):
        content_hash(nan_like)
    with pytest.raises(ValidationError):
        content_hash({"x": nan_like})
    with pytest.raises(ValidationError):
        content_hash([1, nan_like, 2])
