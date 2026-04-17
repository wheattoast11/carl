"""Shared Hypothesis strategies for property-based tests across the repo.

Importable from both carl-studio's ``tests/`` and carl-core's
``packages/carl-core/tests/``. This module lives at the repo root so the
``pythonpath = ["."]`` entry in ``pyproject.toml [tool.pytest.ini_options]``
makes it resolvable from either test tree without colliding with the two
``tests`` packages.
"""
from __future__ import annotations

from hypothesis import strategies as st

# Leaf JSON primitives. Floats are finite to avoid NaN/inf collisions with
# content_hash (which rejects non-finite floats by design).
st_jsonable_primitive = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(min_value=-(2**53), max_value=2**53),
    st.text(max_size=32),
    st.floats(allow_nan=False, allow_infinity=False, width=32),
)

# Recursive JSON-compatible values: nested lists/dicts of primitives. Bounded
# so examples stay small and the full suite finishes in < 60 s.
st_jsonable = st.recursive(
    st_jsonable_primitive,
    lambda children: st.one_of(
        st.lists(children, max_size=5),
        st.dictionaries(
            st.text(min_size=1, max_size=10),
            children,
            max_size=5,
        ),
    ),
    max_leaves=25,
)

# Relative path segment that is safe inside a sandbox: no separators, no
# traversal, no null bytes, no empty components.
st_safe_name = st.text(
    alphabet=st.characters(
        min_codepoint=0x30,  # '0'
        max_codepoint=0x7A,  # 'z'
        whitelist_categories=("Ll", "Lu", "Nd"),
    ),
    min_size=1,
    max_size=8,
).filter(lambda s: s not in (".", "..") and "\x00" not in s)

# Multi-segment safe relative path. Joined under a sandbox root it stays
# inside the root.
st_safe_relpath = st.lists(st_safe_name, min_size=1, max_size=4).map(
    lambda parts: "/".join(parts)
)


__all__ = [
    "st_jsonable",
    "st_jsonable_primitive",
    "st_safe_name",
    "st_safe_relpath",
]
