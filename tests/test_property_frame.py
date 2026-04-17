"""Property tests for carl_studio.frame.WorkFrame.

Invariants:
- model_dump -> model_validate roundtrip equals the original.
- attention_query / decompose are stable functions of inputs.
"""
from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from carl_studio.frame import WorkFrame


_st_name = st.text(min_size=0, max_size=24)
_st_list = st.lists(st.text(min_size=1, max_size=16), min_size=0, max_size=5)


@given(
    domain=_st_name,
    function=_st_name,
    role=_st_name,
    objectives=_st_list,
    constraints=_st_list,
    entities=_st_list,
    metrics=_st_list,
    context=_st_name,
)
@settings(max_examples=100, deadline=None)
def test_workframe_roundtrip_equal(
    domain: str,
    function: str,
    role: str,
    objectives: list[str],
    constraints: list[str],
    entities: list[str],
    metrics: list[str],
    context: str,
) -> None:
    """model_dump -> model_validate reproduces an equivalent WorkFrame."""
    frame = WorkFrame(
        domain=domain,
        function=function,
        role=role,
        objectives=objectives,
        constraints=constraints,
        entities=entities,
        metrics=metrics,
        context=context,
    )
    restored = WorkFrame.model_validate(frame.model_dump())
    assert restored == frame
    assert restored.active == frame.active
    assert restored.attention_query() == frame.attention_query()
    assert restored.decompose() == frame.decompose()


@given(
    domain=_st_name,
    function=_st_name,
    role=_st_name,
    objectives=_st_list,
)
@settings(max_examples=50, deadline=None)
def test_attention_query_deterministic(
    domain: str, function: str, role: str, objectives: list[str]
) -> None:
    """Same frame configuration yields identical attention queries."""
    args = {
        "domain": domain,
        "function": function,
        "role": role,
        "objectives": objectives,
    }
    assert WorkFrame(**args).attention_query() == WorkFrame(**args).attention_query()


@given(entities=_st_list)
@settings(max_examples=50, deadline=None)
def test_entity_patterns_lowercased(entities: list[str]) -> None:
    """Every entity pattern is lowercase and mirrors a non-empty source entity."""
    frame = WorkFrame(entities=entities)
    patterns = frame.entity_patterns()
    # Only non-empty, stripped entities make it into patterns.
    expected = [e.lower() for e in entities if e.strip()]
    assert patterns == expected
    for p in patterns:
        assert p == p.lower()
