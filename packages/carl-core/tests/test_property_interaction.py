"""Property tests for carl_core.interaction.

Invariants:
- Empty chain roundtrips through to_dict/from_dict.
- Arbitrary step sequences preserve order after roundtrip.
- Step input/output JSON-safe data survives the roundtrip.
"""
from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from carl_core.interaction import ActionType, InteractionChain, Step

from hypothesis_strategies import st_jsonable


@given(st.just(None))
@settings(max_examples=5, deadline=None)
def test_empty_chain_roundtrips(_: None) -> None:
    """Empty chain -> to_dict -> from_dict yields equivalent empty chain."""
    chain = InteractionChain()
    restored = InteractionChain.from_dict(chain.to_dict())
    assert restored.chain_id == chain.chain_id
    assert len(restored) == 0
    assert restored.context == chain.context


# A single step draft that will be appended to a chain.
_step_draft = st.fixed_dictionaries(
    {
        "action": st.sampled_from(list(ActionType)),
        "name": st.text(min_size=1, max_size=20),
        "input": st_jsonable,
        "output": st_jsonable,
        "success": st.booleans(),
        "duration_ms": st.one_of(
            st.none(),
            st.floats(
                min_value=0.0, max_value=60_000.0, allow_nan=False,
                allow_infinity=False, width=32,
            ),
        ),
    }
)


@given(drafts=st.lists(_step_draft, min_size=0, max_size=10))
@settings(max_examples=50, deadline=None)
def test_chain_roundtrip_preserves_order(
    drafts: list[dict[str, object]],
) -> None:
    """Record N steps; to_dict -> from_dict yields the same ordered sequence."""
    chain = InteractionChain()
    for d in drafts:
        chain.record(
            action=d["action"],  # type: ignore[arg-type]
            name=d["name"],  # type: ignore[arg-type]
            input=d["input"],
            output=d["output"],
            success=bool(d["success"]),
            duration_ms=d["duration_ms"],  # type: ignore[arg-type]
        )
    restored = InteractionChain.from_dict(chain.to_dict())
    assert len(restored) == len(chain)
    for original, r in zip(chain.steps, restored.steps, strict=True):
        assert r.step_id == original.step_id
        assert r.action == original.action
        assert r.name == original.name
        assert r.success == original.success
        assert r.duration_ms == original.duration_ms


@given(action=st.sampled_from(list(ActionType)), name=st.text(min_size=1, max_size=10))
@settings(max_examples=30, deadline=None)
def test_step_to_dict_is_json_safe(action: ActionType, name: str) -> None:
    """Step.to_dict returns action as the enum string value."""
    s = Step(action=action, name=name)
    d = s.to_dict()
    assert d["action"] == action.value
    assert d["name"] == name
    assert isinstance(d["started_at"], str)
    assert d["success"] is True
