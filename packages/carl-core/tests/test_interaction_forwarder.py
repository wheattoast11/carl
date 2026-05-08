"""Test the global forwarder seam (v0.10).

Pins down the ``set_global_forwarder`` contract:

* The registered fn is invoked with every ``Step`` appended via
  ``InteractionChain.record(...)``.
* Forwarder-raised exceptions never disrupt the chain.
* ``set_global_forwarder(None)`` clears the registration.
"""
from __future__ import annotations

from carl_core.interaction import (
    ActionType,
    InteractionChain,
    Step,
    set_global_forwarder,
)


def test_set_global_forwarder_invoked_per_step() -> None:
    received: list[Step] = []

    def collect(step: Step) -> None:
        received.append(step)

    set_global_forwarder(collect)
    try:
        chain = InteractionChain(chain_id="test-fwd-1")
        chain.record(
            action=ActionType.LLM_REPLY,
            name="reply",
            input={"q": "ping"},
            output={"a": "pong"},
        )
        chain.record(
            action=ActionType.EXTERNAL,
            name="http.get",
            input={"url": "ex.com"},
            output={"status": 200},
        )
        assert len(received) == 2
        assert received[0].action == ActionType.LLM_REPLY
        assert received[0].name == "reply"
        assert received[1].action == ActionType.EXTERNAL
    finally:
        set_global_forwarder(None)


def test_set_global_forwarder_swallows_errors() -> None:
    """A forwarder that raises must NOT disrupt the chain."""

    def boom(step: Step) -> None:
        raise RuntimeError("forwarder exploded")

    set_global_forwarder(boom)
    try:
        chain = InteractionChain(chain_id="test-fwd-err")
        # This must succeed despite the forwarder raising.
        chain.record(
            action=ActionType.LLM_REPLY,
            name="reply",
            input={"q": "ping"},
            output={"a": "pong"},
        )
        assert len(chain.steps) == 1
        assert chain.steps[0].action == ActionType.LLM_REPLY
    finally:
        set_global_forwarder(None)


def test_set_global_forwarder_none_clears() -> None:
    received: list[Step] = []

    def collect(step: Step) -> None:
        received.append(step)

    set_global_forwarder(collect)
    set_global_forwarder(None)

    chain = InteractionChain(chain_id="test-fwd-clear")
    chain.record(
        action=ActionType.LLM_REPLY,
        name="reply",
        input={},
        output={},
    )
    assert received == []  # forwarder was cleared


def test_set_global_forwarder_does_not_break_legacy_record() -> None:
    """No forwarder registered → record() returns the step normally."""
    set_global_forwarder(None)  # ensure clean slate
    chain = InteractionChain(chain_id="test-no-fwd")
    step = chain.record(
        action=ActionType.TRAINING_STEP,
        name="train.step",
        input={"epoch": 1},
        output={"loss": 0.42},
    )
    assert step.action == ActionType.TRAINING_STEP
    assert step.name == "train.step"
    assert len(chain.steps) == 1
