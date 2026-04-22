"""Tests for carl_core.audit (@audited decorator + contextvar) and
carl_core.interaction (content-addressed Step serialization)."""

from __future__ import annotations

import time
from typing import Any

import pytest

from carl_core.audit import (
    CURRENT_CHAIN,
    audited,
    chain_context,
    get_current_chain,
    set_current_chain,
)
from carl_core.interaction import (
    ActionType,
    InteractionChain,
    verify_step_content_hash,
)


# ---------------------------------------------------------------------------
# contextvar — bind / unbind / nesting
# ---------------------------------------------------------------------------


def test_current_chain_is_none_by_default() -> None:
    # Every test gets a fresh contextvar via pytest's isolation; explicit
    # reset guards against leakage if a prior test forgot.
    CURRENT_CHAIN.set(None)
    assert get_current_chain() is None


def test_chain_context_binds_and_resets() -> None:
    chain = InteractionChain()
    assert get_current_chain() is None
    with chain_context(chain):
        assert get_current_chain() is chain
    assert get_current_chain() is None


def test_chain_context_nests_correctly() -> None:
    outer = InteractionChain()
    inner = InteractionChain()
    with chain_context(outer):
        assert get_current_chain() is outer
        with chain_context(inner):
            assert get_current_chain() is inner
        assert get_current_chain() is outer
    assert get_current_chain() is None


def test_chain_context_resets_on_exception() -> None:
    chain = InteractionChain()
    with pytest.raises(RuntimeError):
        with chain_context(chain):
            assert get_current_chain() is chain
            raise RuntimeError("boom")
    assert get_current_chain() is None


def test_set_current_chain_returns_reset_token() -> None:
    chain = InteractionChain()
    token = set_current_chain(chain)
    assert get_current_chain() is chain
    CURRENT_CHAIN.reset(token)
    assert get_current_chain() is None


# ---------------------------------------------------------------------------
# @audited — happy path
# ---------------------------------------------------------------------------


def test_audited_emits_step_on_success() -> None:
    chain = InteractionChain()
    calls = {"n": 0}

    @audited(ActionType.TRAINING_STEP, name="pipeline.train_step")
    def train_step(batch_id: int) -> dict[str, int]:
        calls["n"] += 1
        return {"batch_id": batch_id, "loss": 42}

    with chain_context(chain):
        out = train_step(3)

    assert out == {"batch_id": 3, "loss": 42}
    assert calls["n"] == 1
    steps = chain.by_action(ActionType.TRAINING_STEP)
    assert len(steps) == 1
    assert steps[0].name == "pipeline.train_step"
    assert steps[0].success is True
    assert steps[0].duration_ms is not None


def test_audited_uses_function_name_when_no_name_passed() -> None:
    chain = InteractionChain()

    @audited(ActionType.TOOL_CALL)
    def some_helper() -> str:
        return "ok"

    with chain_context(chain):
        some_helper()

    steps = chain.by_action(ActionType.TOOL_CALL)
    assert steps[0].name == "some_helper"


def test_audited_with_name_fn() -> None:
    chain = InteractionChain()

    def _name(phase_id: int, **_: Any) -> str:
        return f"eval.phase.{phase_id}"

    @audited(ActionType.EVAL_PHASE, name_fn=_name)
    def run_phase(phase_id: int, *, dataset: str = "x") -> dict[str, Any]:
        return {"phase": phase_id, "dataset": dataset}

    with chain_context(chain):
        run_phase(7, dataset="ds")

    steps = chain.by_action(ActionType.EVAL_PHASE)
    assert steps[0].name == "eval.phase.7"


def test_audited_records_input_fn_and_output_fn() -> None:
    chain = InteractionChain()

    def _input(trace: list[int], **_: Any) -> dict[str, int]:
        return {"trace_len": len(trace)}

    def _output(result: dict[str, Any]) -> dict[str, float]:
        return {"score": round(float(result["score"]), 3)}

    @audited(ActionType.REWARD, input_fn=_input, output_fn=_output)
    def score(trace: list[int]) -> dict[str, Any]:
        return {"score": sum(trace) / len(trace), "raw": trace}

    with chain_context(chain):
        score([1, 2, 3, 4])

    step = chain.by_action(ActionType.REWARD)[0]
    assert step.input == {"trace_len": 4}
    assert step.output == {"score": 2.5}


# ---------------------------------------------------------------------------
# @audited — no-chain passthrough
# ---------------------------------------------------------------------------


def test_audited_is_noop_when_no_chain_bound() -> None:
    calls = {"n": 0}

    @audited(ActionType.TOOL_CALL)
    def f() -> int:
        calls["n"] += 1
        return 42

    # No chain_context / set_current_chain — should still run the function.
    CURRENT_CHAIN.set(None)
    assert f() == 42
    assert calls["n"] == 1


# ---------------------------------------------------------------------------
# @audited — failure path
# ---------------------------------------------------------------------------


def test_audited_records_failure_and_reraises() -> None:
    chain = InteractionChain()

    @audited(ActionType.TOOL_CALL, name="boom.tool")
    def boom() -> None:
        raise RuntimeError("nope")

    with chain_context(chain):
        with pytest.raises(RuntimeError, match="nope"):
            boom()

    steps = chain.by_action(ActionType.TOOL_CALL)
    assert len(steps) == 1
    assert steps[0].success is False
    assert "RuntimeError" in str(steps[0].output)
    assert "nope" in str(steps[0].output)
    assert steps[0].duration_ms is not None


# ---------------------------------------------------------------------------
# @audited — name/name_fn conflict
# ---------------------------------------------------------------------------


def test_audited_rejects_name_and_name_fn_together() -> None:
    def _n() -> str:
        return "y"

    with pytest.raises(ValueError, match="one of"):

        @audited(ActionType.TOOL_CALL, name="x", name_fn=_n)
        def _f() -> None:  # pyright: ignore[reportUnusedFunction]
            ...


# ---------------------------------------------------------------------------
# Duration is plausible
# ---------------------------------------------------------------------------


def test_audited_duration_ms_reflects_real_time() -> None:
    chain = InteractionChain()

    @audited(ActionType.TOOL_CALL, name="slow")
    def slow() -> None:
        time.sleep(0.02)

    with chain_context(chain):
        slow()

    step = chain.by_action(ActionType.TOOL_CALL)[0]
    assert step.duration_ms is not None
    assert step.duration_ms >= 15.0  # at least ~15ms; sleep was 20ms


# ---------------------------------------------------------------------------
# Content-addressed Step serialization
# ---------------------------------------------------------------------------


def test_step_to_dict_contains_content_hash() -> None:
    chain = InteractionChain()
    chain.record(ActionType.TOOL_CALL, "test", input="x", output="y")
    d = chain.steps[0].to_dict()
    assert "content_hash" in d
    assert isinstance(d["content_hash"], str)
    assert len(d["content_hash"]) == 64
    assert all(c in "0123456789abcdef" for c in d["content_hash"])


def test_verify_step_content_hash_true_for_untouched_step() -> None:
    chain = InteractionChain()
    chain.record(ActionType.LLM_REPLY, "reply", input="p", output="q")
    d = chain.steps[0].to_dict()
    assert verify_step_content_hash(d) is True


def test_verify_step_content_hash_false_on_tamper() -> None:
    chain = InteractionChain()
    chain.record(ActionType.LLM_REPLY, "reply", input="p", output="q")
    d = chain.steps[0].to_dict()
    # Tamper with the body
    d["output"] = "not-q"
    assert verify_step_content_hash(d) is False


def test_verify_step_content_hash_false_when_hash_missing() -> None:
    assert verify_step_content_hash({"step_id": "x"}) is False


def test_content_hash_stable_across_reserializations() -> None:
    chain = InteractionChain()
    chain.record(ActionType.TOOL_CALL, "x", input={"a": 1}, output={"b": 2})
    d1 = chain.steps[0].to_dict()
    d2 = chain.steps[0].to_dict()
    assert d1["content_hash"] == d2["content_hash"]


def test_content_hash_differs_for_distinct_content() -> None:
    chain = InteractionChain()
    chain.record(ActionType.TOOL_CALL, "a", input=1, output=1)
    chain.record(ActionType.TOOL_CALL, "b", input=2, output=2)
    h1 = chain.steps[0].to_dict()["content_hash"]
    h2 = chain.steps[1].to_dict()["content_hash"]
    assert h1 != h2


def test_content_hash_survives_jsonl_roundtrip() -> None:
    import json

    chain = InteractionChain()
    chain.record(ActionType.EVAL_PHASE, "phase1", input={"ds": "x"}, output={"acc": 0.9})
    jsonl = chain.to_jsonl()
    # Parse each non-header line as JSON; verify each step's hash
    for line in jsonl.split("\n")[1:]:
        if not line.strip():
            continue
        step_dict = json.loads(line)
        assert verify_step_content_hash(step_dict)


# ---------------------------------------------------------------------------
# Content hash + secret scrubbing interact correctly
# ---------------------------------------------------------------------------


def test_content_hash_computed_over_scrubbed_form() -> None:
    """The hash is computed over what's actually serialized — i.e. after
    the secret-scrubbing pass. This is the correct semantics (audit matches
    what's on disk) AND it means the hash never embeds un-scrubbed values.
    """
    chain = InteractionChain()
    # A secret-looking key triggers scrubbing
    chain.record(
        ActionType.EXTERNAL,
        "http.post",
        input={"url": "https://api.example.com", "token": "sk-ant-SECRET12345"},
        output={"status": 200},
    )
    d = chain.steps[0].to_dict()
    # Scrubbing redacted the token
    assert d["input"]["token"] == "<redacted>"
    # Hash verifies (scrubbed form is what's hashed)
    assert verify_step_content_hash(d) is True
    # Re-hash manually from body to be sure
    import json
    import hashlib

    body = {k: v for k, v in d.items() if k != "content_hash"}
    recomputed = hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()
    assert recomputed == d["content_hash"]


# ---------------------------------------------------------------------------
# Integration: @audited + content_hash produce a verifiable chain
# ---------------------------------------------------------------------------


def test_audited_chain_is_content_verifiable() -> None:
    chain = InteractionChain()

    @audited(ActionType.TRAINING_STEP, name="train.step")
    def tstep(i: int) -> int:
        return i * 2

    with chain_context(chain):
        for i in range(5):
            tstep(i)

    for raw in [s.to_dict() for s in chain.steps]:
        assert verify_step_content_hash(raw) is True
