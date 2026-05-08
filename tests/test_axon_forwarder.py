"""tests/test_axon_forwarder.py — AXON forwarder unit tests.

Covers:

* threshold-batched flush (8 sub-tests)
* env opt-out via ``AXON_FORWARD_DISABLED=1``
* consent-off short-circuit
* secret redaction in payload
* full Step → signal mapping table
* idempotency-key stability across replays
* secondary ``coherence.update`` emission
* no-bearer silent drop
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest

from carl_core.interaction import ActionType, InteractionChain, Step
from carl_studio.telemetry.axon import (
    AxonEvent,
    AxonForwarder,
    reset_default_forwarder,
)
from carl_studio.telemetry.axon_signals import (
    ACTION_DISPATCHED,
    COHERENCE_UPDATE,
    INTERACTION_CREATED,
    SKILL_CRYSTALLIZED,
    SKILL_TRAINING_STARTED,
)


@pytest.fixture(autouse=True)
def _reset_forwarder():  # pyright: ignore[reportUnusedFunction]
    yield
    reset_default_forwarder()


def _make_fwd(**overrides: Any) -> AxonForwarder:
    """Test-friendly forwarder: telemetry-on, fake bearer, mocked HTTP."""
    defaults: dict[str, Any] = {
        "consent_check": lambda: True,
        "bearer_token_resolver": lambda: "test-jwt",
        "http": MagicMock(spec=httpx.Client),
    }
    return AxonForwarder(**{**defaults, **overrides})


# ---------------------------------------------------------------------------
# 1. Threshold-batched flush
# ---------------------------------------------------------------------------


def test_buffer_flushes_at_threshold_100() -> None:
    fwd = _make_fwd(batch_size=100)
    for i in range(100):
        fwd._queue.put(
            AxonEvent(
                signal_type=SKILL_TRAINING_STARTED,
                payload={"i": i},
                occurred_at=datetime.now(tz=timezone.utc),
                idempotency_key=f"k-{i}",
            )
        )
    posted = fwd.flush()
    assert posted == 100
    fwd._http.post.assert_called_once()  # type: ignore[attr-defined]
    # Confirm the body has 100 events.
    call_kwargs = fwd._http.post.call_args.kwargs  # type: ignore[attr-defined]
    body = call_kwargs["json"]
    assert len(body["events"]) == 100


# ---------------------------------------------------------------------------
# 2. AXON_FORWARD_DISABLED env opt-out
# ---------------------------------------------------------------------------


def test_disabled_when_axon_forward_disabled_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AXON_FORWARD_DISABLED", "1")
    fwd = _make_fwd()  # consent ON, but env kill-switch wins
    assert fwd.is_enabled() is False

    # on_step should be a no-op even with consent True
    chain = InteractionChain(chain_id="test-disabled")
    chain.record(action=ActionType.LLM_REPLY, name="reply", input={}, output={})
    fwd.on_step(chain.steps[-1])
    assert fwd._queue.qsize() == 0


# ---------------------------------------------------------------------------
# 3. Consent-off short-circuit
# ---------------------------------------------------------------------------


def test_disabled_when_consent_telemetry_off() -> None:
    fwd = _make_fwd(consent_check=lambda: False)
    assert fwd.is_enabled() is False
    chain = InteractionChain(chain_id="test-no-consent")
    chain.record(action=ActionType.TRAINING_STEP, name="train", input={}, output={})
    fwd.on_step(chain.steps[-1])
    assert fwd._queue.qsize() == 0


# ---------------------------------------------------------------------------
# 4. Secret redaction
# ---------------------------------------------------------------------------


def test_secret_keys_redacted_before_forward() -> None:
    """A secret-shaped key in step.input must NOT appear in the forwarded payload."""
    fwd = _make_fwd()
    chain = InteractionChain(chain_id="test-redact")
    # `api_key` is sensitive per carl_core.errors._is_sensitive
    chain.record(
        action=ActionType.LLM_REPLY,
        name="reply",
        input={"api_key": "sk-leaked-secret-value", "prompt": "hello"},
        output={"a": "ok"},
    )
    step = chain.steps[-1]
    events = fwd.map_step_to_events(step)
    assert events, "expected at least one mapped event"

    # The forwarded payload must not contain the literal secret value.
    for ev in events:
        body = ev.model_dump_json()
        assert "sk-leaked-secret-value" not in body, (
            f"secret value leaked into AxonEvent payload: {body}"
        )


# ---------------------------------------------------------------------------
# 5. Step → signal mapping table
# ---------------------------------------------------------------------------


def test_step_to_signal_mapping_table() -> None:
    fwd = _make_fwd()
    cases: list[tuple[ActionType, str]] = [
        (ActionType.TRAINING_STEP, SKILL_TRAINING_STARTED),
        (ActionType.CHECKPOINT, SKILL_CRYSTALLIZED),
        (ActionType.LLM_REPLY, INTERACTION_CREATED),
        (ActionType.REWARD, INTERACTION_CREATED),
        (ActionType.EXTERNAL, ACTION_DISPATCHED),
    ]
    for action, expected_signal in cases:
        chain = InteractionChain(chain_id=f"test-map-{action.name}")
        chain.record(action=action, name=f"name-{action.name}", input={}, output={})
        events = fwd.map_step_to_events(chain.steps[-1])
        signals = {ev.signal_type for ev in events}
        assert expected_signal in signals, (
            f"{action} expected to emit {expected_signal}; got {signals}"
        )


def test_unmapped_action_emits_no_events() -> None:
    """Action types outside the mapping table produce no AxonEvents
    (unless they happen to carry coherence)."""
    fwd = _make_fwd()
    chain = InteractionChain(chain_id="test-unmapped")
    chain.record(action=ActionType.USER_INPUT, name="user", input={}, output={})
    events = fwd.map_step_to_events(chain.steps[-1])
    assert events == []


# ---------------------------------------------------------------------------
# 6. Idempotency-key stability
# ---------------------------------------------------------------------------


def test_idempotency_key_stable_across_replays() -> None:
    expected = hashlib.sha256(
        f"step-abc:{SKILL_TRAINING_STARTED}".encode()
    ).hexdigest()
    actual = AxonForwarder._idempotency_key("step-abc", SKILL_TRAINING_STARTED)
    assert actual == expected


def test_idempotency_key_differs_per_signal() -> None:
    """Two events from the same Step (primary + coherence) must
    produce DIFFERENT idempotency keys."""
    primary = AxonForwarder._idempotency_key("step-x", SKILL_TRAINING_STARTED)
    coherence = AxonForwarder._idempotency_key("step-x", COHERENCE_UPDATE)
    assert primary != coherence


# ---------------------------------------------------------------------------
# 7. Secondary coherence event on phi/kuramoto_r
# ---------------------------------------------------------------------------


def test_kuramoto_r_emits_secondary_coherence_event() -> None:
    fwd = _make_fwd()
    chain = InteractionChain(chain_id="test-coherence")
    chain.record(
        action=ActionType.TRAINING_STEP,
        name="train",
        input={},
        output={},
        kuramoto_r=0.95,
    )
    step = chain.steps[-1]
    events = fwd.map_step_to_events(step)
    signals = [ev.signal_type for ev in events]
    assert SKILL_TRAINING_STARTED in signals
    assert COHERENCE_UPDATE in signals
    # Verify the coherence event carries the kuramoto_r value
    coherence_ev = next(ev for ev in events if ev.signal_type == COHERENCE_UPDATE)
    assert coherence_ev.payload["kuramoto_r"] == 0.95


def test_phi_only_emits_coherence_for_unmapped_action() -> None:
    """A step on an UN-mapped action that carries phi STILL produces a
    coherence-update event (the coherence channel is decoupled from
    primary-signal mapping)."""
    fwd = _make_fwd()
    step = Step(action=ActionType.TOOL_CALL, name="tool.x", phi=0.42)
    events = fwd.map_step_to_events(step)
    signals = [ev.signal_type for ev in events]
    assert signals == [COHERENCE_UPDATE]


# ---------------------------------------------------------------------------
# 8. No-bearer silent drop
# ---------------------------------------------------------------------------


def test_no_bearer_silent_drop() -> None:
    fwd = _make_fwd(bearer_token_resolver=lambda: None)
    fwd._queue.put(
        AxonEvent(
            signal_type=SKILL_TRAINING_STARTED,
            payload={},
            occurred_at=datetime.now(tz=timezone.utc),
            idempotency_key="k-1",
        )
    )
    posted = fwd.flush()
    assert posted == 0
    fwd._http.post.assert_not_called()  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# 9. on_step end-to-end (queue + signal mapping + redaction)
# ---------------------------------------------------------------------------


def test_on_step_enqueues_mapped_events() -> None:
    fwd = _make_fwd()
    chain = InteractionChain(chain_id="test-on-step")
    chain.record(
        action=ActionType.TRAINING_STEP,
        name="train",
        input={"epoch": 1},
        output={"loss": 0.42},
        kuramoto_r=0.7,
    )
    fwd.on_step(chain.steps[-1])
    # 1 primary (skill.training_started) + 1 coherence (coherence.update) = 2
    assert fwd._queue.qsize() == 2


def test_flush_posts_to_correct_url_with_bearer() -> None:
    fwd = _make_fwd(base_url="https://carl.test")
    chain = InteractionChain(chain_id="test-url")
    chain.record(action=ActionType.LLM_REPLY, name="r", input={}, output={})
    fwd.on_step(chain.steps[-1])
    posted = fwd.flush()
    assert posted == 1
    call = fwd._http.post.call_args  # type: ignore[attr-defined]
    assert call.args[0] == "https://carl.test/api/axon/ingest"
    headers = call.kwargs["headers"]
    assert headers["Authorization"] == "Bearer test-jwt"
    assert headers["Content-Type"] == "application/json"


def test_http_failure_swallowed_returns_zero() -> None:
    """If httpx.post raises, flush returns 0 instead of bubbling."""
    fwd = _make_fwd()
    fwd._http.post.side_effect = httpx.ConnectError("boom")  # type: ignore[attr-defined]
    fwd._queue.put(
        AxonEvent(
            signal_type=SKILL_TRAINING_STARTED,
            payload={},
            occurred_at=datetime.now(tz=timezone.utc),
            idempotency_key="k-fail",
        )
    )
    posted = fwd.flush()
    assert posted == 0  # error swallowed; batch dropped


def test_install_default_forwarder_idempotent() -> None:
    """install_default_forwarder is idempotent — multiple calls return
    the same singleton."""
    from carl_studio.telemetry.axon import install_default_forwarder

    first = install_default_forwarder(base_url="https://carl.test")
    second = install_default_forwarder()
    assert first is second
