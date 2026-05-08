"""Tests for :mod:`carl_studio.adapters.slime_submit` (Phase F-S3a).

Covers the HF-token-leak guard (defence in depth), the submit/status
HTTP surface, and the bearer-token resolver. Network is mocked via
``respx`` so tests never touch the real carl.camp.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import httpx
import pytest
import respx
from carl_core.errors import (
    SlimeHfTokenLeakError,
    SlimeManagedSubmitFailedError,
    SlimeRunNotFoundError,
)
from carl_studio.adapters.slime_submit import (
    DEFAULT_CARL_CAMP_BASE,
    STATUS_PATH_PREFIX,
    SUBMIT_PATH,
    SlimeRunHandle,
    SlimeRunStatus,
    SlimeSubmitClient,
    assert_no_user_hf_token_leak,
)
from carl_studio.adapters.slime_translator import SlimeArgs


# ---------------------------------------------------------------------------
# Token-leak guard fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def _no_hf_token_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:  # pyright: ignore[reportUnusedFunction]
    """Strip HF_TOKEN from env so guard exact-match path is deterministic.

    Also stubs ``huggingface_hub.get_token`` to return ``None`` so the
    primary resolver in :func:`assert_no_user_hf_token_leak` doesn't
    pull a stray token from the local HF cache.
    """
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_TOKEN", raising=False)
    try:
        import huggingface_hub  # type: ignore[import-not-found]

        monkeypatch.setattr(huggingface_hub, "get_token", lambda: None)
    except ImportError:
        # huggingface_hub absent; the guard's try/except already handles it.
        pass
    yield


# ---------------------------------------------------------------------------
# assert_no_user_hf_token_leak — 4 cases (the contract)
# ---------------------------------------------------------------------------


def test_invariant_rejects_user_hf_token_in_payload(
    _no_hf_token_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exact-match: a string equal to the resolved user HF token raises."""
    # Synthesized at runtime so the source literal doesn't match GitHub's
    # HF-token secret-scanning regex. Still triggers our hf_[A-Za-z0-9_-]{34,}
    # validator AND the exact-env-match path (HF_TOKEN is set below).
    fake_token = "hf_" + "x" * 36
    monkeypatch.setenv("HF_TOKEN", fake_token)

    payload = {
        "megatron": {"seq_length": 2048},
        "sglang": {"tp_size": 1},
        "slime": {"some_value": fake_token},
        "extra": {},
    }
    with pytest.raises(SlimeHfTokenLeakError) as info:
        assert_no_user_hf_token_leak(payload)
    assert info.value.code == "carl.slime.hf_token_leak"
    assert "leaked" in str(info.value).lower()


def test_invariant_rejects_hf_shape_string_in_payload(
    _no_hf_token_env: None,
) -> None:
    """Shape-match: ``hf_<34+ urlsafe chars>`` raises even without env match."""
    # No env token; the regex alone must catch this.
    payload = {
        "megatron": {},
        "sglang": {},
        "slime": {"reward_model": "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"},
        "extra": {},
    }
    with pytest.raises(SlimeHfTokenLeakError) as info:
        assert_no_user_hf_token_leak(payload)
    assert info.value.code == "carl.slime.hf_token_leak"


def test_invariant_rejects_forbidden_key_pattern(_no_hf_token_env: None) -> None:
    """Key-name match: any key containing ``hf_token`` raises."""
    payload = {
        "megatron": {},
        "sglang": {},
        "slime": {},
        "extra": {"hf_token": "<value-irrelevant>"},
    }
    with pytest.raises(SlimeHfTokenLeakError) as info:
        assert_no_user_hf_token_leak(payload)
    assert info.value.code == "carl.slime.hf_token_leak"
    assert "hf_token" in str(info.value)


def test_invariant_passes_clean_payload(_no_hf_token_env: None) -> None:
    """A genuinely clean payload passes silently (no exception)."""
    payload: dict[str, Any] = {
        "megatron": {
            "tensor_model_parallel_size": 1,
            "seq_length": 2048,
            "lr": 1e-6,
        },
        "sglang": {"tp_size": 1, "dtype": "bfloat16"},
        "slime": {
            "model": "Qwen/Qwen2.5-7B",
            "prompt_data": "openai/gsm8k",
            "rollout_temperature": 1.0,
        },
        "extra": {},
    }
    # Must NOT raise.
    assert_no_user_hf_token_leak(payload)


def test_invariant_passes_pydantic_slime_args(_no_hf_token_env: None) -> None:
    """A SlimeArgs Pydantic model is accepted (model_dump round-trip)."""
    args = SlimeArgs(
        megatron={"seq_length": 2048},
        sglang={"tp_size": 1},
        slime={"model": "Qwen/Qwen2.5-7B"},
        extra={},
    )
    assert_no_user_hf_token_leak(args)


def test_invariant_recurses_into_nested_lists(_no_hf_token_env: None) -> None:
    """Token-shaped strings buried in lists still raise."""
    payload = {
        "extra": {
            "args": [
                "--foo",
                "bar",
                "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",  # shape match
            ]
        }
    }
    with pytest.raises(SlimeHfTokenLeakError):
        assert_no_user_hf_token_leak(payload)


def test_invariant_ignores_non_dict_input(_no_hf_token_env: None) -> None:
    """Non-dict / non-Pydantic inputs are no-ops (caller's shape problem)."""
    assert_no_user_hf_token_leak("a string")
    assert_no_user_hf_token_leak(42)
    assert_no_user_hf_token_leak(None)
    assert_no_user_hf_token_leak([1, 2, 3])


# ---------------------------------------------------------------------------
# SlimeSubmitClient — submit happy path + error paths
# ---------------------------------------------------------------------------


def _client(
    *,
    bearer: str | None = "user-bearer-token",
) -> SlimeSubmitClient:
    """Construct a client with a fixed bearer token resolver."""
    return SlimeSubmitClient(
        base_url=DEFAULT_CARL_CAMP_BASE,
        bearer_token_resolver=lambda: bearer,
    )


@respx.mock
def test_submit_happy_path_returns_handle(_no_hf_token_env: None) -> None:
    """A 202 response yields a SlimeRunHandle with run_id / status_url."""
    route = respx.post(f"{DEFAULT_CARL_CAMP_BASE}{SUBMIT_PATH}").mock(
        return_value=httpx.Response(
            202,
            json={
                "run_id": "abc-123",
                "hf_job_id": "hf-job-7",
                "estimated_cost_micros": 12_500_000,
            },
        )
    )

    args = SlimeArgs(
        megatron={"seq_length": 2048},
        sglang={"tp_size": 1},
        slime={"model": "Qwen/Qwen2.5-7B"},
        extra={},
    )
    client = _client()
    handle = client.submit(args, idempotency_key="test-key-1")

    assert isinstance(handle, SlimeRunHandle)
    assert handle.run_id == "abc-123"
    assert handle.hf_job_id == "hf-job-7"
    assert handle.estimated_cost_micros == 12_500_000
    assert handle.status_url == f"{DEFAULT_CARL_CAMP_BASE}{STATUS_PATH_PREFIX}/abc-123"
    assert route.call_count == 1
    # Idempotency-Key header propagated.
    last_request = route.calls.last.request
    assert last_request.headers.get("Idempotency-Key") == "test-key-1"
    assert last_request.headers.get("Authorization") == "Bearer user-bearer-token"


@respx.mock
def test_submit_runs_pre_flight_token_leak_guard(_no_hf_token_env: None) -> None:
    """The guard fires BEFORE the network call (no respx hit)."""
    route = respx.post(f"{DEFAULT_CARL_CAMP_BASE}{SUBMIT_PATH}").mock(
        return_value=httpx.Response(202, json={"run_id": "x"})
    )

    poisoned = {
        "megatron": {},
        "sglang": {},
        "slime": {},
        "extra": {"hf_token": "anything"},
    }
    client = _client()
    with pytest.raises(SlimeHfTokenLeakError):
        client.submit(poisoned)
    assert route.call_count == 0  # short-circuit before wire


@respx.mock
def test_submit_402_raises_managed_submit_failed(_no_hf_token_env: None) -> None:
    """402 from the dispatcher surfaces as SlimeManagedSubmitFailedError."""
    respx.post(f"{DEFAULT_CARL_CAMP_BASE}{SUBMIT_PATH}").mock(
        return_value=httpx.Response(402, json={"error": "tier_required"})
    )

    args = SlimeArgs(
        megatron={"seq_length": 2048},
        sglang={"tp_size": 1},
        slime={"model": "Qwen/Qwen2.5-7B"},
        extra={},
    )
    client = _client()
    with pytest.raises(SlimeManagedSubmitFailedError) as info:
        client.submit(args)
    assert info.value.code == "carl.slime.managed_submit_failed"
    assert "PAID" in str(info.value)


@respx.mock
def test_submit_401_unauthorized(_no_hf_token_env: None) -> None:
    """401 from carl.camp surfaces as SlimeManagedSubmitFailedError + hint."""
    respx.post(f"{DEFAULT_CARL_CAMP_BASE}{SUBMIT_PATH}").mock(
        return_value=httpx.Response(401, json={"error": "unauthorized"})
    )

    args = SlimeArgs(
        megatron={"seq_length": 2048},
        sglang={},
        slime={"model": "Qwen/Qwen2.5-7B"},
        extra={},
    )
    client = _client()
    with pytest.raises(SlimeManagedSubmitFailedError) as info:
        client.submit(args)
    assert "unauthorized" in str(info.value).lower()


@respx.mock
def test_submit_422_validation_error(_no_hf_token_env: None) -> None:
    """422 with an error.details body surfaces the details in the message."""
    respx.post(f"{DEFAULT_CARL_CAMP_BASE}{SUBMIT_PATH}").mock(
        return_value=httpx.Response(
            422,
            json={"error": {"code": "validation_error", "details": "tp_size required"}},
        )
    )

    args = SlimeArgs(
        megatron={"seq_length": 2048},
        sglang={},
        slime={"model": "Qwen/Qwen2.5-7B"},
        extra={},
    )
    client = _client()
    with pytest.raises(SlimeManagedSubmitFailedError) as info:
        client.submit(args)
    assert "validation_error" in str(info.value)
    assert "tp_size required" in str(info.value)


@respx.mock
def test_submit_500_server_error(_no_hf_token_env: None) -> None:
    respx.post(f"{DEFAULT_CARL_CAMP_BASE}{SUBMIT_PATH}").mock(
        return_value=httpx.Response(503, text="upstream temporarily unavailable")
    )

    args = SlimeArgs(
        megatron={"seq_length": 2048},
        sglang={},
        slime={"model": "Qwen/Qwen2.5-7B"},
        extra={},
    )
    client = _client()
    with pytest.raises(SlimeManagedSubmitFailedError) as info:
        client.submit(args)
    assert "503" in str(info.value)


def test_submit_no_bearer_token_raises(_no_hf_token_env: None) -> None:
    """Missing bearer raises before any network attempt."""
    args = SlimeArgs(
        megatron={"seq_length": 2048},
        sglang={},
        slime={"model": "Qwen/Qwen2.5-7B"},
        extra={},
    )
    client = _client(bearer=None)
    with pytest.raises(SlimeManagedSubmitFailedError) as info:
        client.submit(args)
    assert "carl camp login" in str(info.value)


@respx.mock
def test_submit_response_missing_run_id(_no_hf_token_env: None) -> None:
    """Body without run_id is treated as a malformed response."""
    respx.post(f"{DEFAULT_CARL_CAMP_BASE}{SUBMIT_PATH}").mock(
        return_value=httpx.Response(202, json={"foo": "bar"})
    )
    args = SlimeArgs(
        megatron={"seq_length": 2048}, sglang={}, slime={"model": "x"}, extra={}
    )
    client = _client()
    with pytest.raises(SlimeManagedSubmitFailedError) as info:
        client.submit(args)
    assert "run_id" in str(info.value)


# ---------------------------------------------------------------------------
# SlimeSubmitClient — get_status
# ---------------------------------------------------------------------------


@respx.mock
def test_get_status_happy_path(_no_hf_token_env: None) -> None:
    """200 yields a fully-populated SlimeRunStatus."""
    respx.get(f"{DEFAULT_CARL_CAMP_BASE}{STATUS_PATH_PREFIX}/run-1").mock(
        return_value=httpx.Response(
            200,
            json={
                "run_id": "run-1",
                "status": "running",
                "hf_job_id": "hf-job-9",
                "cost_micros": 750_000,
                "gpu_hours": 1.25,
                "started_at": "2026-05-07T10:00:00+00:00",
            },
        )
    )

    client = _client()
    status = client.get_status("run-1")
    assert isinstance(status, SlimeRunStatus)
    assert status.status == "running"
    assert status.hf_job_id == "hf-job-9"
    assert status.cost_micros == 750_000
    assert abs(status.gpu_hours - 1.25) < 1e-9
    assert status.started_at is not None
    assert status.finished_at is None
    assert status.resonant_id is None


@respx.mock
def test_get_status_404_raises_run_not_found(_no_hf_token_env: None) -> None:
    """404 surfaces as SlimeRunNotFoundError with the right code."""
    respx.get(f"{DEFAULT_CARL_CAMP_BASE}{STATUS_PATH_PREFIX}/missing").mock(
        return_value=httpx.Response(404, json={"error": "not found"})
    )

    client = _client()
    with pytest.raises(SlimeRunNotFoundError) as info:
        client.get_status("missing")
    assert info.value.code == "carl.slime.run_not_found"
    assert "missing" in str(info.value)


@respx.mock
def test_get_status_succeeded_with_resonant_id(_no_hf_token_env: None) -> None:
    """A succeeded run with a linked resonant_id reads through cleanly."""
    respx.get(f"{DEFAULT_CARL_CAMP_BASE}{STATUS_PATH_PREFIX}/r2").mock(
        return_value=httpx.Response(
            200,
            json={
                "run_id": "r2",
                "status": "succeeded",
                "cost_micros": 20_000_000,
                "gpu_hours": 4.0,
                "resonant_id": "res-abc",
                "finished_at": "2026-05-07T11:30:00+00:00",
            },
        )
    )
    client = _client()
    status = client.get_status("r2")
    assert status.status == "succeeded"
    assert status.resonant_id == "res-abc"
    assert status.finished_at is not None


@respx.mock
def test_get_status_unrecognized_status_raises(_no_hf_token_env: None) -> None:
    """An unknown status value raises SlimeManagedSubmitFailedError."""
    respx.get(f"{DEFAULT_CARL_CAMP_BASE}{STATUS_PATH_PREFIX}/r3").mock(
        return_value=httpx.Response(200, json={"run_id": "r3", "status": "exploding"})
    )
    client = _client()
    with pytest.raises(SlimeManagedSubmitFailedError) as info:
        client.get_status("r3")
    assert "exploding" in str(info.value)


def test_get_status_empty_run_id_raises(_no_hf_token_env: None) -> None:
    client = _client()
    with pytest.raises(SlimeManagedSubmitFailedError):
        client.get_status("")


# ---------------------------------------------------------------------------
# Network failure paths
# ---------------------------------------------------------------------------


@respx.mock
def test_submit_network_error_wrapped(_no_hf_token_env: None) -> None:
    """httpx transport errors are wrapped in SlimeManagedSubmitFailedError."""
    respx.post(f"{DEFAULT_CARL_CAMP_BASE}{SUBMIT_PATH}").mock(
        side_effect=httpx.ConnectError("boom")
    )

    args = SlimeArgs(
        megatron={"seq_length": 2048}, sglang={}, slime={"model": "x"}, extra={}
    )
    client = _client()
    with pytest.raises(SlimeManagedSubmitFailedError) as info:
        client.submit(args)
    assert "network error" in str(info.value).lower()


# ---------------------------------------------------------------------------
# Error code contract — every new code is importable from carl_core.errors
# ---------------------------------------------------------------------------


def test_error_codes_have_stable_codes() -> None:
    """The 3 new error codes match the spec strings."""
    leak = SlimeHfTokenLeakError("x")
    assert leak.code == "carl.slime.hf_token_leak"

    failed = SlimeManagedSubmitFailedError("x")
    assert failed.code == "carl.slime.managed_submit_failed"

    nf = SlimeRunNotFoundError("x")
    assert nf.code == "carl.slime.run_not_found"
