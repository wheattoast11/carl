"""carl-studio managed slime training client (Phase F-S3a).

Submits a :class:`SlimeArgs` payload to carl.camp's ``POST
/api/train/slime/submit`` (Phase F-B3b) and polls
``GET /api/train/slime/<run_id>`` for status. The pre-flight guard
:func:`assert_no_user_hf_token_leak` refuses to send the request if the
user's HF token is also present in the ``SlimeArgs`` payload (defence
in depth — the carl.camp side also enforces this server-side, but
double-checking here means a misconfigured payload fails fast, before
the network round-trip).

Cross-system invariant (carl.camp agent, 2026-04-21): when the v0.10
managed-slime dispatcher ships on carl.camp, it MUST use carl.camp's
HF token (``CARL_CAMP_HF_TOKEN``) — NEVER the user's encrypted token.
Mixing paths leaks user credentials. carl.camp's existing
``dispatch-hf/[runId]`` route is BYO-only; managed needs a fork at the
top of the dispatcher.

This module is import-cheap (httpx + pydantic + stdlib) so importing
it does not pay the cost of pulling slime, megatron, or sglang.
"""

from __future__ import annotations

import os
import re
import time
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any, Literal, cast

import httpx
from pydantic import BaseModel, ConfigDict

from carl_core.errors import (
    SlimeHfTokenLeakError,
    SlimeManagedSubmitFailedError,
    SlimeRunNotFoundError,
)
from carl_core.interaction import ActionType, InteractionChain

# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

DEFAULT_CARL_CAMP_BASE: str = "https://carl.camp"
SUBMIT_PATH: str = "/api/train/slime/submit"
STATUS_PATH_PREFIX: str = "/api/train/slime"

# Forbidden keys: any dict key whose lowercase form contains any of these
# substrings is treated as carrying an HF-tokenish payload. Conservative
# on purpose — false positives are easy to fix (rename the key); a
# missed leak is not.
_FORBIDDEN_KEY_PATTERNS: tuple[str, ...] = ("hf_token", "huggingface_token")

# HF token shape: ``hf_`` followed by 34+ url-safe characters. Tightened
# in March 2024 — older tokens may be slightly shorter, but the new
# minted ones honor this shape. Conservative on the lower bound (34)
# matches HF's own validator.
_HF_TOKEN_SHAPE: re.Pattern[str] = re.compile(r"^hf_[A-Za-z0-9_\-]{34,}$")


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


SlimeRunState = Literal[
    "pending",
    "queued",
    "running",
    "succeeded",
    "failed",
    "canceled",
]


class SlimeRunHandle(BaseModel):
    """Server-issued handle for a managed slime run.

    Returned from :meth:`SlimeSubmitClient.submit`. Carry this around to
    poll status or look up the run later.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str
    submitted_at: datetime
    status_url: str
    hf_job_id: str | None = None
    estimated_cost_micros: int = 0


class SlimeRunStatus(BaseModel):
    """Snapshot of a managed slime run's current state."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str
    status: SlimeRunState
    hf_job_id: str | None = None
    cost_micros: int = 0
    gpu_hours: float = 0.0
    resonant_id: str | None = None
    error_code: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None


# ---------------------------------------------------------------------------
# HF token-leak guard (defence in depth)
# ---------------------------------------------------------------------------


def assert_no_user_hf_token_leak(slime_args: Any) -> None:
    """Refuse to send a managed-slime payload that carries a user HF token.

    The carl.camp managed dispatcher uses ``CARL_CAMP_HF_TOKEN``
    exclusively. Any payload containing the user's HF token (in a
    field, value, or key name) is rejected here before the network
    round-trip. The carl.camp side enforces the same invariant via
    ``assertNoCredentialMix``; this is the studio half of the
    defence-in-depth pair.

    Recursively scans the payload for:

    1. **Forbidden key names** — keys whose lowercase form contains
       ``hf_token`` or ``huggingface_token``.
    2. **Exact-match HF tokens** — string values equal to the locally
       resolvable HF token (via ``huggingface_hub.get_token()`` or the
       ``HF_TOKEN`` env var).
    3. **HF-token-shaped strings** — values matching ``^hf_[A-Za-z0-9_-]{34,}$``.

    Args:
        slime_args: The payload to scan. Accepts any JSON-able structure
            — a Pydantic ``SlimeArgs`` model, a plain ``dict``, or any
            nested combination of dicts/lists/scalars. Non-dict/non-model
            inputs are a no-op (the caller's problem to validate the
            shape; this function only owns the leak guard).

    Raises:
        SlimeHfTokenLeakError: when any of the above patterns matches.
            Carries a stable ``code = carl.slime.hf_token_leak``.
    """
    # Pydantic v2 → dict; everything else passes through.
    payload: Any
    if hasattr(slime_args, "model_dump"):
        payload = slime_args.model_dump()
    elif isinstance(slime_args, dict):
        payload = cast(dict[str, Any], slime_args)
    else:
        return

    # Resolve the user's HF token, if any. Compare by EXACT match (not
    # prefix) so we don't false-positive on similar-looking strings.
    # huggingface_hub.get_token() already consults HF_TOKEN, the cached
    # ~/.cache/huggingface/token file, and HF_HUB_TOKEN — but its import
    # is heavy enough that we fall through to plain os.environ when
    # the package isn't installed.
    user_token: str | None
    try:
        from huggingface_hub import get_token as _hf_get_token

        user_token = _hf_get_token() or os.environ.get("HF_TOKEN")
    except Exception:  # noqa: BLE001 — defensive: any HF-side failure → env fallback
        user_token = os.environ.get("HF_TOKEN")

    def _scan(obj: Any, path: str) -> None:
        if isinstance(obj, dict):
            obj_dict = cast(dict[Any, Any], obj)
            for raw_key, raw_val in obj_dict.items():
                if isinstance(raw_key, str) and any(
                    p in raw_key.lower() for p in _FORBIDDEN_KEY_PATTERNS
                ):
                    raise SlimeHfTokenLeakError(
                        f"forbidden key {raw_key!r} at {path}.{raw_key} — "
                        "managed slime path uses CARL_CAMP_HF_TOKEN only",
                        context={"path": f"{path}.{raw_key}", "key": raw_key},
                    )
                _scan(raw_val, f"{path}.{raw_key}")
        elif isinstance(obj, (list, tuple)):
            obj_seq = cast("list[Any] | tuple[Any, ...]", obj)
            for i, item in enumerate(obj_seq):
                _scan(item, f"{path}[{i}]")
        elif isinstance(obj, str):
            if user_token and obj == user_token:
                raise SlimeHfTokenLeakError(
                    f"user HF token leaked at {path} — "
                    "managed slime path uses CARL_CAMP_HF_TOKEN only",
                    context={"path": path},
                )
            if _HF_TOKEN_SHAPE.match(obj):
                raise SlimeHfTokenLeakError(
                    f"HF-token-shaped string at {path} — "
                    "managed slime path uses CARL_CAMP_HF_TOKEN only",
                    context={"path": path},
                )
        # Other primitives (int / float / bool / None) — no leak surface.

    _scan(payload, "slime_args")


# ---------------------------------------------------------------------------
# Bearer token resolution
# ---------------------------------------------------------------------------


def _default_bearer_resolver() -> str | None:
    """Public CLAUDE.md-doc'd resolution chain.

    Order: ``CARL_CAMP_TOKEN`` env → ``~/.carl/camp_token`` legacy file
    → ``LocalDB.get_auth("jwt")``. Mirrors the helpers at
    ``carl_studio.tier._resolve_bearer_token_for_verify``,
    ``cli/resonant._resolve_bearer_token``, and
    ``a2a/_cli._resolve_bearer_token``.

    Returns ``None`` when nothing is found; the submit call surfaces
    that as :class:`SlimeManagedSubmitFailedError` with a hint to run
    ``carl camp login``.
    """
    try:
        from carl_studio.tier import (
            _resolve_bearer_token_for_verify,  # pyright: ignore[reportPrivateUsage]
        )

        return _resolve_bearer_token_for_verify()
    except Exception:  # noqa: BLE001 — defensive: any resolver failure → no auth
        return None


# ---------------------------------------------------------------------------
# SlimeSubmitClient
# ---------------------------------------------------------------------------


class SlimeSubmitClient:
    """HTTP client for the managed slime submit + status surface.

    The client is import-cheap (httpx + stdlib only) so wiring it into
    the CLI doesn't pay the slime/megatron/sglang import cost. The
    submit path runs the HF-token-leak guard locally before the wire.

    Args:
        base_url: Override the default ``CARL_CAMP_BASE`` env / fallback
            ``https://carl.camp``.
        bearer_token_resolver: Optional callable returning the user's
            JWT. Defaults to :func:`_default_bearer_resolver` which
            walks the standard CLAUDE.md-doc'd resolution chain.
        http: Optional ``httpx.Client`` injection — useful for tests
            (respx) and for callers that want a shared client/timeouts.
        chain: Optional :class:`InteractionChain` for audit emission.
            Every submit / status / poll call records a step.
    """

    def __init__(
        self,
        *,
        base_url: str | None = None,
        bearer_token_resolver: Callable[[], str | None] | None = None,
        http: httpx.Client | None = None,
        chain: InteractionChain | None = None,
    ) -> None:
        env_base = os.environ.get("CARL_CAMP_BASE")
        self._base_url = (base_url or env_base or DEFAULT_CARL_CAMP_BASE).rstrip("/")
        self._bearer_resolver: Callable[[], str | None] = (
            bearer_token_resolver or _default_bearer_resolver
        )
        self._http: httpx.Client = http or httpx.Client(timeout=httpx.Timeout(30.0))
        self._chain = chain

    # ------------------------------------------------------------------
    # Submit
    # ------------------------------------------------------------------

    def submit(
        self,
        slime_args: Any,
        *,
        idempotency_key: str | None = None,
    ) -> SlimeRunHandle:
        """Submit a managed slime run to carl.camp.

        Pre-flight: :func:`assert_no_user_hf_token_leak`. Raises
        :class:`SlimeHfTokenLeakError` if the payload contains a user HF
        token in any field.

        Args:
            slime_args: A :class:`SlimeArgs` Pydantic model OR any
                ``dict``-shaped payload acceptable to
                ``POST /api/train/slime/submit``.
            idempotency_key: Optional ``Idempotency-Key`` header value
                so retries don't double-submit.

        Returns:
            :class:`SlimeRunHandle` with the server-issued ``run_id``,
            ``status_url`` (a fully-qualified URL the caller can use
            for polling), and optional ``hf_job_id`` /
            ``estimated_cost_micros``.

        Raises:
            SlimeHfTokenLeakError: pre-flight guard tripped.
            SlimeManagedSubmitFailedError: any other submission failure
                — missing bearer, network error, non-2xx response, or
                malformed body.
        """
        assert_no_user_hf_token_leak(slime_args)

        token = self._bearer_resolver()
        if not token:
            raise SlimeManagedSubmitFailedError(
                "no bearer token available — run `carl camp login` first",
                context={"hint": "set CARL_CAMP_TOKEN or run carl camp login"},
            )

        headers: dict[str, str] = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key

        # Convert Pydantic to dict if needed.
        if hasattr(slime_args, "model_dump"):
            body: Any = slime_args.model_dump()
        else:
            body = slime_args

        url = f"{self._base_url}{SUBMIT_PATH}"
        try:
            resp = self._http.post(url, json=body, headers=headers)
        except httpx.HTTPError as exc:
            self._record_step(
                name="slime.managed.submit",
                success=False,
                input_data={"url": url},
                output_data={"error": str(exc), "error_type": type(exc).__name__},
            )
            raise SlimeManagedSubmitFailedError(
                f"network error: {exc}",
                context={"url": url},
                cause=exc,
            ) from exc

        status_code = int(getattr(resp, "status_code", 0) or 0)
        if status_code == 401:
            raise SlimeManagedSubmitFailedError(
                "unauthorized — bearer token invalid or expired",
                context={"status": status_code, "hint": "run: carl camp login"},
            )
        if status_code == 402:
            raise SlimeManagedSubmitFailedError(
                "tier_required — managed slime requires PAID tier",
                context={"status": status_code, "hint": "carl camp upgrade"},
            )
        if status_code == 422:
            details = _safe_json_error_details(resp)
            raise SlimeManagedSubmitFailedError(
                f"validation_error: {details}",
                context={"status": status_code},
            )
        if status_code == 429:
            raise SlimeManagedSubmitFailedError(
                "rate_limited — back off and retry",
                context={"status": status_code},
            )
        if status_code >= 500:
            raise SlimeManagedSubmitFailedError(
                f"server error {status_code}: {_text_snippet(resp)}",
                context={"status": status_code},
            )
        if status_code not in (200, 202):
            raise SlimeManagedSubmitFailedError(
                f"unexpected status {status_code}: {_text_snippet(resp)}",
                context={"status": status_code},
            )

        try:
            payload: Any = resp.json()
        except ValueError as exc:
            # json.JSONDecodeError IS-A ValueError — single catch covers both.
            raise SlimeManagedSubmitFailedError(
                "carl.camp returned a non-JSON submit response",
                context={"status": status_code},
                cause=exc,
            ) from exc

        if not isinstance(payload, dict):
            raise SlimeManagedSubmitFailedError(
                "carl.camp submit response wasn't a JSON object",
                context={"status": status_code},
            )
        body_obj = cast(dict[str, Any], payload)

        run_id = body_obj.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            raise SlimeManagedSubmitFailedError(
                f"submit response missing run_id: {body_obj!r}",
                context={"status": status_code},
            )

        raw_hf_job_id = body_obj.get("hf_job_id")
        hf_job_id: str | None = (
            str(raw_hf_job_id) if isinstance(raw_hf_job_id, str) and raw_hf_job_id else None
        )
        try:
            estimated_cost_micros = int(body_obj.get("estimated_cost_micros", 0) or 0)
        except (TypeError, ValueError):
            estimated_cost_micros = 0

        handle = SlimeRunHandle(
            run_id=run_id,
            submitted_at=datetime.now(tz=timezone.utc),
            status_url=f"{self._base_url}{STATUS_PATH_PREFIX}/{run_id}",
            hf_job_id=hf_job_id,
            estimated_cost_micros=estimated_cost_micros,
        )

        self._record_step(
            name="slime.managed.submit",
            success=True,
            input_data={"url": url, "has_idempotency_key": idempotency_key is not None},
            output_data={
                "run_id": run_id,
                "hf_job_id": hf_job_id,
                "estimated_cost_micros": estimated_cost_micros,
                "http_status": status_code,
            },
        )
        return handle

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self, run_id: str) -> SlimeRunStatus:
        """Fetch the current state of a managed slime run.

        Args:
            run_id: The ``run_id`` from :class:`SlimeRunHandle`.

        Returns:
            :class:`SlimeRunStatus` with the latest state.

        Raises:
            SlimeRunNotFoundError: 404 — id never existed or wrong org.
            SlimeManagedSubmitFailedError: missing bearer, network
                error, non-2xx other than 404, or malformed body.
        """
        if not run_id:
            raise SlimeManagedSubmitFailedError("run_id is required")

        token = self._bearer_resolver()
        if not token:
            raise SlimeManagedSubmitFailedError(
                "no bearer token available",
                context={"hint": "set CARL_CAMP_TOKEN or run carl camp login"},
            )

        url = f"{self._base_url}{STATUS_PATH_PREFIX}/{run_id}"
        headers: dict[str, str] = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
        }
        try:
            resp = self._http.get(url, headers=headers)
        except httpx.HTTPError as exc:
            raise SlimeManagedSubmitFailedError(
                f"network error: {exc}",
                context={"url": url, "run_id": run_id},
                cause=exc,
            ) from exc

        status_code = int(getattr(resp, "status_code", 0) or 0)
        if status_code == 404:
            raise SlimeRunNotFoundError(
                f"slime run {run_id} not found",
                context={"run_id": run_id},
            )
        if status_code == 401:
            raise SlimeManagedSubmitFailedError(
                "unauthorized — bearer token invalid or expired",
                context={"status": status_code, "run_id": run_id},
            )
        if status_code != 200:
            raise SlimeManagedSubmitFailedError(
                f"status fetch failed: HTTP {status_code}: {_text_snippet(resp)}",
                context={"status": status_code, "run_id": run_id},
            )

        try:
            payload: Any = resp.json()
        except ValueError as exc:
            raise SlimeManagedSubmitFailedError(
                "carl.camp returned a non-JSON status response",
                context={"status": status_code, "run_id": run_id},
                cause=exc,
            ) from exc
        if not isinstance(payload, dict):
            raise SlimeManagedSubmitFailedError(
                "carl.camp status response wasn't a JSON object",
                context={"status": status_code, "run_id": run_id},
            )
        body = cast(dict[str, Any], payload)

        return _parse_run_status(body, run_id=run_id)

    # ------------------------------------------------------------------
    # Poll
    # ------------------------------------------------------------------

    def poll_until_done(
        self,
        run_id: str,
        *,
        interval_s: float = 5.0,
        timeout_s: float = 24 * 3600,
    ) -> SlimeRunStatus:
        """Block until the run reaches a terminal state.

        Terminal states: ``succeeded`` / ``failed`` / ``canceled``.

        Args:
            run_id: The ``run_id`` from :class:`SlimeRunHandle`.
            interval_s: Seconds between polls. Default 5s.
            timeout_s: Hard cap on total polling time. Default 24h.

        Returns:
            :class:`SlimeRunStatus` in a terminal state.

        Raises:
            SlimeManagedSubmitFailedError: poll deadline exceeded, or
                propagated from :meth:`get_status`.
            SlimeRunNotFoundError: propagated from :meth:`get_status`.
        """
        if interval_s <= 0:
            raise ValueError("interval_s must be > 0")
        if timeout_s <= 0:
            raise ValueError("timeout_s must be > 0")

        deadline = time.time() + float(timeout_s)
        while time.time() < deadline:
            status = self.get_status(run_id)
            if status.status in ("succeeded", "failed", "canceled"):
                return status
            time.sleep(interval_s)
        raise SlimeManagedSubmitFailedError(
            f"poll timeout exceeded ({timeout_s}s) for run {run_id}",
            context={"run_id": run_id, "timeout_s": timeout_s},
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _record_step(
        self,
        *,
        name: str,
        success: bool,
        input_data: dict[str, Any],
        output_data: dict[str, Any],
    ) -> None:
        if self._chain is None:
            return
        try:
            self._chain.record(
                ActionType.EXTERNAL,
                name=name,
                input=input_data,
                output=output_data,
                success=success,
            )
        except Exception:  # noqa: BLE001 — audit must never break the caller
            pass


# ---------------------------------------------------------------------------
# Module-private helpers (parsing carl.camp responses)
# ---------------------------------------------------------------------------


_VALID_RUN_STATES: frozenset[str] = frozenset({
    "pending",
    "queued",
    "running",
    "succeeded",
    "failed",
    "canceled",
})


def _parse_run_status(body: dict[str, Any], *, run_id: str) -> SlimeRunStatus:
    """Coerce a carl.camp status response into :class:`SlimeRunStatus`."""
    raw_status = body.get("status")
    if not isinstance(raw_status, str) or raw_status not in _VALID_RUN_STATES:
        raise SlimeManagedSubmitFailedError(
            f"unrecognized run status {raw_status!r} for run {run_id}",
            context={"run_id": run_id, "status_value": raw_status},
        )
    status_value = cast(SlimeRunState, raw_status)

    raw_hf_job_id = body.get("hf_job_id")
    hf_job_id: str | None = (
        str(raw_hf_job_id) if isinstance(raw_hf_job_id, str) and raw_hf_job_id else None
    )

    try:
        cost_micros = int(body.get("cost_micros", 0) or 0)
    except (TypeError, ValueError):
        cost_micros = 0

    try:
        gpu_hours = float(body.get("gpu_hours", 0.0) or 0.0)
    except (TypeError, ValueError):
        gpu_hours = 0.0

    raw_resonant_id = body.get("resonant_id")
    resonant_id: str | None = (
        str(raw_resonant_id) if isinstance(raw_resonant_id, str) and raw_resonant_id else None
    )

    raw_error_code = body.get("error_code")
    error_code: str | None = (
        str(raw_error_code) if isinstance(raw_error_code, str) and raw_error_code else None
    )

    started_at = _maybe_iso_to_dt(body.get("started_at"))
    finished_at = _maybe_iso_to_dt(body.get("finished_at"))

    body_run_id = body.get("run_id")
    final_run_id = (
        str(body_run_id) if isinstance(body_run_id, str) and body_run_id else run_id
    )

    return SlimeRunStatus(
        run_id=final_run_id,
        status=status_value,
        hf_job_id=hf_job_id,
        cost_micros=cost_micros,
        gpu_hours=gpu_hours,
        resonant_id=resonant_id,
        error_code=error_code,
        started_at=started_at,
        finished_at=finished_at,
    )


def _maybe_iso_to_dt(raw: Any) -> datetime | None:
    if not isinstance(raw, str) or not raw:
        return None
    try:
        # Accept "...Z" (RFC 3339 UTC marker) — fromisoformat() rejects
        # the literal "Z" suffix on Python < 3.11; we're on >= 3.11 per
        # pyproject so it works, but we normalize defensively for older
        # responses.
        normalized = raw[:-1] + "+00:00" if raw.endswith("Z") else raw
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None


def _safe_json_error_details(resp: httpx.Response) -> Any:
    """Pull the ``error.details`` field from a carl.camp 422 response.

    Falls back to a snippet of the raw body if the response isn't JSON.
    """
    try:
        payload: Any = resp.json()
    except ValueError:
        return _text_snippet(resp)
    if isinstance(payload, dict):
        body = cast(dict[str, Any], payload)
        error = body.get("error")
        if isinstance(error, dict):
            return cast(dict[str, Any], error).get("details", body)
        return body
    return payload


def _text_snippet(resp: httpx.Response, *, limit: int = 200) -> str:
    """Truncated text view of an httpx response body."""
    try:
        text = resp.text
    except Exception:  # noqa: BLE001 — defensive: any decode failure → empty
        return ""
    return text[:limit]


__all__ = [
    "DEFAULT_CARL_CAMP_BASE",
    "STATUS_PATH_PREFIX",
    "SUBMIT_PATH",
    "SlimeRunHandle",
    "SlimeRunState",
    "SlimeRunStatus",
    "SlimeSubmitClient",
    "assert_no_user_hf_token_leak",
]
