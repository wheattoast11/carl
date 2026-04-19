"""OpenReward managed-reward client.

OpenReward hosts a reward-model API. We talk to it as a :class:`AsyncBaseConnection`
with :class:`ConnectionKind.MODEL` because, from the connection registry's
perspective, it is a 3P model endpoint. The managed-hosting surface is
gated behind the PAID tier per ``connection.openreward`` in
:data:`carl_core.tier.FEATURE_TIERS`.

Authentication: API key via ``api_key`` constructor arg or the
``OPENREWARD_API_KEY`` environment variable. No key => we fail open at
construction but refuse to :meth:`open` (the tier check still runs first).

Lazy imports: ``httpx`` is imported inside :meth:`_connect` so importing
this module never pulls a live HTTP client into processes that don't need
one.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, cast

from carl_core.connection import (
    AsyncBaseConnection,
    ConnectionDirection,
    ConnectionKind,
    ConnectionPolicyError,
    ConnectionScope,
    ConnectionSpec,
    ConnectionTransport,
    ConnectionTrust,
    ConnectionUnavailableError,
)
from carl_core.interaction import InteractionChain
from carl_core.tier import Tier, tier_allows

if TYPE_CHECKING:
    pass


OPENREWARD_CONNECTION_SPEC: ConnectionSpec = ConnectionSpec(
    name="carl.reward.openreward",
    scope=ConnectionScope.THREE_P,
    kind=ConnectionKind.MODEL,
    direction=ConnectionDirection.EGRESS,
    transport=ConnectionTransport.HTTP,
    trust=ConnectionTrust.METERED,
    metadata={"provider": "openreward"},
)


class OpenRewardClient(AsyncBaseConnection):
    """Thin async client for OpenReward's ``/score`` endpoint.

    Tier gate: :meth:`open` refuses unless the resolved tier permits
    ``connection.openreward`` (PAID). The check happens before any
    network activity, so FREE-tier callers never even establish a session.
    """

    spec = OPENREWARD_CONNECTION_SPEC

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://api.openreward.ai",
        *,
        chain: InteractionChain | None = None,
        timeout: float = 30.0,
    ) -> None:
        super().__init__(chain=chain)
        self._api_key = api_key or os.environ.get("OPENREWARD_API_KEY")
        self._base_url = base_url.rstrip("/") if base_url else ""
        self._timeout = float(timeout)
        self._client: Any = None

    # -- lifecycle -------------------------------------------------------

    async def _connect(self) -> None:
        # 1. Tier gate — do this first so FREE callers never hit the wire.
        current = _resolve_tier()
        if not tier_allows(current, "connection.openreward"):
            raise ConnectionPolicyError(
                "OpenReward managed hosting requires CARL Paid. "
                "Upgrade at https://carl.camp/pricing",
                code="carl.connection.policy.tier",
                context={
                    "feature": "connection.openreward",
                    "tier_required": Tier.PAID.value,
                    "tier_current": current.value,
                    "upgrade_url": "https://carl.camp/pricing",
                },
            )

        # 2. Transport presence.
        try:
            import httpx
        except ImportError as exc:
            raise ConnectionUnavailableError(
                "httpx is required for OpenRewardClient. Install with: pip install httpx",
                context={"package": "httpx"},
                cause=exc,
            ) from exc

        if not self._base_url:
            raise ConnectionUnavailableError(
                "OpenRewardClient.base_url is empty",
                context={"base_url": self._base_url},
            )

        # 3. Open the session.
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=self._timeout,
            headers=self._auth_headers(),
        )

    async def _authenticate(self) -> None:
        # Trust level is METERED; by contract carl-core will call
        # _authenticate during open(). We satisfy the contract by
        # refusing to proceed without an API key.
        if not self._api_key:
            raise ConnectionUnavailableError(
                "OpenRewardClient requires an API key. Pass api_key= or "
                "set OPENREWARD_API_KEY",
                context={"env_var": "OPENREWARD_API_KEY"},
            )

    async def _close(self) -> None:
        client = self._client
        self._client = None
        if client is not None:
            try:
                await client.aclose()
            except Exception:  # noqa: BLE001
                pass

    # -- public API -----------------------------------------------------

    async def score(
        self,
        response: str,
        reference: str | None = None,
        rubric: str | None = None,
    ) -> float:
        """Return a scalar reward for ``response``.

        Parameters
        ----------
        response
            The agent's textual output.
        reference
            Optional reference / gold answer.
        rubric
            Optional rubric identifier or textual rubric description.

        Raises
        ------
        ConnectionPolicyError
            If the caller is on a tier below PAID.
        ConnectionUnavailableError
            If the HTTP client is not ready or httpx is missing.
        """
        if not isinstance(response, str):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise TypeError(f"response must be str, got {type(response).__name__}")
        if self._client is None:
            raise ConnectionUnavailableError(
                "OpenRewardClient.score called before open(); use `async with client:` "
                "or call `await client.open()` first",
                context={"connection_id": self.connection_id},
            )

        payload: dict[str, Any] = {"response": response}
        if reference is not None:
            payload["reference"] = reference
        if rubric is not None:
            payload["rubric"] = rubric

        async with self.transact("score"):
            result = await self._post_json("/score", payload)

        raw: Any = result.get("score")
        try:
            value = float(raw) if raw is not None else 0.0
        except (TypeError, ValueError):
            value = 0.0
        if value < 0.0:
            return 0.0
        if value > 1.0:
            return 1.0
        return value

    # -- HTTP plumbing --------------------------------------------------

    async def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        if self._client is None:
            raise ConnectionUnavailableError(
                "OpenRewardClient HTTP session not open",
                context={"path": path},
            )
        response = await self._client.post(path, json=payload)
        if hasattr(response, "raise_for_status"):
            response.raise_for_status()
        data_raw: Any = response.json()
        if not isinstance(data_raw, dict):
            raise ValueError(
                f"OpenReward {path} returned non-dict: {type(data_raw).__name__}",
            )
        data_dict = cast(dict[Any, Any], data_raw)
        return {str(k): v for k, v in data_dict.items()}

    # -- helpers --------------------------------------------------------

    def _auth_headers(self) -> dict[str, str]:
        headers = {"content-type": "application/json"}
        if self._api_key:
            headers["authorization"] = f"Bearer {self._api_key}"
        return headers


def _resolve_tier() -> Tier:
    """Resolve the caller's effective tier without hard-depending on carl_studio.

    carl_studio.tier.detect_effective_tier pulls from settings + local DB;
    we use it when available and fall back to FREE when not (e.g. a
    minimal carl-core-only process). Any failure path defaults to FREE so
    we fail closed.
    """
    try:
        from carl_studio.tier import detect_effective_tier
        from carl_studio.settings import CARLSettings

        settings = CARLSettings.load()
        return detect_effective_tier(settings.tier)
    except Exception:  # noqa: BLE001
        return Tier.FREE


__all__ = [
    "OpenRewardClient",
    "OPENREWARD_CONNECTION_SPEC",
]
