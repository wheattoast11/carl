"""OpenEnv runtime bridge.

Bidirectional adapter between carl-studio's :class:`EnvironmentConnection`
and the OpenEnv ``step/reset/state`` runtime contract.

Two concrete surfaces:

* :class:`OpenEnvServer` — wraps a local :class:`EnvironmentConnection` as
  an OpenEnv server. ``reset`` / ``step`` / ``state`` are the only three
  entrypoints the runtime requires; the server dispatches ``step``'s
  ``tool`` field to the matching method on the wrapped env.
* :class:`OpenEnvClient` — consumes a remote OpenEnv server image as a
  :class:`EnvironmentConnection` subclass over HTTP JSON-RPC. Tools are
  dispatched dynamically; ``reset`` / ``step`` round-trip through the
  standard OpenEnv endpoints. ``httpx`` is a lazy import so the runtime
  stays optional.

The ``EnvironmentSpec.lane`` surface is preserved — OpenEnv is the
runtime contract, lanes remain orthogonal metadata for routing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, cast

from carl_core.connection import (
    ConnectionDirection,
    ConnectionKind,
    ConnectionScope,
    ConnectionSpec,
    ConnectionState,
    ConnectionTransport,
    ConnectionTrust,
    ConnectionUnavailableError,
)
from carl_core.interaction import ActionType, InteractionChain

from carl_studio.environments.connection import EnvironmentConnection
from carl_studio.environments.protocol import EnvironmentLane, EnvironmentSpec


def _empty_args() -> dict[str, Any]:
    return {}


def _empty_meta() -> dict[str, Any]:
    return {}


# ---------------------------------------------------------------------------
# Wire types
# ---------------------------------------------------------------------------


@dataclass
class OpenEnvAction:
    """Agent-side action in the OpenEnv ``step`` contract.

    Attributes
    ----------
    tool
        The name of a tool method on the wrapped environment (e.g.
        ``"execute_code"``). Must match a method in ``spec.tools``.
    args
        Keyword arguments for the tool method. Must be JSON-serializable.
    reasoning
        Optional free-form rationale surfaced to telemetry.
    """

    tool: str
    args: dict[str, Any] = field(default_factory=_empty_args)
    reasoning: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "args": dict(self.args),
            "reasoning": self.reasoning,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> OpenEnvAction:
        tool_raw = payload.get("tool")
        if not isinstance(tool_raw, str) or not tool_raw:
            raise ValueError("OpenEnvAction.tool must be a non-empty string")
        tool: str = tool_raw
        args_raw: Any = payload.get("args", {})
        if not isinstance(args_raw, dict):
            raise ValueError("OpenEnvAction.args must be a dict")
        args_dict = cast(dict[Any, Any], args_raw)
        args_typed: dict[str, Any] = {str(k): v for k, v in args_dict.items()}
        reasoning_raw = payload.get("reasoning")
        if reasoning_raw is not None and not isinstance(reasoning_raw, str):
            raise ValueError("OpenEnvAction.reasoning must be a string if present")
        reasoning: str | None = reasoning_raw
        return cls(tool=tool, args=args_typed, reasoning=reasoning)


@dataclass
class OpenEnvObservation:
    """Environment-side observation in the OpenEnv ``step``/``reset`` contract.

    Attributes
    ----------
    text
        Free-form textual observation (task description on reset, tool
        result summary on step).
    tool_result
        Raw return value from the tool method (None for reset).
    reward
        Cumulative reward at this turn.
    done
        True if the episode terminated.
    turn
        Current turn counter (0 after reset, +1 after each step).
    meta
        Arbitrary extra fields — tool name, args digest, etc.
    """

    text: str | None = None
    tool_result: Any = None
    reward: float = 0.0
    done: bool = False
    turn: int = 0
    meta: dict[str, Any] = field(default_factory=_empty_meta)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "tool_result": self.tool_result,
            "reward": float(self.reward),
            "done": bool(self.done),
            "turn": int(self.turn),
            "meta": dict(self.meta),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> OpenEnvObservation:
        if not isinstance(payload, dict):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise ValueError("OpenEnvObservation payload must be a dict")
        text_raw = payload.get("text")
        text: str | None
        if text_raw is None:
            text = None
        elif isinstance(text_raw, str):
            text = text_raw
        else:
            text = str(text_raw)
        reward_raw = payload.get("reward", 0.0)
        try:
            reward = float(reward_raw) if reward_raw is not None else 0.0
        except (TypeError, ValueError):
            reward = 0.0
        done = bool(payload.get("done", False))
        turn_raw = payload.get("turn", 0)
        try:
            turn = int(turn_raw) if turn_raw is not None else 0
        except (TypeError, ValueError):
            turn = 0
        meta_raw: Any = payload.get("meta", {})
        meta: dict[str, Any]
        if isinstance(meta_raw, dict):
            meta_dict = cast(dict[Any, Any], meta_raw)
            meta = {str(k): v for k, v in meta_dict.items()}
        else:
            meta = {}
        return cls(
            text=text,
            tool_result=payload.get("tool_result"),
            reward=reward,
            done=done,
            turn=turn,
            meta=meta,
        )


# ---------------------------------------------------------------------------
# Server: wrap a local EnvironmentConnection
# ---------------------------------------------------------------------------


class OpenEnvServer:
    """Expose a local :class:`EnvironmentConnection` as an OpenEnv server.

    The three OpenEnv entrypoints — :meth:`reset`, :meth:`step`,
    :meth:`state` — drive the wrapped environment through its
    :class:`EnvironmentConnection` lifecycle. Tool dispatch is resolved by
    name against ``spec.tools``; unknown tools raise :class:`ValueError`
    (over HTTP this becomes a 400).

    The server does NOT own the connection lifecycle. Callers construct a
    ready environment (either via ``with env:`` or an explicit ``open()``)
    and hand it to the server. :meth:`close` is a no-op; callers close the
    wrapped env when they are done.
    """

    def __init__(
        self,
        connection: EnvironmentConnection,
        *,
        chain: InteractionChain | None = None,
    ) -> None:
        # The isinstance check defends against duck-typed call sites that
        # pass a raw BaseEnvironment (would skip FSM telemetry silently).
        if not isinstance(connection, EnvironmentConnection):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise TypeError(
                "OpenEnvServer requires an EnvironmentConnection, "
                f"got {type(connection).__name__}",
            )
        self._env: EnvironmentConnection = connection
        self._chain = chain
        if chain is not None:
            # Chain takes precedence over whatever the env had attached.
            self._env.attach_chain(chain)

    # -- properties ------------------------------------------------------

    @property
    def env(self) -> EnvironmentConnection:
        return self._env

    @property
    def connection_id(self) -> str:
        return self._env.connection_id

    @property
    def spec(self) -> EnvironmentSpec:
        return self._env.spec

    # -- OpenEnv entrypoints --------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        **kwargs: Any,
    ) -> OpenEnvObservation:
        """Reset the wrapped environment and return the initial observation.

        ``seed`` is accepted for OpenEnv compatibility; it is recorded on
        the telemetry step but not forwarded to the env (concrete envs
        rarely honour RNG seeds today).
        """
        # Ensure the env is at least open before reset. Opening an already
        # READY connection raises — honour the user's lifecycle without
        # forcing a specific ordering.
        if self._env.state == ConnectionState.INIT:
            self._env.open()

        payload = self._env.reset(**kwargs)
        text = payload if isinstance(payload, str) else None
        meta: dict[str, Any] = {
            "op": "reset",
            "seed": seed,
            "kwargs_keys": sorted(kwargs.keys()),
            "connection_id": self._env.connection_id,
        }
        obs = OpenEnvObservation(
            text=text,
            tool_result=None,
            reward=float(getattr(self._env, "reward", 0.0) or 0.0),
            done=bool(getattr(self._env, "done", False)),
            turn=int(self._env.turn_count),
            meta=meta,
        )
        self._record_step("openenv.reset", obs, success=True)
        return obs

    def step(self, action: OpenEnvAction | dict[str, Any]) -> OpenEnvObservation:
        """Dispatch an :class:`OpenEnvAction` to the wrapped environment.

        Parameters
        ----------
        action
            Either an :class:`OpenEnvAction` or a dict payload matching
            :meth:`OpenEnvAction.from_dict`.
        """
        if isinstance(action, dict):
            action = OpenEnvAction.from_dict(action)
        elif not isinstance(action, OpenEnvAction):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise TypeError(
                f"OpenEnvServer.step expects OpenEnvAction or dict, got {type(action).__name__}",
            )

        tool_name = action.tool
        if tool_name not in self._env.spec.tools:
            raise ValueError(
                f"Tool '{tool_name}' not declared in spec.tools for {self._env.spec.name}; "
                f"available: {sorted(self._env.spec.tools)}",
            )
        tool = getattr(self._env, tool_name, None)
        if tool is None or not callable(tool):
            raise AttributeError(
                f"Wrapped env {type(self._env).__name__} has no callable attribute '{tool_name}'",
            )

        success = True
        tool_result: Any = None
        try:
            with self._env.transact(tool_name):
                tool_result = tool(**action.args)
        except BaseException:  # noqa: BLE001 — telemetry then re-raise
            success = False
            err_meta: dict[str, Any] = {
                "op": "step",
                "tool": tool_name,
                "error": True,
                "connection_id": self._env.connection_id,
            }
            obs = OpenEnvObservation(
                text=None,
                tool_result=None,
                reward=float(getattr(self._env, "reward", 0.0) or 0.0),
                done=bool(getattr(self._env, "done", False)),
                turn=int(self._env.turn_count),
                meta=err_meta,
            )
            self._record_step("openenv.step", obs, success=success)
            raise

        text = tool_result if isinstance(tool_result, str) else None
        ok_meta: dict[str, Any] = {
            "op": "step",
            "tool": tool_name,
            "args_keys": sorted(action.args.keys()),
            "reasoning_present": action.reasoning is not None,
            "connection_id": self._env.connection_id,
        }
        obs = OpenEnvObservation(
            text=text,
            tool_result=tool_result,
            reward=float(getattr(self._env, "reward", 0.0) or 0.0),
            done=bool(getattr(self._env, "done", False)),
            turn=int(self._env.turn_count),
            meta=ok_meta,
        )
        self._record_step("openenv.step", obs, success=True)
        return obs

    def state(self) -> dict[str, Any]:
        """Return canonical state for checkpoint/resume.

        The shape is stable: ``connection_id``, ``spec``, ``turn_count``,
        ``history``, ``reward``, ``done``, ``connection_state``.
        """
        spec_payload: dict[str, Any] = {
            "lane": self._env.spec.lane.value,
            "name": self._env.spec.name,
            "tools": list(self._env.spec.tools),
            "max_turns": self._env.spec.max_turns,
            "reward_type": self._env.spec.reward_type,
            "multimodal": self._env.spec.multimodal,
            "dataset_columns": list(self._env.spec.dataset_columns),
        }
        return {
            "connection_id": self._env.connection_id,
            "connection_state": self._env.state.value,
            "spec": spec_payload,
            "turn_count": int(self._env.turn_count),
            "history": list(self._env.history),
            "reward": float(getattr(self._env, "reward", 0.0) or 0.0),
            "done": bool(getattr(self._env, "done", False)),
        }

    def serialize(self) -> str:
        """Return canonical JSON for checkpoint/resume.

        Output is ASCII-safe and sorted; equivalent environments in
        equivalent states serialize identically.
        """
        return json.dumps(self.state(), sort_keys=True, default=_json_default)

    # -- helpers ---------------------------------------------------------

    def _record_step(
        self,
        name: str,
        obs: OpenEnvObservation,
        *,
        success: bool,
    ) -> None:
        if self._chain is None:
            return
        try:
            self._chain.record(
                action=ActionType.EXTERNAL,
                name=name,
                input={
                    "connection_id": self._env.connection_id,
                    "spec_name": self._env.spec.name,
                    "meta": obs.meta,
                },
                output={
                    "turn": obs.turn,
                    "reward": obs.reward,
                    "done": obs.done,
                    "text_present": obs.text is not None,
                },
                success=success,
            )
        except (TypeError, AttributeError, ValueError):
            # Telemetry is best-effort. Malformed chains must never break
            # env step/reset.
            pass


# ---------------------------------------------------------------------------
# Client: consume a remote OpenEnv server over HTTP JSON-RPC
# ---------------------------------------------------------------------------


OPENENV_CLIENT_CONNECTION_SPEC: ConnectionSpec = ConnectionSpec(
    name="carl.env.openenv.client",
    scope=ConnectionScope.THREE_P,
    kind=ConnectionKind.ENVIRONMENT,
    direction=ConnectionDirection.BIDIRECTIONAL,
    transport=ConnectionTransport.HTTP,
    trust=ConnectionTrust.AUTHENTICATED,
    metadata={"protocol": "openenv"},
)


class OpenEnvClient(EnvironmentConnection):
    """Consume a remote OpenEnv server as a carl-studio environment.

    The remote server must expose three endpoints::

        POST /reset   body: {"seed": int | null, ...dataset cols}
        POST /step    body: OpenEnvAction.to_dict()
        GET  /state

    The client mirrors the wrapped server's declared spec on construction
    so TRL's environment_factory can still auto-discover tools. Tool calls
    are dispatched dynamically: ``client.<tool>(arg=...)`` round-trips
    through :meth:`step`.
    """

    connection_spec = OPENENV_CLIENT_CONNECTION_SPEC

    # Placeholder — the real spec is loaded from the remote on open().
    # A stub is required for BaseEnvironment's isinstance checks to work
    # before ``open()`` is called.
    spec = EnvironmentSpec(
        lane=EnvironmentLane.CODE,
        name="openenv.client",
        tools=(),
        max_turns=10,
        reward_type="binary",
        dataset_columns=(),
    )

    def __init__(
        self,
        base_url: str = "",
        *,
        chain: InteractionChain | None = None,
        timeout: float = 30.0,
        headers: dict[str, str] | None = None,
    ) -> None:
        # ``base_url`` is optional to preserve the zero-arg contract for
        # TRL's environment_factory auto-discovery. Callers who want to
        # actually talk to a server pass base_url explicitly.
        super().__init__()
        self._base_url = base_url.rstrip("/") if base_url else ""
        self._timeout = float(timeout)
        self._headers: dict[str, str] = dict(headers or {"content-type": "application/json"})
        if chain is not None:
            self.attach_chain(chain)
        self._httpx_client: Any = None
        self._remote_spec_loaded: bool = False

    # -- lifecycle -------------------------------------------------------

    def _env_connect(self) -> None:
        """Lazy-import httpx and open a session to the remote."""
        try:
            import httpx
        except ImportError as exc:
            raise ConnectionUnavailableError(
                "httpx is required for OpenEnvClient. Install with: pip install httpx",
                context={"package": "httpx"},
                cause=exc,
            ) from exc
        if not self._base_url:
            raise ConnectionUnavailableError(
                "OpenEnvClient.base_url is empty; cannot open a remote connection",
                context={"base_url": self._base_url},
            )
        self._httpx_client = httpx.Client(
            base_url=self._base_url,
            timeout=self._timeout,
            headers=self._headers,
        )
        # Pull remote spec on connect so TRL sees real tools post-open().
        try:
            self._load_remote_spec()
        except Exception as exc:  # noqa: BLE001
            # Leave the default stub in place; dynamic step dispatch still
            # works if the caller knows the tool name.
            self._record_client_warning("openenv.spec_load_failed", repr(exc))

    def _env_close(self) -> None:
        client = self._httpx_client
        self._httpx_client = None
        if client is not None:
            try:
                client.close()
            except Exception:  # noqa: BLE001
                pass

    # -- remote interactions --------------------------------------------

    def _load_remote_spec(self) -> None:
        payload = self._get_json("/state")
        spec_payload_raw: Any = payload.get("spec", {})
        if not isinstance(spec_payload_raw, dict):
            return
        spec_payload = cast(dict[str, Any], spec_payload_raw)
        try:
            lane_val: Any = spec_payload.get("lane", "code")
            lane = EnvironmentLane(lane_val) if isinstance(lane_val, str) else EnvironmentLane.CODE
            tools_raw: Any = spec_payload.get("tools", [])
            if isinstance(tools_raw, list):
                tools_list = cast(list[Any], tools_raw)
                tools: tuple[str, ...] = tuple(str(t) for t in tools_list)
            else:
                tools = ()
            cols_raw: Any = spec_payload.get("dataset_columns", [])
            if isinstance(cols_raw, list):
                cols_list = cast(list[Any], cols_raw)
                cols: tuple[str, ...] = tuple(str(c) for c in cols_list)
            else:
                cols = ()
            name_raw: Any = spec_payload.get("name", "openenv.client")
            name = str(name_raw)
            max_turns_raw: Any = spec_payload.get("max_turns", 10)
            max_turns = int(max_turns_raw) if max_turns_raw is not None else 10
            reward_type_raw: Any = spec_payload.get("reward_type", "binary")
            reward_type = str(reward_type_raw)
            multimodal_raw: Any = spec_payload.get("multimodal", False)
            multimodal = bool(multimodal_raw)
            new_spec = EnvironmentSpec(
                lane=lane,
                name=name,
                tools=tools,
                max_turns=max_turns,
                reward_type=reward_type,
                multimodal=multimodal,
                dataset_columns=cols,
            )
            # Per-instance spec override (TRL reads type(self).spec in some
            # paths, but also self.spec in others; set both for safety).
            object.__setattr__(self, "spec", new_spec)
            self._remote_spec_loaded = True
        except (TypeError, ValueError, KeyError):
            # Keep the stub; best-effort only.
            pass

    def reset(self, **kwargs: Any) -> str | None:
        """Reset the remote env. Falls through to base telemetry."""
        if self._httpx_client is None:
            # Standalone reset without an open connection is allowed by
            # BaseEnvironment's contract — record the event but do not
            # attempt the remote call.
            return super().reset(**kwargs)

        payload: dict[str, Any] = {"seed": kwargs.pop("seed", None), **_safe_kwargs(kwargs)}
        with self.transact("reset"):
            response = self._post_json("/reset", payload)
        obs = OpenEnvObservation.from_dict(response)
        self.reward = obs.reward
        self.done = obs.done
        self._turn_count = obs.turn
        # Emit parent-level telemetry (and reset local counters first).
        super().reset(**kwargs)
        # Re-apply the remote-authoritative values after super() clobbers them.
        self.reward = obs.reward
        self.done = obs.done
        self._turn_count = obs.turn
        return obs.text

    def step(self, action: OpenEnvAction | dict[str, Any]) -> OpenEnvObservation:
        """Send a step action to the remote server and return the observation."""
        if self._httpx_client is None:
            raise ConnectionUnavailableError(
                "OpenEnvClient.step called before open(); call .open() first",
                context={"connection_id": self.connection_id},
            )
        if isinstance(action, dict):
            action = OpenEnvAction.from_dict(action)
        elif not isinstance(action, OpenEnvAction):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise TypeError(
                f"OpenEnvClient.step expects OpenEnvAction or dict, got {type(action).__name__}",
            )
        with self.transact(f"step:{action.tool}"):
            response = self._post_json("/step", action.to_dict())
        obs = OpenEnvObservation.from_dict(response)
        self.reward = obs.reward
        self.done = obs.done
        self._turn_count = obs.turn
        return obs

    def remote_state(self) -> dict[str, Any]:
        """Return the remote server's /state payload verbatim."""
        if self._httpx_client is None:
            raise ConnectionUnavailableError(
                "OpenEnvClient.remote_state called before open(); call .open() first",
                context={"connection_id": self.connection_id},
            )
        return self._get_json("/state")

    # -- HTTP plumbing ---------------------------------------------------

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        if self._httpx_client is None:
            raise ConnectionUnavailableError(
                "OpenEnvClient HTTP session not open",
                context={"path": path},
            )
        response = self._httpx_client.post(path, json=payload)
        if hasattr(response, "raise_for_status"):
            response.raise_for_status()
        data_raw: Any = response.json()
        if not isinstance(data_raw, dict):
            raise ValueError(f"OpenEnv {path} returned non-dict: {type(data_raw).__name__}")
        data_dict = cast(dict[Any, Any], data_raw)
        return {str(k): v for k, v in data_dict.items()}

    def _get_json(self, path: str) -> dict[str, Any]:
        if self._httpx_client is None:
            raise ConnectionUnavailableError(
                "OpenEnvClient HTTP session not open",
                context={"path": path},
            )
        response = self._httpx_client.get(path)
        if hasattr(response, "raise_for_status"):
            response.raise_for_status()
        data_raw: Any = response.json()
        if not isinstance(data_raw, dict):
            raise ValueError(f"OpenEnv {path} returned non-dict: {type(data_raw).__name__}")
        data_dict = cast(dict[Any, Any], data_raw)
        return {str(k): v for k, v in data_dict.items()}

    def _record_client_warning(self, name: str, detail: str) -> None:
        chain = self.chain
        if chain is None:
            return
        try:
            chain.record(
                action=ActionType.EXTERNAL,
                name=name,
                input={
                    "connection_id": self.connection_id,
                    "base_url": self._base_url,
                    "detail": detail[:500],
                },
                success=False,
            )
        except (TypeError, AttributeError, ValueError):
            pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _json_default(obj: Any) -> Any:
    """JSON fallback for objects state() may include."""
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        return obj.to_dict()
    if hasattr(obj, "value"):
        return obj.value
    return str(obj)


def _safe_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Filter to JSON-serializable kwargs (best-effort)."""
    out: dict[str, Any] = {}
    for k, v in kwargs.items():
        try:
            json.dumps(v, default=_json_default)
        except (TypeError, ValueError):
            out[k] = repr(v)
            continue
        out[k] = v
    return out


__all__ = [
    "OpenEnvAction",
    "OpenEnvObservation",
    "OpenEnvServer",
    "OpenEnvClient",
    "OPENENV_CLIENT_CONNECTION_SPEC",
]
