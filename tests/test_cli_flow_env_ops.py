"""Tests for ``/pull-env`` and ``/publish-env`` flow operations."""

from __future__ import annotations

from typing import Any, cast

import pytest

from carl_core.interaction import ActionType, InteractionChain

from carl_studio.cli import operations as ops_mod
from carl_studio.environments import registry as registry_mod
from carl_studio.environments.protocol import (
    BaseEnvironment,
    EnvironmentLane,
    EnvironmentSpec,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_registry():  # pyright: ignore[reportUnusedFunction]
    saved = dict(registry_mod._REGISTRY)  # pyright: ignore[reportPrivateUsage]
    registry_mod._REGISTRY.clear()  # pyright: ignore[reportPrivateUsage]
    yield
    registry_mod._REGISTRY.clear()  # pyright: ignore[reportPrivateUsage]
    registry_mod._REGISTRY.update(saved)  # pyright: ignore[reportPrivateUsage]


# ---------------------------------------------------------------------------
# Ops registered
# ---------------------------------------------------------------------------


class TestOpsRegistered:
    def test_pull_env_registered(self):
        assert "pull-env" in ops_mod.list_operations()
        assert ops_mod.get_operation("pull-env") is not None

    def test_publish_env_registered(self):
        assert "publish-env" in ops_mod.list_operations()
        assert ops_mod.get_operation("publish-env") is not None


# ---------------------------------------------------------------------------
# /pull-env
# ---------------------------------------------------------------------------


class _FakeEnv(BaseEnvironment):
    spec = EnvironmentSpec(
        lane=EnvironmentLane.CODE,
        name="flow-demo",
        tools=("noop",),
        dataset_columns=(),
    )

    def reset(self, **kwargs: Any) -> str | None:
        return super().reset(**kwargs)

    def noop(self) -> str:
        """No-op.

        Args:
        """
        return ""


class TestPullEnv:
    def test_missing_argument_records_failure(self):
        chain = InteractionChain()
        op = ops_mod.OPERATIONS["pull-env"]
        op(chain, [])
        step = chain.last()
        assert step is not None
        assert step.name == "pull-env"
        assert step.success is False
        assert step.output.get("exit_code") == 2

    def test_successful_pull_records_spec(self, monkeypatch: pytest.MonkeyPatch):
        def _fake_from_hub(name: str, *, revision: str | None = None, **_kwargs: Any) -> type:
            assert name == "carl-ai/flow-demo"
            assert revision is None
            return _FakeEnv

        monkeypatch.setattr(registry_mod, "from_hub", _fake_from_hub)
        chain = InteractionChain()
        op = ops_mod.OPERATIONS["pull-env"]
        op(chain, ["carl-ai/flow-demo"])
        step = chain.last()
        assert step is not None
        assert step.success is True
        assert step.action == ActionType.CLI_CMD
        assert step.output["name"] == "flow-demo"
        assert step.output["lane"] == "code"
        assert step.output["tools"] == ["noop"]

    def test_revision_argument_passes_through(self, monkeypatch: pytest.MonkeyPatch):
        captured: dict[str, Any] = {}

        def _fake_from_hub(name: str, *, revision: str | None = None, **_kwargs: Any) -> type:
            captured["name"] = name
            captured["revision"] = revision
            return _FakeEnv

        monkeypatch.setattr(registry_mod, "from_hub", _fake_from_hub)
        chain = InteractionChain()
        op = ops_mod.OPERATIONS["pull-env"]
        op(chain, ["carl-ai/flow-demo", "--revision", "v1.2"])
        assert captured["revision"] == "v1.2"

    def test_failure_in_from_hub_records_error(self, monkeypatch: pytest.MonkeyPatch):
        def _fake_from_hub(*a: Any, **k: Any) -> Any:  # noqa: ARG001
            raise RuntimeError("no network")

        monkeypatch.setattr(registry_mod, "from_hub", _fake_from_hub)
        chain = InteractionChain()
        op = ops_mod.OPERATIONS["pull-env"]
        op(chain, ["carl-ai/flow-demo"])
        step = chain.last()
        assert step is not None
        assert step.success is False
        assert "no network" in step.output["error"]


# ---------------------------------------------------------------------------
# /publish-env
# ---------------------------------------------------------------------------


class TestPublishEnv:
    def test_missing_args_records_failure(self):
        chain = InteractionChain()
        op = ops_mod.OPERATIONS["publish-env"]
        op(chain, ["only-one"])
        step = chain.last()
        assert step is not None
        assert step.success is False

    def test_publish_flow(self, monkeypatch: pytest.MonkeyPatch):
        registry_mod.register_environment(_FakeEnv)

        captured: dict[str, Any] = {}

        def _fake_publish(cls: type, repo_id: str, *, private: bool = False, **_kwargs: Any) -> str:
            captured["cls"] = cls.__name__
            captured["repo_id"] = repo_id
            captured["private"] = private
            return "https://huggingface.co/carl-ai/flow-demo/commit/xyz"

        monkeypatch.setattr(registry_mod, "publish_to_hub", _fake_publish)
        chain = InteractionChain()
        op = ops_mod.OPERATIONS["publish-env"]
        op(chain, ["flow-demo", "carl-ai/flow-demo"])
        step = chain.last()
        assert step is not None
        assert step.success is True
        assert step.output["repo_id"] == "carl-ai/flow-demo"
        assert step.output["private"] is False
        assert "huggingface.co" in step.output["commit_url"]
        assert captured["cls"] == "_FakeEnv"

    def test_publish_private_flag(self, monkeypatch: pytest.MonkeyPatch):
        registry_mod.register_environment(_FakeEnv)

        captured: dict[str, Any] = {}

        def _fake_publish(cls: type, repo_id: str, *, private: bool = False, **_kwargs: Any) -> str:
            captured["private"] = private
            return "https://huggingface.co/repo/commit/xxx"

        monkeypatch.setattr(registry_mod, "publish_to_hub", _fake_publish)
        chain = InteractionChain()
        op = ops_mod.OPERATIONS["publish-env"]
        op(chain, ["flow-demo", "carl-ai/flow-demo", "--private"])
        step = chain.last()
        assert step is not None
        assert step.success is True
        assert captured["private"] is True

    def test_publish_unknown_local_name_records_error(
        self, monkeypatch: pytest.MonkeyPatch,
    ):
        # Don't register the env — publish should surface the KeyError.
        def _should_not_be_called(*a: Any, **k: Any) -> str:  # noqa: ARG001
            return cast(str, pytest.fail("publish_to_hub should not run"))

        monkeypatch.setattr(registry_mod, "publish_to_hub", _should_not_be_called)

        chain = InteractionChain()
        op = ops_mod.OPERATIONS["publish-env"]
        op(chain, ["not-registered", "carl-ai/nowhere"])
        step = chain.last()
        assert step is not None
        assert step.success is False
        assert "Unknown" in step.output["error"] or "not-registered" in step.output["error"]
