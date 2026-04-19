"""Tests for HF Environments Hub integration — from_hub + publish_to_hub."""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any

import pytest

from carl_core.connection import ConnectionUnavailableError, reset_registry

from carl_studio.environments import registry as registry_mod
from carl_studio.environments.protocol import (
    BaseEnvironment,
    EnvironmentLane,
    EnvironmentSpec,
)


@pytest.fixture(autouse=True)
def _clean_state():  # pyright: ignore[reportUnusedFunction]
    reset_registry()
    # Save/restore the registry so mocked envs don't leak across tests.
    saved = dict(registry_mod._REGISTRY)  # pyright: ignore[reportPrivateUsage]
    registry_mod._REGISTRY.clear()  # pyright: ignore[reportPrivateUsage]
    yield
    registry_mod._REGISTRY.clear()  # pyright: ignore[reportPrivateUsage]
    registry_mod._REGISTRY.update(saved)  # pyright: ignore[reportPrivateUsage]
    reset_registry()


# ---------------------------------------------------------------------------
# Sample on-disk env for snapshot_download to emit
# ---------------------------------------------------------------------------


_SAMPLE_MODULE = textwrap.dedent(
    """
    from __future__ import annotations

    from carl_studio.environments.protocol import (
        BaseEnvironment,
        EnvironmentLane,
        EnvironmentSpec,
    )


    class HubDemoEnv(BaseEnvironment):
        spec = EnvironmentSpec(
            lane=EnvironmentLane.CODE,
            name="hub-demo",
            tools=("do_thing",),
            max_turns=3,
            reward_type="binary",
            dataset_columns=("task",),
        )

        def reset(self, **kwargs):
            super().reset(**kwargs)
            return kwargs.get("task")

        def do_thing(self, payload: str) -> str:
            '''Stub tool.

            Args:
                payload: anything.
            '''
            self._record_turn("do_thing", {"payload": payload}, payload)
            return payload
    """
).strip()


_SAMPLE_MANIFEST = textwrap.dedent(
    """
    schema_version: 1
    carl_studio_generated: true
    lane: code
    name: hub-demo
    tools:
      - do_thing
    max_turns: 3
    reward_type: binary
    multimodal: false
    system_prompt: ""
    dataset_columns:
      - task
    class_name: HubDemoEnv
    module: hub_demo.py
    version: 0.0.1
    """
).strip()


def _seed_snapshot(tmp_path: Path) -> Path:
    """Build a pretend Hub snapshot on disk."""
    target = tmp_path / "snapshot"
    target.mkdir()
    (target / "carl.env.yaml").write_text(_SAMPLE_MANIFEST, encoding="utf-8")
    (target / "hub_demo.py").write_text(_SAMPLE_MODULE, encoding="utf-8")
    return target


# ---------------------------------------------------------------------------
# from_hub
# ---------------------------------------------------------------------------


class TestFromHub:
    def test_rejects_empty_name(self):
        with pytest.raises(ValueError):
            registry_mod.from_hub("")

    def test_missing_huggingface_hub_raises_connection_error(
        self, monkeypatch: pytest.MonkeyPatch,
    ):
        # Simulate huggingface_hub being absent by making the import raise.
        # The helper catches ImportError and translates to ConnectionUnavailableError.
        def _raise() -> Any:
            raise ConnectionUnavailableError(
                "huggingface_hub is required for environment hub access. "
                "Install with: pip install huggingface-hub",
                context={"package": "huggingface_hub"},
            )

        monkeypatch.setattr(registry_mod, "_import_snapshot_download", _raise)
        with pytest.raises(ConnectionUnavailableError):
            registry_mod.from_hub("x/y")

    def test_snapshot_download_failure_wraps_error(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ):
        def _fake_download(**_kwargs: Any) -> str:
            raise RuntimeError("network down")

        monkeypatch.setattr(
            registry_mod, "_import_snapshot_download", lambda: _fake_download,
        )
        with pytest.raises(ConnectionUnavailableError, match="network down"):
            registry_mod.from_hub("carl-ai/hub-demo", cache_dir=tmp_path)

    def test_missing_manifest_raises(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ):
        snap = tmp_path / "snap"
        snap.mkdir()
        (snap / "hub_demo.py").write_text(_SAMPLE_MODULE, encoding="utf-8")

        def _fake_download(**_kwargs: Any) -> str:
            return str(snap)

        monkeypatch.setattr(
            registry_mod, "_import_snapshot_download", lambda: _fake_download,
        )
        with pytest.raises(ConnectionUnavailableError, match="missing carl.env.yaml"):
            registry_mod.from_hub("carl-ai/hub-demo", cache_dir=tmp_path)

    def test_loads_module_and_registers(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ):
        snap = _seed_snapshot(tmp_path)

        def _fake_download(**_kwargs: Any) -> str:
            return str(snap)

        monkeypatch.setattr(
            registry_mod, "_import_snapshot_download", lambda: _fake_download,
        )

        cls = registry_mod.from_hub("carl-ai/hub-demo", cache_dir=tmp_path)
        assert issubclass(cls, BaseEnvironment)
        assert cls.spec.name == "hub-demo"
        assert registry_mod.is_registered("hub-demo")

    def test_idempotent_second_pull(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ):
        snap = _seed_snapshot(tmp_path)

        def _fake_download(**_kwargs: Any) -> str:
            return str(snap)

        monkeypatch.setattr(
            registry_mod, "_import_snapshot_download", lambda: _fake_download,
        )
        registry_mod.from_hub("carl-ai/hub-demo", cache_dir=tmp_path)
        # Second pull must not raise "already registered" — the helper
        # returns the existing class.
        registry_mod.from_hub("carl-ai/hub-demo", cache_dir=tmp_path)

    def test_resolves_hf_token_from_get_token(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ):
        snap = _seed_snapshot(tmp_path)

        observed_tokens: list[Any] = []

        def _fake_download(**kwargs: Any) -> str:
            observed_tokens.append(kwargs.get("token"))
            return str(snap)

        def _fake_get_token() -> str:
            return "hf_fake_token"

        monkeypatch.setattr(
            registry_mod, "_import_snapshot_download", lambda: _fake_download,
        )
        # huggingface_hub.get_token is looked up at call time; fake the
        # real module so _resolve_hf_token picks it up.
        import huggingface_hub as hf_hub  # type: ignore

        monkeypatch.setattr(hf_hub, "get_token", _fake_get_token, raising=False)
        monkeypatch.delenv("HF_TOKEN", raising=False)

        registry_mod.from_hub("carl-ai/hub-demo", cache_dir=tmp_path)
        assert observed_tokens == ["hf_fake_token"]

    def test_explicit_token_beats_get_token(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ):
        snap = _seed_snapshot(tmp_path)
        observed_tokens: list[Any] = []

        def _fake_download(**kwargs: Any) -> str:
            observed_tokens.append(kwargs.get("token"))
            return str(snap)

        monkeypatch.setattr(
            registry_mod, "_import_snapshot_download", lambda: _fake_download,
        )
        import huggingface_hub as hf_hub  # type: ignore

        monkeypatch.setattr(
            hf_hub, "get_token", lambda: "hf_default", raising=False,
        )

        registry_mod.from_hub(
            "carl-ai/hub-demo", token="hf_explicit", cache_dir=tmp_path,
        )
        assert observed_tokens == ["hf_explicit"]


# ---------------------------------------------------------------------------
# _load_env_manifest
# ---------------------------------------------------------------------------


class TestLoadManifest:
    def test_parses_valid_manifest(self, tmp_path: Path):
        path = tmp_path / "carl.env.yaml"
        path.write_text(_SAMPLE_MANIFEST, encoding="utf-8")
        spec = registry_mod._load_env_manifest(path)  # pyright: ignore[reportPrivateUsage]
        assert spec.name == "hub-demo"
        assert spec.lane == EnvironmentLane.CODE
        assert spec.tools == ("do_thing",)
        assert spec.dataset_columns == ("task",)

    def test_malformed_yaml_raises(self, tmp_path: Path):
        path = tmp_path / "carl.env.yaml"
        path.write_text("foo: [unclosed", encoding="utf-8")
        with pytest.raises(ConnectionUnavailableError):
            registry_mod._load_env_manifest(path)  # pyright: ignore[reportPrivateUsage]

    def test_non_mapping_raises(self, tmp_path: Path):
        path = tmp_path / "carl.env.yaml"
        path.write_text("- 1\n- 2\n", encoding="utf-8")
        with pytest.raises(ConnectionUnavailableError):
            registry_mod._load_env_manifest(path)  # pyright: ignore[reportPrivateUsage]

    def test_missing_required_field_raises(self, tmp_path: Path):
        path = tmp_path / "carl.env.yaml"
        path.write_text("lane: code\n", encoding="utf-8")  # no name / tools
        with pytest.raises(ConnectionUnavailableError):
            registry_mod._load_env_manifest(path)  # pyright: ignore[reportPrivateUsage]


# ---------------------------------------------------------------------------
# publish_to_hub
# ---------------------------------------------------------------------------


class _FakeCommitInfo:
    commit_url = "https://huggingface.co/carl-ai/hub-demo/commit/abc"


class _FakeHfApi:
    def __init__(self, *, token: str | None = None) -> None:  # noqa: ARG002
        pass


class TestPublishToHub:
    def _fake_publish_api(
        self,
        calls: dict[str, Any],
    ) -> tuple[Any, Any, Any]:
        def _fake_upload(*, folder_path: str, repo_id: str, token: Any, commit_message: str):
            calls["upload"] = {
                "folder_path": folder_path,
                "repo_id": repo_id,
                "token": token,
                "commit_message": commit_message,
            }
            # Snapshot what got written.
            files = sorted(Path(folder_path).iterdir())
            calls["files"] = [f.name for f in files]
            return _FakeCommitInfo()

        def _fake_create(**kwargs: Any) -> None:
            calls["create"] = kwargs

        return _FakeHfApi, _fake_upload, _fake_create

    def test_rejects_non_env_class(self):
        with pytest.raises(TypeError):
            registry_mod.publish_to_hub(object, "x/y")  # type: ignore[arg-type]

    def test_rejects_empty_repo_id(self, tmp_path: Path):
        class DummyEnv(BaseEnvironment):
            spec = EnvironmentSpec(
                lane=EnvironmentLane.CODE,
                name="dummy-env",
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

        with pytest.raises(ValueError):
            registry_mod.publish_to_hub(DummyEnv, "")

    def test_packages_manifest_and_source(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ):
        # Write the env class into a real module file on disk so inspect
        # can find its source.
        env_file = tmp_path / "env_src.py"
        env_file.write_text(
            textwrap.dedent(
                """
                from __future__ import annotations
                from carl_studio.environments.protocol import (
                    BaseEnvironment, EnvironmentLane, EnvironmentSpec,
                )


                class PackagedEnv(BaseEnvironment):
                    spec = EnvironmentSpec(
                        lane=EnvironmentLane.CODE,
                        name="packaged-env",
                        tools=("noop",),
                        dataset_columns=(),
                    )

                    def reset(self, **kwargs):
                        return super().reset(**kwargs)

                    def noop(self) -> str:
                        '''No-op.

                        Args:
                        '''
                        return ""
                """
            ).strip(),
            encoding="utf-8",
        )

        # Dynamically load the module so inspect.getsourcefile returns env_file.
        import importlib.util
        import sys

        module_spec = importlib.util.spec_from_file_location("tmp_packaged_env_mod", env_file)
        assert module_spec is not None and module_spec.loader is not None
        module = importlib.util.module_from_spec(module_spec)
        sys.modules["tmp_packaged_env_mod"] = module
        try:
            module_spec.loader.exec_module(module)
            env_cls = module.PackagedEnv
        except BaseException:
            sys.modules.pop("tmp_packaged_env_mod", None)
            raise

        calls: dict[str, Any] = {}
        monkeypatch.setattr(
            registry_mod,
            "_import_hub_publish_api",
            lambda: self._fake_publish_api(calls),
        )
        # Redirect the staging root to a clean temp path.
        monkeypatch.setattr(registry_mod, "_DEFAULT_CACHE_DIR", tmp_path / "cache")

        url = registry_mod.publish_to_hub(env_cls, "carl-ai/packaged", private=True)
        assert url.startswith("https://huggingface.co")
        assert "upload" in calls
        assert calls["upload"]["repo_id"] == "carl-ai/packaged"
        # The staged folder must contain the manifest and the source file.
        files = calls["files"]
        assert "carl.env.yaml" in files
        assert "env_src.py" in files
        # And the create_repo call should carry private=True.
        assert calls["create"]["private"] is True

    def test_manifest_round_trip_parseable(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ):
        """Written manifest must parse back via _load_env_manifest."""
        env_file = tmp_path / "env_src.py"
        env_file.write_text(
            textwrap.dedent(
                """
                from __future__ import annotations
                from carl_studio.environments.protocol import (
                    BaseEnvironment, EnvironmentLane, EnvironmentSpec,
                )


                class PackagedEnv(BaseEnvironment):
                    spec = EnvironmentSpec(
                        lane=EnvironmentLane.CODE,
                        name="round-trip-env",
                        tools=("noop",),
                        max_turns=5,
                        reward_type="binary",
                        system_prompt="demo",
                        dataset_columns=("task",),
                    )

                    def reset(self, **kwargs):
                        return super().reset(**kwargs)

                    def noop(self) -> str:
                        '''No-op.

                        Args:
                        '''
                        return ""
                """
            ).strip(),
            encoding="utf-8",
        )
        import importlib.util
        import sys

        module_spec = importlib.util.spec_from_file_location("tmp_rt_env", env_file)
        assert module_spec is not None and module_spec.loader is not None
        module = importlib.util.module_from_spec(module_spec)
        sys.modules["tmp_rt_env"] = module
        try:
            module_spec.loader.exec_module(module)
            env_cls = module.PackagedEnv
        except BaseException:
            sys.modules.pop("tmp_rt_env", None)
            raise

        calls: dict[str, Any] = {}
        monkeypatch.setattr(
            registry_mod,
            "_import_hub_publish_api",
            lambda: self._fake_publish_api(calls),
        )
        monkeypatch.setattr(registry_mod, "_DEFAULT_CACHE_DIR", tmp_path / "cache")

        registry_mod.publish_to_hub(env_cls, "carl-ai/round-trip")

        staged_folder = Path(calls["upload"]["folder_path"])
        manifest_path = staged_folder / "carl.env.yaml"
        assert manifest_path.exists()
        spec = registry_mod._load_env_manifest(manifest_path)  # pyright: ignore[reportPrivateUsage]
        assert spec.name == "round-trip-env"
        assert spec.tools == ("noop",)
        assert spec.max_turns == 5
        assert spec.system_prompt == "demo"
