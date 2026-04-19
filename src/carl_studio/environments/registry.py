"""Environment registry -- discover and look up CARL environments by name or lane.

In addition to the local registry (``register_environment`` / ``get_environment``),
this module provides a bridge to the Hugging Face Environments Hub:

* :func:`from_hub` pulls an environment from HF Hub (a repo with
  ``carl.env.yaml`` + a Python module), caches it under
  ``~/.carl/environments/<name>/<revision>``, dynamically imports the
  class, and registers it.
* :func:`publish_to_hub` packages a local :class:`EnvironmentConnection`
  subclass into a Hub-publishable repo (manifest + source file) and
  uploads it.

``huggingface_hub`` is a lazy import. Absence => :class:`ConnectionUnavailableError`
with an install hint.
"""

from __future__ import annotations

import importlib.util
import inspect
import os
import sys
from pathlib import Path
from typing import Any, Type, TypeVar, cast

from carl_core.connection import ConnectionUnavailableError

from carl_studio.environments.protocol import BaseEnvironment, EnvironmentLane, EnvironmentSpec

_REGISTRY: dict[str, Type[BaseEnvironment]] = {}

# Preserve the decorated class's concrete type so that ``@register_environment``
# does not widen subclasses (e.g. CodingSandboxEnv) back to BaseEnvironment.
_EnvT = TypeVar("_EnvT", bound=Type[BaseEnvironment])

_DEFAULT_CACHE_DIR = Path(os.environ.get("CARL_HOME", "~/.carl")).expanduser() / "environments"

_MANIFEST_NAME = "carl.env.yaml"


def register_environment(cls: _EnvT) -> _EnvT:
    """Class decorator to register an environment.

    Usage::

        @register_environment
        class MySandbox(BaseEnvironment):
            spec = EnvironmentSpec(lane=EnvironmentLane.CODE, name="my-sandbox", ...)
    """
    if not hasattr(cls, "spec") or not isinstance(cls.spec, EnvironmentSpec):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise TypeError(f"{cls.__name__} must have a class-level `spec: EnvironmentSpec`")
    if cls.spec.name in _REGISTRY:
        raise ValueError(
            f"Environment '{cls.spec.name}' already registered by "
            f"{_REGISTRY[cls.spec.name].__name__}",
        )
    _REGISTRY[cls.spec.name] = cls
    return cls


def get_environment(name: str) -> Type[BaseEnvironment]:
    """Look up environment class by name.

    Raises KeyError if name is not registered.
    """
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys())) or "(none)"
        raise KeyError(f"Unknown environment: '{name}'. Available: {available}")
    return _REGISTRY[name]


def get_environment_factory(name: str) -> Type[BaseEnvironment]:
    """Return an environment class suitable for TRL's environment_factory.

    Identical to get_environment but named for clarity when passing to GRPOTrainer.
    """
    return get_environment(name)


def list_environments(lane: EnvironmentLane | None = None) -> list[EnvironmentSpec]:
    """List registered environments, optionally filtered by lane."""
    specs = [cls.spec for cls in _REGISTRY.values()]
    if lane is not None:
        specs = [s for s in specs if s.lane == lane]
    return sorted(specs, key=lambda s: s.name)


def is_registered(name: str) -> bool:
    """Check if an environment name is registered."""
    return name in _REGISTRY


def clear_registry() -> None:
    """Clear all registered environments. For testing only."""
    _REGISTRY.clear()


# ===========================================================================
# HF Environments Hub integration
# ===========================================================================


def from_hub(
    name: str,
    *,
    revision: str | None = None,
    token: str | None = None,
    cache_dir: Path | None = None,
) -> Type[BaseEnvironment]:
    """Pull an environment from the HF Environments Hub and register it.

    Parameters
    ----------
    name
        Full HF Hub repo id, e.g. ``"carl-ai/code-sandbox-env"``.
    revision
        Optional git revision (branch, tag, or commit hash). Defaults to
        the repo's default branch.
    token
        Optional HF auth token. If ``None``, falls back to
        ``huggingface_hub.get_token()`` then ``HF_TOKEN``. Private repos
        require a token.
    cache_dir
        Root directory for the local cache. Defaults to
        ``~/.carl/environments``. The class is cached under
        ``<cache_dir>/<name>/<revision>/``.

    Returns
    -------
    Type[BaseEnvironment]
        The loaded class, registered in the local registry.

    Raises
    ------
    ConnectionUnavailableError
        If ``huggingface_hub`` is not installed or the snapshot download
        fails.
    """
    if not isinstance(name, str) or not name:  # pyright: ignore[reportUnnecessaryIsInstance]
        raise ValueError("from_hub: name must be a non-empty string")

    snapshot_download: Any = _import_snapshot_download()

    effective_token = _resolve_hf_token(token)
    effective_revision = revision or "main"
    root = (cache_dir or _DEFAULT_CACHE_DIR).expanduser()
    local_dir = root / name.replace("/", "__") / effective_revision
    local_dir.mkdir(parents=True, exist_ok=True)

    try:
        snapshot_path_raw = snapshot_download(
            repo_id=name,
            revision=effective_revision,
            local_dir=str(local_dir),
            token=effective_token,
        )
    except Exception as exc:  # noqa: BLE001
        raise ConnectionUnavailableError(
            f"Failed to pull environment '{name}' from HF Hub: {exc}",
            context={"name": name, "revision": effective_revision},
            cause=exc,
        ) from exc

    snapshot_path = Path(str(snapshot_path_raw))
    manifest_path = snapshot_path / _MANIFEST_NAME
    if not manifest_path.exists():
        raise ConnectionUnavailableError(
            f"Hub repo '{name}' is missing {_MANIFEST_NAME}",
            context={"name": name, "snapshot": str(snapshot_path)},
        )

    spec = _load_env_manifest(manifest_path)
    module_file = _resolve_module_file(snapshot_path, manifest_path)
    env_cls = _load_env_class(module_file, spec)

    # Register (or fetch if already registered — keeps idempotence).
    if not is_registered(spec.name):
        register_environment(env_cls)
    return env_cls


def publish_to_hub(
    env_cls: Type[BaseEnvironment],
    repo_id: str,
    *,
    token: str | None = None,
    private: bool = False,
    commit_message: str = "carl-studio: publish environment",
) -> str:
    """Package an environment class and upload it to HF Hub.

    Writes a temporary directory containing:

    * ``carl.env.yaml`` — manifest with lane, name, tools, version
    * ``<module_basename>.py`` — the source file hosting ``env_cls``

    The manifest also records the fully-qualified class name so
    :func:`from_hub` can reconstruct the import path.

    Returns
    -------
    str
        The commit URL returned by the Hub.
    """
    if not isinstance(env_cls, type) or not issubclass(env_cls, BaseEnvironment):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise TypeError(
            f"publish_to_hub requires a BaseEnvironment subclass, "
            f"got {env_cls!r}",
        )
    if not hasattr(env_cls, "spec") or not isinstance(env_cls.spec, EnvironmentSpec):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise TypeError(
            f"{env_cls.__name__} is missing a valid `spec: EnvironmentSpec`",
        )
    if not isinstance(repo_id, str) or not repo_id:  # pyright: ignore[reportUnnecessaryIsInstance]
        raise ValueError("publish_to_hub: repo_id must be a non-empty string")

    _HfApi, hf_hub_upload_folder, hf_repo_create = _import_hub_publish_api()

    src_file = _source_file_for(env_cls)
    if src_file is None:
        raise ConnectionUnavailableError(
            f"Cannot locate source file for {env_cls.__module__}.{env_cls.__name__}",
            context={"class": f"{env_cls.__module__}.{env_cls.__name__}"},
        )

    effective_token = _resolve_hf_token(token)
    # Construct HfApi to surface bad tokens early even though we don't
    # call its instance methods directly (upload_folder / create_repo are
    # the module-level helpers).
    _HfApi(token=effective_token)

    # Stage the payload in a temp directory under the cache dir so we can
    # keep a breadcrumb for debugging after publish.
    stage_root = _DEFAULT_CACHE_DIR / "publish" / repo_id.replace("/", "__")
    stage_root.mkdir(parents=True, exist_ok=True)

    manifest_text = _render_manifest(env_cls)
    (stage_root / _MANIFEST_NAME).write_text(manifest_text, encoding="utf-8")

    # Copy the source file verbatim — preserves imports the user wrote.
    src_path = Path(src_file)
    dest_name = src_path.name
    (stage_root / dest_name).write_text(src_path.read_text(encoding="utf-8"), encoding="utf-8")

    # Ensure the repo exists before upload (idempotent).
    try:
        hf_repo_create(
            repo_id=repo_id,
            token=effective_token,
            private=private,
            exist_ok=True,
            repo_type="model",  # HF Environments Hub currently uses model repos
        )
    except Exception as exc:  # noqa: BLE001
        raise ConnectionUnavailableError(
            f"Failed to create/open repo '{repo_id}' on HF Hub: {exc}",
            context={"repo_id": repo_id, "private": private},
            cause=exc,
        ) from exc

    try:
        commit_ref = hf_hub_upload_folder(
            folder_path=str(stage_root),
            repo_id=repo_id,
            token=effective_token,
            commit_message=commit_message,
        )
    except Exception as exc:  # noqa: BLE001
        raise ConnectionUnavailableError(
            f"Failed to upload to '{repo_id}': {exc}",
            context={"repo_id": repo_id},
            cause=exc,
        ) from exc

    # Unwrap CommitInfo-like result to a URL string.
    if isinstance(commit_ref, str):
        return commit_ref
    url = getattr(commit_ref, "commit_url", None) or getattr(commit_ref, "url", None)
    return str(url) if url else f"https://huggingface.co/{repo_id}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_env_manifest(path: Path) -> EnvironmentSpec:
    """Parse a ``carl.env.yaml`` manifest into an :class:`EnvironmentSpec`."""
    try:
        import yaml  # lazy — pyyaml is already a hard dep, but keep the module light
    except ImportError as exc:  # pragma: no cover — pyyaml is a declared dep
        raise ConnectionUnavailableError(
            "pyyaml is required to read carl.env.yaml manifests. "
            "Install with: pip install pyyaml",
            context={"package": "pyyaml"},
            cause=exc,
        ) from exc

    raw_text = path.read_text(encoding="utf-8")
    try:
        raw_data: Any = yaml.safe_load(raw_text)
    except yaml.YAMLError as exc:
        raise ConnectionUnavailableError(
            f"Malformed {path.name}: {exc}",
            context={"path": str(path)},
            cause=exc,
        ) from exc
    if not isinstance(raw_data, dict):
        raise ConnectionUnavailableError(
            f"{path.name} must be a YAML mapping at the top level, got {type(raw_data).__name__}",
            context={"path": str(path)},
        )
    data: dict[str, Any] = {str(k): v for k, v in cast(dict[Any, Any], raw_data).items()}

    try:
        lane_str = data["lane"]
        lane = EnvironmentLane(lane_str)
        spec_name = str(data["name"])
        tools_raw: Any = data.get("tools", [])
        if not isinstance(tools_raw, list):
            raise ValueError("'tools' must be a list")
        tools_list = cast(list[Any], tools_raw)
        tools = tuple(str(t) for t in tools_list)
        max_turns = int(data.get("max_turns", 10))
        reward_type = str(data.get("reward_type", "binary"))
        multimodal = bool(data.get("multimodal", False))
        system_prompt = str(data.get("system_prompt", ""))
        cols_raw: Any = data.get("dataset_columns", [])
        if not isinstance(cols_raw, list):
            raise ValueError("'dataset_columns' must be a list")
        cols_list = cast(list[Any], cols_raw)
        dataset_columns = tuple(str(c) for c in cols_list)
    except (KeyError, ValueError, TypeError) as exc:
        raise ConnectionUnavailableError(
            f"Manifest {path.name} is missing or malformed: {exc}",
            context={"path": str(path)},
            cause=exc,
        ) from exc

    return EnvironmentSpec(
        lane=lane,
        name=spec_name,
        tools=tools,
        max_turns=max_turns,
        reward_type=reward_type,
        multimodal=multimodal,
        system_prompt=system_prompt,
        dataset_columns=dataset_columns,
    )


def _resolve_module_file(snapshot_path: Path, manifest_path: Path) -> Path:
    """Find the Python module hosting the env class.

    Precedence:
      1. ``module`` key in the manifest (filename relative to snapshot).
      2. A single ``*.py`` file at the snapshot root (excluding ``__init__``).
      3. ``<spec.name>.py``.
    """
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise ConnectionUnavailableError(
            "pyyaml is required to read carl.env.yaml",
            context={"package": "pyyaml"},
            cause=exc,
        ) from exc
    manifest_data_raw: Any = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    manifest_data: dict[str, Any]
    if isinstance(manifest_data_raw, dict):
        manifest_data = {str(k): v for k, v in cast(dict[Any, Any], manifest_data_raw).items()}
    else:
        manifest_data = {}
    declared = manifest_data.get("module")
    if isinstance(declared, str) and declared:
        candidate = snapshot_path / declared
        if candidate.exists():
            return candidate

    py_files = [
        p
        for p in sorted(snapshot_path.glob("*.py"))
        if p.name != "__init__.py"
    ]
    if len(py_files) == 1:
        return py_files[0]

    name = manifest_data.get("name")
    if isinstance(name, str) and name:
        candidate = snapshot_path / f"{name.replace('-', '_')}.py"
        if candidate.exists():
            return candidate

    raise ConnectionUnavailableError(
        "Cannot determine the environment module file from the snapshot",
        context={
            "snapshot": str(snapshot_path),
            "candidates": [p.name for p in py_files],
        },
    )


def _load_env_class(module_file: Path, spec: EnvironmentSpec) -> Type[BaseEnvironment]:
    """Dynamically import ``module_file`` and find the matching env class."""
    module_name = f"carl_studio_hub.{module_file.stem}_{abs(hash(str(module_file))):x}"
    module_spec = importlib.util.spec_from_file_location(module_name, module_file)
    if module_spec is None or module_spec.loader is None:
        raise ConnectionUnavailableError(
            f"Could not build import spec for {module_file}",
            context={"module_file": str(module_file)},
        )
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_name] = module
    try:
        module_spec.loader.exec_module(module)
    except Exception as exc:  # noqa: BLE001
        sys.modules.pop(module_name, None)
        raise ConnectionUnavailableError(
            f"Failed to import {module_file}: {exc}",
            context={"module_file": str(module_file)},
            cause=exc,
        ) from exc

    matching: list[Type[BaseEnvironment]] = []
    for _obj_name, obj in inspect.getmembers(module, inspect.isclass):
        if not issubclass(obj, BaseEnvironment) or obj is BaseEnvironment:
            continue
        if getattr(obj, "__module__", "") != module.__name__:
            continue
        obj_spec = getattr(obj, "spec", None)
        if isinstance(obj_spec, EnvironmentSpec) and obj_spec.name == spec.name:
            matching.append(obj)

    if not matching:
        raise ConnectionUnavailableError(
            f"No BaseEnvironment subclass in {module_file.name} matches spec name '{spec.name}'",
            context={"module_file": str(module_file), "spec_name": spec.name},
        )
    if len(matching) > 1:
        raise ConnectionUnavailableError(
            f"Multiple classes in {module_file.name} match spec name '{spec.name}'",
            context={
                "module_file": str(module_file),
                "matches": [c.__name__ for c in matching],
            },
        )
    return matching[0]


def _source_file_for(env_cls: type) -> str | None:
    # Try inspect first — the canonical path for regular imports.
    try:
        path = inspect.getsourcefile(env_cls)
    except (TypeError, OSError):
        path = None
    # Dynamically-loaded classes (importlib.util.spec_from_file_location)
    # don't always have their module registered in sys.modules in a way
    # that inspect can discover. Fall back to the module's __file__.
    if path is None:
        module_name = getattr(env_cls, "__module__", None)
        if isinstance(module_name, str):
            module = sys.modules.get(module_name)
            module_file = getattr(module, "__file__", None) if module is not None else None
            if isinstance(module_file, str) and module_file:
                path = module_file
    if path is None:
        return None
    candidate = Path(path)
    return str(candidate) if candidate.exists() else None


def _render_manifest(env_cls: Type[BaseEnvironment]) -> str:
    """Render a deterministic ``carl.env.yaml`` for an env class."""
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise ConnectionUnavailableError(
            "pyyaml is required to write carl.env.yaml",
            context={"package": "pyyaml"},
            cause=exc,
        ) from exc

    spec = env_cls.spec
    module_path = Path(str(_source_file_for(env_cls))).name if _source_file_for(env_cls) else ""
    version = getattr(env_cls, "version", None) or _project_version_hint()
    payload: dict[str, object] = {
        "schema_version": 1,
        "carl_studio_generated": True,
        "lane": spec.lane.value,
        "name": spec.name,
        "tools": list(spec.tools),
        "max_turns": int(spec.max_turns),
        "reward_type": spec.reward_type,
        "multimodal": bool(spec.multimodal),
        "system_prompt": spec.system_prompt,
        "dataset_columns": list(spec.dataset_columns),
        "class_name": env_cls.__name__,
        "module": module_path,
        "version": version,
    }
    return yaml.safe_dump(payload, sort_keys=True, default_flow_style=False)


def _project_version_hint() -> str:
    """Best-effort version tag for the manifest."""
    try:
        from carl_studio import __version__ as v

        return str(v)
    except Exception:  # noqa: BLE001
        return "0.0.0+local"


def _resolve_hf_token(explicit: str | None) -> str | None:
    """Resolve HF token: explicit arg > huggingface_hub.get_token > HF_TOKEN env."""
    if explicit:
        return explicit
    try:
        from huggingface_hub import get_token

        token = get_token()
        if token:
            return token
    except Exception:  # noqa: BLE001
        pass
    return os.environ.get("HF_TOKEN")


def _import_snapshot_download() -> Any:
    """Lazy-import ``huggingface_hub.snapshot_download`` with friendly error."""
    try:
        from huggingface_hub import snapshot_download as _sd  # pyright: ignore[reportUnknownVariableType]
    except ImportError as exc:
        raise ConnectionUnavailableError(
            "huggingface_hub is required for environment hub access. "
            "Install with: pip install huggingface-hub",
            context={"package": "huggingface_hub"},
            cause=exc,
        ) from exc
    return cast(Any, _sd)


def _import_hub_publish_api() -> tuple[Any, Any, Any]:
    """Lazy-import the subset of huggingface_hub used for publishing."""
    try:
        from huggingface_hub import HfApi, create_repo, upload_folder  # pyright: ignore[reportUnknownVariableType]
    except ImportError as exc:
        raise ConnectionUnavailableError(
            "huggingface_hub is required to publish environments. "
            "Install with: pip install huggingface-hub",
            context={"package": "huggingface_hub"},
            cause=exc,
        ) from exc
    return cast(Any, HfApi), cast(Any, upload_folder), cast(Any, create_repo)


__all__ = [
    "register_environment",
    "get_environment",
    "get_environment_factory",
    "list_environments",
    "is_registered",
    "clear_registry",
    "from_hub",
    "publish_to_hub",
]
