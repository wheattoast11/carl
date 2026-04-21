"""Slime adapter.

Routes CARL training submissions to THUDM/slime — an Apache-2.0 RL
post-training framework wiring Megatron-LM + SGLang. The adapter is the
only piece of this repo that knows how to spawn slime; the arg surface
lives in :mod:`carl_studio.adapters.slime_translator`.

Slime itself + Megatron-LM are user-installed (slime's quick_start.md
documents the build). CARL never silently pip-installs a multi-GB
training stack. The :mod:`carl-studio[slime]` optional extra pulls only
the thin rollout-side deps (``sglang``); everything else is surfaced as
an :class:`AdapterError` with ``code="carl.adapter.unavailable"``.

Tier gating is orthogonal. The plain adapter is ``train.slime`` (FREE).
Managed orchestration, MoE presets, and async disaggregated mode are
PAID and live in downstream CLI handlers, not here.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
from textwrap import dedent
from typing import Any, cast

from carl_core.connection import (
    ConnectionDirection,
    ConnectionKind,
    ConnectionScope,
    ConnectionSpec,
    ConnectionTransport,
    ConnectionTrust,
)

from ._common import (
    JobState,
    cancel_common,
    logs_common,
    new_run_id,
    now_iso,
    save_state,
    spawn,
    state_dir,
    status_common,
)
from .connection import TrainingConnection
from .protocol import AdapterError, BackendJob, BackendStatus
from .slime_translator import SlimeArgs, translate_config


_INSTALL_HINT = (
    "slime is not installed. Install path: "
    "(1) pip install 'carl-studio[slime]' — pulls sglang only; "
    "(2) follow https://thudm.github.io/slime/ to build slime + Megatron-LM "
    "against your CUDA/ROCm environment."
)


# ---------------------------------------------------------------------------
# Entrypoint template
# ---------------------------------------------------------------------------
#
# Slime is typically invoked as ``torchrun ... train.py --flags``. The exact
# module path varies by release; slime's docs settled on an installed
# ``slime.cli`` surface plus a handful of recipe modules under
# ``slime/recipes/``. We generate a small launcher that:
#   1. decodes a JSON config blob from an env var (same pattern as Unsloth),
#   2. imports the slime entry symbol the user selected (default:
#      ``slime.cli:main``), and
#   3. invokes it with the flattened CLI args.
#
# The launcher is self-contained — the spawned process does not need
# carl-studio in its environment.

_ENTRYPOINT_TEMPLATE = dedent(
    '''\
    """Auto-generated slime launcher (carl-studio adapter)."""
    import json
    import os
    import sys

    CFG = json.loads(os.environ["CARL_SLIME_CONFIG"])
    entry = CFG.get("entry", "slime.cli:main")

    module_name, _, attr = entry.partition(":")
    if not attr:
        attr = "main"

    try:
        module = __import__(module_name, fromlist=[attr])
    except ImportError as exc:
        print(f"slime is not importable in this env ({entry}): {exc}", file=sys.stderr)
        sys.exit(2)

    fn = getattr(module, attr, None)
    if fn is None:
        print(f"slime entry {entry!r} has no attribute {attr!r}", file=sys.stderr)
        sys.exit(3)

    sys.argv = [module_name, *CFG["argv"]]
    rc = fn() or 0
    sys.exit(int(rc))
    '''
)


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class SlimeAdapter(TrainingConnection):
    """Adapter that launches THUDM/slime via a generated Python entrypoint.

    Availability probes are staged:
      1. ``slime`` importable?
      2. ``sglang`` importable?
      3. ``megatron`` importable? (Megatron-LM ships as ``megatron.core``
         modern and ``megatron`` legacy; we accept either.)

    Any of the three missing → :meth:`available` returns False and
    :meth:`submit` raises :class:`AdapterError`
    ``code="carl.adapter.unavailable"`` with the per-dep install hint.
    """

    spec = ConnectionSpec(
        name="carl.training.slime",
        scope=ConnectionScope.THREE_P,
        kind=ConnectionKind.TRAINING,
        direction=ConnectionDirection.EGRESS,
        transport=ConnectionTransport.SUBPROCESS,
        trust=ConnectionTrust.PUBLIC,
    )

    name = "slime"
    _INSTALL_HINT = _INSTALL_HINT

    # -- availability ----------------------------------------------------

    def available(self) -> bool:
        """Return True iff slime, sglang, and megatron are all importable.

        Never raises. Uses ``importlib.util.find_spec`` (no side effects).
        """
        try:
            if importlib.util.find_spec("slime") is None:
                return False
            if importlib.util.find_spec("sglang") is None:
                return False
            if (
                importlib.util.find_spec("megatron") is None
                and importlib.util.find_spec("megatron.core") is None
            ):
                return False
        except (ImportError, ValueError):
            return False
        return True

    def missing_dependencies(self) -> list[str]:
        """Return the list of missing upstream package names, empty when ready.

        Useful for CLI dry-run output: callers can render the exact
        per-dep state instead of just "unavailable".
        """
        missing: list[str] = []
        for mod in ("slime", "sglang"):
            try:
                if importlib.util.find_spec(mod) is None:
                    missing.append(mod)
            except (ImportError, ValueError):
                missing.append(mod)
        try:
            if (
                importlib.util.find_spec("megatron") is None
                and importlib.util.find_spec("megatron.core") is None
            ):
                missing.append("megatron-lm")
        except (ImportError, ValueError):
            missing.append("megatron-lm")
        return missing

    # -- translation (public helper for --dry-run) -----------------------

    def translate(self, carl_config: dict[str, Any]) -> SlimeArgs:
        """Public helper: translate carl.yaml → slime args without submission."""
        return translate_config(carl_config)

    # -- submit ----------------------------------------------------------

    def submit(self, carl_config: dict[str, Any]) -> BackendJob:
        translated = translate_config(carl_config)

        if not self.available():
            raise AdapterError(
                f"slime backend is not available on this machine. "
                f"Missing: {', '.join(self.missing_dependencies()) or 'unknown'}. "
                f"{self._INSTALL_HINT}",
                code="carl.adapter.unavailable",
                context={
                    "backend": self.name,
                    "missing": self.missing_dependencies(),
                    "install_hint": self._INSTALL_HINT,
                },
            )

        run_id = new_run_id("slime")
        run_dir = state_dir(self.name) / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        script_path = run_dir / "train.py"
        log_path = run_dir / "run.log"
        script_path.write_text(_ENTRYPOINT_TEMPLATE, encoding="utf-8")

        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        entry = resolve_entry(carl_config)
        launcher_cfg = {"entry": entry, "argv": translated.to_cli_args()}
        env["CARL_SLIME_CONFIG"] = json.dumps(launcher_cfg)

        cmd = resolve_launch_cmd(carl_config, script_path=str(script_path))

        pid = spawn(cmd, log_path=log_path, env=env)

        state = JobState(
            run_id=run_id,
            backend=self.name,
            status=BackendStatus.RUNNING,
            submitted_at=now_iso(),
            pid=pid,
            log_path=str(log_path),
            config=dict(carl_config),
            raw={
                "translated": translated.to_dict(),
                "script_path": str(script_path),
                "entry": entry,
                "cmd": cmd,
            },
        )
        save_state(state)
        return state.to_job()

    def status(self, run_id: str) -> BackendJob:
        return status_common(self.name, run_id)

    def logs(self, run_id: str, *, tail: int = 100) -> list[str]:
        return logs_common(self.name, run_id, tail=tail)

    def cancel(self, run_id: str) -> bool:
        return cancel_common(self.name, run_id)


# ---------------------------------------------------------------------------
# Helpers (internal)
# ---------------------------------------------------------------------------


def _slime_block(carl_config: dict[str, Any]) -> dict[str, Any]:
    """Normalize the ``carl_config['slime']`` sub-block to a typed dict.

    Returns an empty dict when the block is missing or not a mapping.
    Never raises.
    """
    raw = carl_config.get("slime")
    if isinstance(raw, dict):
        return cast(dict[str, Any], raw)
    return {}


def resolve_entry(carl_config: dict[str, Any]) -> str:
    """Resolve slime's Python entry point from ``carl_config``.

    Order:
      1. Explicit ``slime.entry`` override.
      2. ``SLIME_ENTRY`` env var (useful in CI).
      3. Default: ``slime.cli:main``.
    """
    block = _slime_block(carl_config)
    override = block.get("entry")
    if isinstance(override, str) and override.strip():
        return override.strip()
    env_override = os.environ.get("SLIME_ENTRY")
    if env_override and env_override.strip():
        return env_override.strip()
    return "slime.cli:main"


def resolve_launch_cmd(carl_config: dict[str, Any], *, script_path: str) -> list[str]:
    """Resolve the subprocess argv for spawning the slime launcher.

    By default we use ``torchrun`` when it is on PATH (single-node
    convenience — slime needs the torch.distributed init even for small
    runs). When torchrun isn't available or the caller set
    ``slime.launcher == "python"``, we fall back to the current Python
    interpreter.

    Callers can override entirely by passing ``slime.launcher_cmd`` as
    a list of strings — that wins over everything.
    """
    block = _slime_block(carl_config)

    raw_cmd = block.get("launcher_cmd")
    if isinstance(raw_cmd, list):
        cmd_items = cast(list[Any], raw_cmd)
        if cmd_items:
            return [str(v) for v in cmd_items] + [script_path]

    launcher = "torchrun"
    raw = block.get("launcher")
    if isinstance(raw, str) and raw.strip():
        launcher = raw.strip()

    if launcher == "python":
        return [sys.executable, script_path]

    if launcher == "torchrun":
        resolved = shutil.which("torchrun")
        if resolved is None:
            # Graceful fallback — surface a clear hint rather than failing
            # on a cryptic FileNotFoundError.
            return [sys.executable, script_path]
        try:
            nproc = int(block.get("nproc_per_node", 1))
        except (TypeError, ValueError):
            nproc = 1
        return [resolved, f"--nproc_per_node={max(1, nproc)}", script_path]

    # Any other launcher string — e.g. "accelerate launch" — is treated
    # as a whitespace-separated prefix. Keeps the simple cases simple.
    prefix = launcher.split()
    return [*prefix, script_path]


__all__ = ["SlimeAdapter", "resolve_entry", "resolve_launch_cmd"]
