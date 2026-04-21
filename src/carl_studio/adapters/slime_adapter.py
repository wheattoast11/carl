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

from carl_core.config_requirements import (
    ConfigLocation,
    ConfigReport,
    ConfigRequirement,
    ConfigStatus,
    require_env,
    require_yaml_key,
)
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

    # -- NL-interpretable config surface (v0.16) -------------------------

    def config_requirements(
        self, carl_config: dict[str, Any] | None = None
    ) -> ConfigReport:
        """Return the NL-interpretable configuration surface for slime.

        Carl's chat agent reads this to answer natural-language setup
        questions ("what do I need to run slime?") **without ever seeing
        the values**. Each :class:`ConfigRequirement` records presence +
        (optional) 12-hex fingerprint only.

        Args:
            carl_config: Optional dict parsed from ``carl.yaml``. When
                provided, carl.yaml key checks are performed against it.
                When ``None`` (doctor-style invocation), only env-level
                checks run.

        The report groups requirements by category:

          * ``auth``       — API tokens (HF, carl.camp).
          * ``dataset``    — base_model + dataset_repo + split.
          * ``algorithm``  — method, mode, advantage_estimator.
          * ``parallelism``— tensor/pipeline/expert parallel dims.
          * ``install``    — slime / sglang / megatron reachability.
        """
        cfg: dict[str, Any] = carl_config if isinstance(carl_config, dict) else {}

        requirements: list[ConfigRequirement] = []

        # -- auth --------------------------------------------------------
        requirements.append(
            require_env(
                "HF_TOKEN",
                description=(
                    "Hugging Face user token. Slime reads models + datasets "
                    "from HF Hub; gated models (GLM, Qwen3-MoE) need this set. "
                    "Get yours at https://huggingface.co/settings/tokens."
                ),
                category="auth",
            )
        )
        requirements.append(
            require_env(
                "CARL_CAMP_TOKEN",
                description=(
                    "carl.camp bearer token. Only needed when publishing "
                    "Resonants or agent cards; BYO-compute slime runs do "
                    "not require it. Unset by default — that's OK."
                ),
                category="auth",
            )
        )
        # carl.camp HF token: only relevant when we later ship the managed
        # dispatcher. Flagged here so Carl can point at it when asked about
        # managed slime.
        requirements.append(
            require_env(
                "CARL_CAMP_HF_TOKEN",
                description=(
                    "carl.camp's own HF token (used ONLY by the managed-slime "
                    "dispatcher — v0.10+). User HF tokens must NEVER be used "
                    "for managed runs; this is a cross-system invariant."
                ),
                category="auth",
            )
        )

        # -- dataset ----------------------------------------------------
        requirements.append(
            require_yaml_key(
                "base_model",
                cfg,
                description=(
                    "Model to train. HF Hub id (e.g., Qwen/Qwen3-7B) or local "
                    "path. Slime + Megatron resolve the weights at launch."
                ),
                category="dataset",
                required=True,
            )
        )
        requirements.append(
            require_yaml_key(
                "dataset_repo",
                cfg,
                description=(
                    "Prompt dataset. HF Hub id or local parquet/jsonl path. "
                    "Becomes slime's --prompt-data."
                ),
                category="dataset",
                required=True,
            )
        )
        requirements.append(
            require_yaml_key(
                "dataset_split",
                cfg,
                description="Dataset split to draw prompts from. Defaults to 'train'.",
                category="dataset",
                required=False,
                default_value="train",
            )
        )

        # -- algorithm --------------------------------------------------
        requirements.append(
            require_yaml_key(
                "method",
                cfg,
                description=(
                    "Training method: 'grpo' (default, reinforcement), "
                    "'sft' (supervised), or 'distill' (on-policy distillation)."
                ),
                category="algorithm",
                required=False,
                default_value="grpo",
            )
        )
        requirements.append(
            require_yaml_key(
                "mode",
                cfg,
                description=(
                    "Slime deployment mode. 'sync' (default, GPU-colocated) or "
                    "'async' (decoupled — PAID: train.slime.async_disaggregated)."
                ),
                category="algorithm",
                required=False,
                default_value="sync",
            )
        )

        # -- parallelism ------------------------------------------------
        requirements.append(
            require_yaml_key(
                "slime.tensor_parallel",
                cfg,
                description=(
                    "Megatron tensor-parallel dimension. Defaults to 1 "
                    "(single GPU). For 7B-scale runs use 2-4; 100B+ MoE "
                    "typically uses 8-16."
                ),
                category="parallelism",
                required=False,
                default_value="1",
            )
        )
        requirements.append(
            require_yaml_key(
                "slime.pipeline_parallel",
                cfg,
                description=(
                    "Megatron pipeline-parallel dimension. Defaults to 1. "
                    "Needed for model weights that don't fit on a single node."
                ),
                category="parallelism",
                required=False,
                default_value="1",
            )
        )
        requirements.append(
            require_yaml_key(
                "slime.expert_parallel",
                cfg,
                description=(
                    "Megatron expert-parallel dimension (MoE only). Defaults "
                    "to 1. Set when training GLM-5, DeepSeek-V3, or Qwen3-MoE."
                ),
                category="parallelism",
                required=False,
                default_value="1",
            )
        )

        # -- install (probe) --------------------------------------------
        # We don't leak the importability check through require_env; instead,
        # we re-use missing_dependencies() for a direct truth surface.
        missing = set(self.missing_dependencies())
        for pkg, hint in (
            ("slime", "Follow https://thudm.github.io/slime/ to build from source."),
            ("sglang", "pip install 'carl-studio[slime]' pulls sglang."),
            ("megatron-lm", "Build Megatron-LM per slime's quick_start.md."),
        ):
            status = ConfigStatus.UNSET if pkg in missing else ConfigStatus.SET
            requirements.append(
                ConfigRequirement(
                    name=pkg,
                    location=ConfigLocation.EXTERNAL,
                    required=True,
                    status=status,
                    description=hint,
                    category="install",
                )
            )

        return ConfigReport(component="slime", requirements=requirements)

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
